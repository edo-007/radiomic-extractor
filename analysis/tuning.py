from __future__ import annotations

import json
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from math import sqrt
from typing import Any

import numpy as np
import pandas as pd
from joblib import Memory
from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid,
    RandomizedSearchCV,
    StratifiedGroupKFold,
)

from .benchmark import METRICS, _continuous_scores, _fold_metrics, iter_cv_splits
from .config import AnalysisConfig, CrossValidationConfig
from .data import AnalysisDataset
from .models import build_estimator
from .preprocessing import build_analysis_pipeline, selected_original_feature_indices
from .tuning_config import TuningConfig


TuningProgressCallback = Callable[[str, str, int, int], None]


@dataclass
class TuningResult:
    summary: pd.DataFrame
    outer_folds: pd.DataFrame
    search_results: pd.DataFrame
    final_parameters: dict[str, dict[str, Any]]
    selected_features: pd.DataFrame
    final_models: dict[str, object]


def run_nested_tuning(
    dataset: AnalysisDataset,
    analysis_config: AnalysisConfig,
    tuning_config: TuningConfig,
    progress_callback: TuningProgressCallback | None = None,
) -> TuningResult:
    """Esegue tuning interno e valutazione esterna senza contaminare i fold."""
    enabled_models = tuning_config.enabled_models()
    outer_config = CrossValidationConfig(
        method="repeated_stratified_kfold",
        n_splits=tuning_config.outer_cv.n_splits,
        n_repeats=tuning_config.outer_cv.n_repeats,
        random_state=tuning_config.random_state,
    )
    outer_splits = iter_cv_splits(dataset, outer_config)
    total_searches = len(enabled_models) * (len(outer_splits) + 1)
    completed_searches = 0
    outer_rows: list[dict[str, Any]] = []
    search_frames: list[pd.DataFrame] = []
    final_parameters: dict[str, dict[str, Any]] = {}
    selected_feature_rows: list[dict[str, Any]] = []
    final_models: dict[str, object] = {}

    with tempfile.TemporaryDirectory(prefix="radiomic-tuning-cache-") as cache_dir:
        memory = Memory(cache_dir, verbose=0)
        for model_name, model_tuning in enabled_models.items():
            parameter_space = {
                **tuning_config.preprocessing_parameters,
                **model_tuning.parameters,
            }

            for split in outer_splits:
                search = _build_search(
                    model_name=model_name,
                    parameter_space=parameter_space,
                    analysis_config=analysis_config,
                    tuning_config=tuning_config,
                    random_state=(
                        tuning_config.random_state
                        + (split.repeat - 1) * tuning_config.outer_cv.n_splits
                        + split.fold
                    ),
                    memory=memory,
                )
                X_train = dataset.X.iloc[split.train_indices]
                X_test = dataset.X.iloc[split.test_indices]
                y_train = dataset.y.iloc[split.train_indices]
                y_test = dataset.y.iloc[split.test_indices]
                groups_train = dataset.groups.iloc[split.train_indices]

                try:
                    search.fit(X_train, y_train, groups=groups_train)
                    predictions = search.best_estimator_.predict(X_test)
                    scores = _continuous_scores(search.best_estimator_, X_test)
                except Exception as error:
                    raise RuntimeError(
                        f"Tuning fallito per '{model_name}', ripetizione "
                        f"{split.repeat}, outer fold {split.fold}: {error}"
                    ) from error

                outer_rows.append(
                    {
                        "model": model_name,
                        "repeat": split.repeat,
                        "outer_fold": split.fold,
                        "train_rows": len(split.train_indices),
                        "test_rows": len(split.test_indices),
                        "best_inner_score": float(search.best_score_),
                        "best_parameters": json.dumps(
                            _json_safe(search.best_params_),
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        **_fold_metrics(y_test, predictions, scores),
                    }
                )
                search_frames.append(
                    _search_results_frame(
                        search,
                        model_name=model_name,
                        phase="outer",
                        repeat=split.repeat,
                        outer_fold=split.fold,
                    )
                )
                completed_searches += 1
                if progress_callback is not None:
                    progress_callback(
                        model_name,
                        f"outer {split.repeat}/{tuning_config.outer_cv.n_repeats}, "
                        f"fold {split.fold}/{tuning_config.outer_cv.n_splits}",
                        completed_searches,
                        total_searches,
                    )

            final_search = _build_search(
                model_name=model_name,
                parameter_space=parameter_space,
                analysis_config=analysis_config,
                tuning_config=tuning_config,
                random_state=tuning_config.random_state + 100_000,
                memory=memory,
            )
            try:
                final_search.fit(dataset.X, dataset.y, groups=dataset.groups)
            except Exception as error:
                raise RuntimeError(
                    f"Tuning finale sull'intero dataset fallito per '{model_name}': {error}"
                ) from error

            final_parameters[model_name] = {
                "selection_metric": tuning_config.refit,
                "inner_cv_score": float(final_search.best_score_),
                "inner_cv_metrics": {
                    metric: float(
                        final_search.cv_results_[f"mean_test_{metric}"][final_search.best_index_]
                    )
                    for metric in tuning_config.score_names()
                },
                "parameters": _json_safe(final_search.best_params_),
            }
            selected_indices = selected_original_feature_indices(
                final_search.best_estimator_,
                dataset.X.shape[1],
            )
            # La cache è temporanea e non deve finire nel modello serializzato.
            final_search.best_estimator_.memory = None
            final_models[model_name] = final_search.best_estimator_
            for rank, feature_index in enumerate(selected_indices, start=1):
                selected_feature_rows.append(
                    {
                        "model": model_name,
                        "rank": rank,
                        "feature": str(dataset.X.columns[feature_index]),
                    }
                )
            search_frames.append(
                _search_results_frame(
                    final_search,
                    model_name=model_name,
                    phase="final",
                    repeat=0,
                    outer_fold=0,
                )
            )
            completed_searches += 1
            if progress_callback is not None:
                progress_callback(
                    model_name,
                    "modello finale",
                    completed_searches,
                    total_searches,
                )

    outer_folds = pd.DataFrame(outer_rows)
    return TuningResult(
        summary=_summarise_outer_folds(outer_folds, tuning_config.refit),
        outer_folds=outer_folds,
        search_results=pd.concat(search_frames, ignore_index=True),
        final_parameters=final_parameters,
        selected_features=pd.DataFrame(selected_feature_rows),
        final_models=final_models,
    )


def _build_search(
    *,
    model_name: str,
    parameter_space: dict[str, list[Any]],
    analysis_config: AnalysisConfig,
    tuning_config: TuningConfig,
    random_state: int,
    memory: Memory | None,
):
    base_model_config = getattr(analysis_config.models, model_name)
    estimator = build_estimator(
        model_name,
        base_model_config,
        random_state=random_state,
    )
    pipeline = build_analysis_pipeline(
        estimator,
        analysis_config.preprocessing,
        random_state=random_state,
        memory=memory,
    )
    unknown_parameters = sorted(set(parameter_space) - set(pipeline.get_params(deep=True)))
    if unknown_parameters:
        raise ValueError(
            f"Parametri di tuning non validi per '{model_name}': "
            + ", ".join(unknown_parameters)
        )

    inner_cv = StratifiedGroupKFold(
        n_splits=tuning_config.inner_cv.n_splits,
        shuffle=True,
        random_state=random_state,
    )
    common_options = {
        "estimator": pipeline,
        "scoring": tuning_config.score_names(),
        "cv": inner_cv,
        "n_jobs": tuning_config.n_jobs,
        "refit": tuning_config.refit,
        "error_score": "raise",
        "return_train_score": False,
    }
    if tuning_config.strategy == "grid":
        return GridSearchCV(param_grid=parameter_space, **common_options)

    available_combinations = len(ParameterGrid(parameter_space))
    return RandomizedSearchCV(
        param_distributions=parameter_space,
        n_iter=min(tuning_config.n_iter, available_combinations),
        random_state=random_state,
        **common_options,
    )


def _search_results_frame(
    search,
    *,
    model_name: str,
    phase: str,
    repeat: int,
    outer_fold: int,
) -> pd.DataFrame:
    results = pd.DataFrame(search.cv_results_)
    useful_columns = [
        column
        for column in results.columns
        if column.startswith("param_")
        or column.startswith(("rank_test_", "mean_test_", "std_test_"))
    ]
    results = results.loc[:, useful_columns].copy()
    results.insert(0, "outer_fold", outer_fold)
    results.insert(0, "repeat", repeat)
    results.insert(0, "phase", phase)
    results.insert(0, "model", model_name)
    return results


def _summarise_outer_folds(outer_folds: pd.DataFrame, selection_metric: str = "roc_auc") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name, model_folds in outer_folds.groupby("model", sort=False):
        row: dict[str, Any] = {"model": model_name, "n_outer_folds": len(model_folds)}
        for metric in METRICS:
            mean = float(model_folds[metric].mean())
            standard_deviation = float(model_folds[metric].std(ddof=1))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = standard_deviation
            row[f"{metric}_ci95"] = (
                1.96 * standard_deviation / sqrt(len(model_folds))
                if len(model_folds) > 1
                else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(f"{selection_metric}_mean", ascending=False).reset_index(
        drop=True
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
