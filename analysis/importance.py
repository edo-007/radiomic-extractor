from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance

from .benchmark import iter_cv_splits
from .config import AnalysisConfig
from .data import AnalysisDataset
from .preprocessing import (
    build_analysis_pipeline,
    selected_original_feature_indices,
)


ImportanceProgressCallback = Callable[[str, int, int], None]


def run_cross_validated_importance(
    dataset: AnalysisDataset,
    config: AnalysisConfig,
    estimators: dict[str, object],
    progress_callback: ImportanceProgressCallback | None = None,
) -> pd.DataFrame:
    """Permutation importance calcolata esclusivamente sui validation fold."""
    if config.cross_validation.method == "leave_one_out":
        raise ValueError(
            "Permutation importance non disponibile con LOO: il validation "
            "fold contiene un solo paziente"
        )
    splits = iter_cv_splits(dataset, config.cross_validation)
    feature_names = np.asarray(dataset.X.columns, dtype=object)
    feature_count = len(feature_names)
    repeats = config.importance.permutation_repeats
    total_folds = len(estimators) * len(splits)
    completed_folds = 0
    rows: list[dict[str, object]] = []

    for model_name, estimator in estimators.items():
        measurements = np.zeros((feature_count, len(splits) * repeats), dtype=float)
        selection_counts = np.zeros(feature_count, dtype=int)
        for split_index, split in enumerate(splits):
            pipeline = build_analysis_pipeline(
                clone(estimator),
                config.preprocessing,
                random_state=config.cross_validation.random_state,
            )
            X_train = dataset.X.iloc[split.train_indices]
            X_test = dataset.X.iloc[split.test_indices]
            y_train = dataset.y.iloc[split.train_indices]
            y_test = dataset.y.iloc[split.test_indices]
            try:
                pipeline.fit(X_train, y_train)
                selected_indices = selected_original_feature_indices(
                    pipeline,
                    feature_count,
                )
                transformed_X_test = pipeline[:-1].transform(X_test)
                importance = permutation_importance(
                    pipeline.named_steps["model"],
                    transformed_X_test,
                    y_test,
                    scoring=config.importance.scoring,
                    n_repeats=config.importance.permutation_repeats,
                    random_state=(
                        config.cross_validation.random_state + split_index
                    ),
                    n_jobs=config.importance.n_jobs,
                )
            except Exception as error:
                raise RuntimeError(
                    f"Errore nel calcolo delle importance per '{model_name}', "
                    f"ripetizione {split.repeat}, fold {split.fold}: {error}"
                ) from error

            if importance.importances.shape[0] != len(selected_indices):
                raise RuntimeError(
                    "Il numero di importance non coincide con le feature "
                    "selezionate nel fold"
                )
            measurement_slice = slice(split_index * repeats, (split_index + 1) * repeats)
            measurements[selected_indices, measurement_slice] = importance.importances
            selection_counts[selected_indices] += 1

            completed_folds += 1
            if progress_callback is not None:
                progress_callback(model_name, completed_folds, total_folds)

        for feature_index, feature_name in enumerate(feature_names):
            values = measurements[feature_index]
            rows.append(
                {
                    "model": model_name,
                    "feature": feature_name,
                    "importance_mean": float(np.mean(values)),
                    "importance_std": float(np.std(values, ddof=1)),
                    "positive_fraction": float(np.mean(values > 0.0)),
                    "selection_fraction": float(
                        selection_counts[feature_index] / len(splits)
                    ),
                    "n_measurements": len(values),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["model", "importance_mean"],
        ascending=[True, False],
    ).reset_index(drop=True)
