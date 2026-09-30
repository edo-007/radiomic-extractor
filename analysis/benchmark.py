from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold

from .config import AnalysisConfig, CrossValidationConfig
from .data import AnalysisDataset
from .preprocessing import build_analysis_pipeline


METRICS = (
    "roc_auc",
    "balanced_accuracy",
    "accuracy",
    "sensitivity",
    "specificity",
    "f1",
)


@dataclass(frozen=True)
class FoldSplit:
    repeat: int
    fold: int
    train_indices: np.ndarray
    test_indices: np.ndarray


@dataclass(frozen=True)
class BenchmarkResult:
    summary: pd.DataFrame
    folds: pd.DataFrame


def iter_cv_splits(
    dataset: AnalysisDataset,
    config: CrossValidationConfig,
) -> list[FoldSplit]:
    patient_labels = pd.DataFrame(
        {"patient": dataset.groups, "target": dataset.y}
    ).drop_duplicates("patient")
    groups_per_class = patient_labels.groupby("target")["patient"].nunique()
    if config.method == "leave_one_out":
        if len(groups_per_class) != 2 or int(groups_per_class.min()) < 2:
            counts = ", ".join(
                f"classe {label}={count} pazienti"
                for label, count in groups_per_class.items()
            )
            raise ValueError(
                "La LOO richiede almeno 2 pazienti per classe, perche' ogni "
                f"paziente viene escluso a turno; disponibili: {counts}"
            )
        splitter = LeaveOneGroupOut()
        return [
            FoldSplit(
                repeat=1,
                fold=fold,
                train_indices=train_indices,
                test_indices=test_indices,
            )
            for fold, (train_indices, test_indices) in enumerate(
                splitter.split(dataset.X, dataset.y, groups=dataset.groups),
                start=1,
            )
        ]

    if len(groups_per_class) != 2 or int(groups_per_class.min()) < config.n_splits:
        counts = ", ".join(
            f"classe {label}={count} pazienti"
            for label, count in groups_per_class.items()
        )
        raise ValueError(
            f"n_splits={config.n_splits} richiede almeno {config.n_splits} "
            f"pazienti per classe; disponibili: {counts}"
        )

    repeat_count = (
        config.n_repeats
        if config.method == "repeated_stratified_kfold"
        else 1
    )
    splits: list[FoldSplit] = []
    for repeat in range(repeat_count):
        splitter = StratifiedGroupKFold(
            n_splits=config.n_splits,
            shuffle=True,
            random_state=config.random_state + repeat,
        )
        for fold, (train_indices, test_indices) in enumerate(
            splitter.split(dataset.X, dataset.y, groups=dataset.groups),
            start=1,
        ):
            splits.append(
                FoldSplit(
                    repeat=repeat + 1,
                    fold=fold,
                    train_indices=train_indices,
                    test_indices=test_indices,
                )
            )
    return splits


def _continuous_scores(pipeline, X) -> np.ndarray:
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(X)
        classes = list(pipeline.classes_)
        if 1 not in classes:
            raise ValueError("Il modello non espone la classe positiva 1")
        return np.asarray(probabilities[:, classes.index(1)], dtype=float)
    if hasattr(pipeline, "decision_function"):
        scores = np.asarray(pipeline.decision_function(X), dtype=float)
        if scores.ndim == 2:
            classes = list(pipeline.classes_)
            scores = scores[:, classes.index(1)]
        return scores
    return np.asarray(pipeline.predict(X), dtype=float)


def _fold_metrics(y_true, y_pred, y_score) -> dict[str, float]:
    roc_auc = (
        float(roc_auc_score(y_true, y_score))
        if len(np.unique(y_true)) == 2
        else float("nan")
    )
    return {
        "roc_auc": roc_auc,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


def _summarise_folds(folds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name, model_folds in folds.groupby("model", sort=False):
        row: dict[str, object] = {
            "model": model_name,
            "n_folds": len(model_folds),
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = float(model_folds[metric].mean())
            row[f"{metric}_std"] = float(model_folds[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        "roc_auc_mean", ascending=False
    ).reset_index(drop=True)


def run_benchmark(
    dataset: AnalysisDataset,
    config: AnalysisConfig,
    estimators: dict[str, object],
) -> BenchmarkResult:
    splits = iter_cv_splits(dataset, config.cross_validation)
    fold_rows: list[dict[str, object]] = []
    for model_name, estimator in estimators.items():
        loo_true: list[int] = []
        loo_predictions: list[int] = []
        loo_scores: list[float] = []
        loo_train_sizes: list[int] = []
        for split in splits:
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
                predictions = pipeline.predict(X_test)
                scores = _continuous_scores(pipeline, X_test)
            except Exception as error:
                raise RuntimeError(
                    f"Errore nel modello '{model_name}', ripetizione "
                    f"{split.repeat}, fold {split.fold}: {error}"
                ) from error
            if config.cross_validation.method == "leave_one_out":
                patient_targets = np.unique(np.asarray(y_test, dtype=int))
                if len(patient_targets) != 1:
                    raise ValueError(
                        "Il validation fold LOO contiene target discordanti "
                        "per lo stesso paziente"
                    )
                loo_true.append(int(patient_targets[0]))
                loo_predictions.append(
                    int(np.mean(np.asarray(predictions, dtype=float)) >= 0.5)
                )
                loo_scores.append(float(np.mean(np.asarray(scores, dtype=float))))
                loo_train_sizes.append(len(split.train_indices))
            else:
                fold_rows.append(
                    {
                        "model": model_name,
                        "repeat": split.repeat,
                        "fold": split.fold,
                        "train_rows": len(split.train_indices),
                        "test_rows": len(split.test_indices),
                        **_fold_metrics(y_test, predictions, scores),
                    }
                )

        if config.cross_validation.method == "leave_one_out":
            fold_rows.append(
                {
                    "model": model_name,
                    "repeat": 1,
                    "fold": 0,
                    "train_rows": int(np.mean(loo_train_sizes)),
                    "test_rows": len(loo_true),
                    **_fold_metrics(
                        np.asarray(loo_true),
                        np.asarray(loo_predictions),
                        np.asarray(loo_scores),
                    ),
                }
            )

    folds = pd.DataFrame(fold_rows)
    summary = _summarise_folds(folds)
    if config.cross_validation.method == "leave_one_out":
        summary["n_folds"] = len(splits)
    return BenchmarkResult(summary=summary, folds=folds)
