from __future__ import annotations

from functools import partial

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from .config import PreprocessingConfig


class NonConstantFilter(BaseEstimator, TransformerMixin):
    """Rimuove le colonne costanti usando solo il training fold."""

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        if values.ndim != 2:
            raise ValueError("Le feature devono essere una matrice bidimensionale")
        self.n_features_in_ = values.shape[1]
        variances = np.nanvar(values, axis=0)
        self.support_ = np.isfinite(variances) & (variances > 0.0)
        if not self.support_.any():
            raise ValueError("Tutte le feature sono costanti nel training fold")
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=float)
        return values[:, self.support_]

    def get_support(self):
        return self.support_


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Conserva la prima feature di ogni gruppo altamente correlato."""

    def __init__(self, threshold: float = 0.90):
        self.threshold = threshold

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        if values.ndim != 2:
            raise ValueError("Le feature devono essere una matrice bidimensionale")
        self.n_features_in_ = values.shape[1]
        if self.n_features_in_ <= 1:
            self.support_ = np.ones(self.n_features_in_, dtype=bool)
            return self

        correlation = np.abs(np.corrcoef(values, rowvar=False))
        correlation = np.nan_to_num(correlation, nan=0.0, posinf=1.0, neginf=1.0)
        support = np.ones(self.n_features_in_, dtype=bool)
        for column_index in range(1, self.n_features_in_):
            earlier_kept = np.flatnonzero(support[:column_index])
            if earlier_kept.size and np.any(
                correlation[earlier_kept, column_index] > self.threshold
            ):
                support[column_index] = False
        self.support_ = support
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=float)
        return values[:, self.support_]

    def get_support(self):
        return self.support_


class AdaptiveSelectKBest(BaseEstimator, TransformerMixin):
    """SelectKBest che adatta k alle feature rimaste nel singolo fold."""

    def __init__(
        self,
        k: int = 20,
        method: str = "f_classif",
        random_state: int = 42,
    ):
        self.k = k
        self.method = method
        self.random_state = random_state

    def fit(self, X, y):
        values = np.asarray(X, dtype=float)
        if values.shape[1] == 0:
            raise ValueError("Nessuna feature disponibile per la selezione")
        effective_k = min(self.k, values.shape[1])
        if self.method == "f_classif":
            score_func = f_classif
        elif self.method == "mutual_info":
            score_func = partial(
                mutual_info_classif,
                random_state=self.random_state,
            )
        else:
            raise ValueError(f"Metodo di selezione non supportato: {self.method}")
        self.selector_ = SelectKBest(score_func=score_func, k=effective_k)
        self.selector_.fit(values, y)
        self.n_features_in_ = values.shape[1]
        self.effective_k_ = effective_k
        return self

    def transform(self, X):
        return self.selector_.transform(np.asarray(X, dtype=float))

    def get_support(self):
        return self.selector_.get_support()


def build_analysis_pipeline(
    estimator,
    config: PreprocessingConfig,
    *,
    random_state: int,
    memory=None,
) -> Pipeline:
    steps: list[tuple[str, object]] = [
        (
            "imputer",
            SimpleImputer(
                strategy=config.impute_strategy,
                keep_empty_features=True,
            ),
        ),
    ]
    if config.remove_constant_features:
        steps.append(("constant_filter", NonConstantFilter()))
    if config.correlation_threshold is not None:
        steps.append(
            (
                "correlation_filter",
                CorrelationFilter(threshold=config.correlation_threshold),
            )
        )
    scaling = "none" if isinstance(estimator, RandomForestClassifier) else config.scaling
    if scaling == "standard":
        steps.append(("scaler", StandardScaler()))
    elif scaling == "robust":
        steps.append(("scaler", RobustScaler()))
    if config.feature_selection.enabled:
        steps.append(
            (
                "feature_selection",
                AdaptiveSelectKBest(
                    k=config.feature_selection.k,
                    method=config.feature_selection.method,
                    random_state=random_state,
                ),
            )
        )
    steps.append(("model", estimator))
    return Pipeline(steps, memory=memory)


def selected_original_feature_indices(
    fitted_pipeline: Pipeline,
    original_feature_count: int,
) -> np.ndarray:
    """Mappa le feature trasformate alle colonne originali del CSV."""
    indices = np.arange(original_feature_count)
    for step_name, transformer in fitted_pipeline.steps:
        if step_name == "model":
            break
        get_support = getattr(transformer, "get_support", None)
        if get_support is None:
            continue
        support = np.asarray(get_support(), dtype=bool)
        if len(support) != len(indices):
            raise ValueError(
                f"Il trasformatore '{step_name}' ha restituito una maschera "
                "incompatibile con le feature correnti: "
                f"maschera={len(support)}, feature={len(indices)}"
            )
        indices = indices[support]
    return indices
