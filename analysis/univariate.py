from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from .config import UnivariateConfig
from .data import AnalysisDataset


def _fdr_benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    adjusted = pd.Series(np.nan, index=p_values.index, dtype=float)
    finite = p_values[np.isfinite(p_values.to_numpy(dtype=float))]
    if finite.empty:
        return adjusted

    ordered = finite.sort_values()
    count = len(ordered)
    ranks = np.arange(1, count + 1, dtype=float)
    raw_adjusted = ordered.to_numpy(dtype=float) * count / ranks
    monotonic = np.minimum.accumulate(raw_adjusted[::-1])[::-1]
    adjusted.loc[ordered.index] = np.clip(monotonic, 0.0, 1.0)
    return adjusted


def _cohens_d(positive: np.ndarray, negative: np.ndarray) -> float:
    if len(positive) < 2 or len(negative) < 2:
        return float("nan")
    degrees_of_freedom = len(positive) + len(negative) - 2
    pooled_variance = (
        (len(positive) - 1) * np.var(positive, ddof=1)
        + (len(negative) - 1) * np.var(negative, ddof=1)
    ) / degrees_of_freedom
    if pooled_variance <= 0.0 or not np.isfinite(pooled_variance):
        return float("nan")
    return float((np.mean(positive) - np.mean(negative)) / np.sqrt(pooled_variance))


def run_univariate_analysis(
    dataset: AnalysisDataset,
    config: UnivariateConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature_name in dataset.X.columns:
        feature = dataset.X[feature_name]
        valid = feature.notna()
        values = feature.loc[valid].to_numpy(dtype=float)
        labels = dataset.y.loc[valid].to_numpy(dtype=int)
        positive = values[labels == 1]
        negative = values[labels == 0]

        p_value = float("nan")
        effect_size = float("nan")
        effect_name = "rank_biserial"
        if len(positive) > 0 and len(negative) > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                if config.test == "mannwhitney":
                    statistic, p_value = stats.mannwhitneyu(
                        positive,
                        negative,
                        alternative="two-sided",
                    )
                    effect_size = float(
                        (2.0 * statistic) / (len(positive) * len(negative)) - 1.0
                    )
                else:
                    _statistic, p_value = stats.ttest_ind(
                        positive,
                        negative,
                        equal_var=False,
                        nan_policy="omit",
                    )
                    effect_name = "cohens_d"
                    effect_size = _cohens_d(positive, negative)

        auc = float("nan")
        if len(np.unique(labels)) == 2 and len(np.unique(values)) > 1:
            auc = float(roc_auc_score(labels, values))

        rows.append(
            {
                "feature": feature_name,
                "n_negative": len(negative),
                "n_positive": len(positive),
                "median_negative": (
                    float(np.median(negative)) if len(negative) else float("nan")
                ),
                "median_positive": (
                    float(np.median(positive)) if len(positive) else float("nan")
                ),
                "auc": auc,
                "discriminative_auc": (
                    max(auc, 1.0 - auc) if np.isfinite(auc) else float("nan")
                ),
                "effect_name": effect_name,
                "effect_size": effect_size,
                "p_value": float(p_value),
            }
        )

    results = pd.DataFrame(rows)
    if config.correction == "fdr_bh":
        results["adjusted_p_value"] = _fdr_benjamini_hochberg(
            results["p_value"]
        )
    else:
        results["adjusted_p_value"] = results["p_value"]

    return results.sort_values(
        ["adjusted_p_value", "discriminative_auc"],
        ascending=[True, False],
        na_position="last",
    ).reset_index(drop=True)
