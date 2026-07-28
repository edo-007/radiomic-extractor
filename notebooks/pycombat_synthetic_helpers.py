from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

matplotlib_cache = Path(tempfile.gettempdir()) / "matplotlib"
matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

import matplotlib.pyplot as plt
from scipy.stats import bartlett, f_oneway, ttest_ind
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

plt.style.use("seaborn-v0_8-whitegrid")


@dataclass(frozen=True)
class SyntheticCombatData:
    Y: np.ndarray
    batch: np.ndarray
    batch_names: np.ndarray
    clinical_numeric: np.ndarray
    clinical_label: np.ndarray
    feature_names: np.ndarray
    n_signal_features: int
    metadata: pd.DataFrame
    features: pd.DataFrame


@dataclass(frozen=True)
class CombatDiagnostics:
    p_batch_before: np.ndarray
    p_batch_after: np.ndarray
    p_scale_before: np.ndarray
    p_scale_after: np.ndarray
    p_clinical_before: np.ndarray
    p_clinical_after: np.ndarray
    summary: pd.DataFrame


def find_project_root(start: Path | None = None) -> Path:
    """Find the repository root that contains the local pycombat package."""
    start_path = (start or Path(__file__)).resolve()
    candidates = [start_path, *start_path.parents, Path.cwd(), *Path.cwd().parents]

    for candidate in candidates:
        package_init = candidate / "pycombat-radiomics" / "pycombat" / "__init__.py"
        if package_init.exists():
            return candidate

        package_init = candidate / "pycombat" / "__init__.py"
        if package_init.exists() and candidate.name == "pycombat-radiomics":
            return candidate.parent

    raise FileNotFoundError(
        "Cannot find pycombat-radiomics/pycombat/__init__.py. "
        "Start Jupyter from the repository root or pass a valid project_root."
    )


def load_combat_class(project_root: Path | None = None) -> Any:
    """Import Combat from the local pycombat-radiomics folder."""
    root = find_project_root(project_root)
    pycombat_root = root / "pycombat-radiomics"

    if str(pycombat_root) not in sys.path:
        sys.path.insert(0, str(pycombat_root))

    from pycombat import Combat

    return Combat


def create_synthetic_dataset(
    random_state: int = 42,
    n_per_batch: int = 45,
    n_features: int = 40,
    n_signal_features: int = 8,
) -> SyntheticCombatData:
    """Create synthetic radiomic-like data with known batch and clinical effects."""
    rng = np.random.default_rng(random_state)
    batch_names = np.array(["Scanner_A", "Scanner_B", "Scanner_C"])
    feature_names = np.array(
        [f"feature_{idx:02d}" for idx in range(1, n_features + 1)]
    )
    batch = np.repeat(batch_names, n_per_batch)

    clinical_numeric = np.concatenate(
        [
            rng.permutation(
                np.r_[
                    np.zeros(n_per_batch // 2),
                    np.ones(n_per_batch - n_per_batch // 2),
                ]
            )
            for _ in batch_names
        ]
    )
    clinical_label = np.where(clinical_numeric == 1, "INSTABILE", "STABILE")

    baseline = rng.normal(loc=0.0, scale=0.2, size=n_features)
    clinical_effect = np.zeros(n_features)
    clinical_effect[:n_signal_features] = rng.normal(
        loc=1.0,
        scale=0.15,
        size=n_signal_features,
    )
    clinical_effect[:n_signal_features] *= rng.choice(
        [-1, 1],
        size=n_signal_features,
    )

    batch_shift = {
        "Scanner_A": rng.normal(loc=0.0, scale=0.10, size=n_features),
        "Scanner_B": rng.normal(loc=1.2, scale=0.25, size=n_features),
        "Scanner_C": rng.normal(loc=-1.0, scale=0.25, size=n_features),
    }
    batch_scale = {"Scanner_A": 1.0, "Scanner_B": 1.6, "Scanner_C": 0.65}

    Y = np.zeros((batch.size, n_features), dtype=float)
    for row_idx, batch_name in enumerate(batch):
        noise = rng.normal(
            loc=0.0,
            scale=0.65 * batch_scale[batch_name],
            size=n_features,
        )
        Y[row_idx] = (
            baseline
            + clinical_effect * clinical_numeric[row_idx]
            + batch_shift[batch_name]
            + noise
        )

    metadata = pd.DataFrame(
        {
            "patient_id": [f"SYN_{idx:03d}" for idx in range(1, batch.size + 1)],
            "batch": batch,
            "tipo": clinical_label,
        }
    )
    features = pd.DataFrame(Y, columns=feature_names)

    return SyntheticCombatData(
        Y=Y,
        batch=batch,
        batch_names=batch_names,
        clinical_numeric=clinical_numeric,
        clinical_label=clinical_label,
        feature_names=feature_names,
        n_signal_features=n_signal_features,
        metadata=metadata,
        features=features,
    )


def dataset_preview(data: SyntheticCombatData) -> pd.DataFrame:
    return pd.concat([data.metadata, data.features], axis=1).head()


def array_quality(values: np.ndarray) -> pd.Series:
    return pd.Series(
        {
            "righe": values.shape[0],
            "colonne": values.shape[1],
            "NaN": int(np.isnan(values).sum()),
            "Inf": int(np.isinf(values).sum()),
        }
    )


def apply_combat(
    data: SyntheticCombatData,
    project_root: Path | None = None,
) -> tuple[np.ndarray, Any]:
    """Run ComBat while preserving the simulated clinical effect."""
    Combat = load_combat_class(project_root)
    X = data.clinical_numeric.reshape(-1, 1)
    combat = Combat()
    Y_after = combat.fit_transform(Y=data.Y, b=data.batch, X=X)
    return Y_after, combat


def batch_location_pvalues(values: np.ndarray, data: SyntheticCombatData) -> np.ndarray:
    return np.array(
        [
            f_oneway(
                *(
                    values[data.batch == batch_name, feature_idx]
                    for batch_name in data.batch_names
                )
            )[1]
            for feature_idx in range(values.shape[1])
        ]
    )


def batch_scale_pvalues(values: np.ndarray, data: SyntheticCombatData) -> np.ndarray:
    return np.array(
        [
            bartlett(
                *(
                    values[data.batch == batch_name, feature_idx]
                    for batch_name in data.batch_names
                )
            )[1]
            for feature_idx in range(values.shape[1])
        ]
    )


def clinical_pvalues(values: np.ndarray, data: SyntheticCombatData) -> np.ndarray:
    return np.array(
        [
            ttest_ind(
                values[data.clinical_numeric == 0, feature_idx],
                values[data.clinical_numeric == 1, feature_idx],
                equal_var=False,
            )[1]
            for feature_idx in range(values.shape[1])
        ]
    )


def compute_diagnostics(
    data: SyntheticCombatData,
    Y_after: np.ndarray,
    alpha: float = 0.05,
) -> CombatDiagnostics:
    p_batch_before = batch_location_pvalues(data.Y, data)
    p_batch_after = batch_location_pvalues(Y_after, data)
    p_scale_before = batch_scale_pvalues(data.Y, data)
    p_scale_after = batch_scale_pvalues(Y_after, data)
    p_clinical_before = clinical_pvalues(data.Y, data)
    p_clinical_after = clinical_pvalues(Y_after, data)

    signal_mask = np.arange(data.Y.shape[1]) < data.n_signal_features
    summary = pd.DataFrame(
        {
            "controllo": [
                "Feature con media diversa tra batch",
                "Feature con varianza diversa tra batch",
                "Feature cliniche simulate ancora significative",
                "Feature non cliniche falsamente significative",
            ],
            "prima": [
                int((p_batch_before < alpha).sum()),
                int((p_scale_before < alpha).sum()),
                int((p_clinical_before[signal_mask] < alpha).sum()),
                int((p_clinical_before[~signal_mask] < alpha).sum()),
            ],
            "dopo": [
                int((p_batch_after < alpha).sum()),
                int((p_scale_after < alpha).sum()),
                int((p_clinical_after[signal_mask] < alpha).sum()),
                int((p_clinical_after[~signal_mask] < alpha).sum()),
            ],
        }
    )

    return CombatDiagnostics(
        p_batch_before=p_batch_before,
        p_batch_after=p_batch_after,
        p_scale_before=p_scale_before,
        p_scale_after=p_scale_after,
        p_clinical_before=p_clinical_before,
        p_clinical_after=p_clinical_after,
        summary=summary,
    )


def pca_scores(values: np.ndarray) -> np.ndarray:
    scaled = StandardScaler().fit_transform(values)
    return PCA(n_components=2, random_state=0).fit_transform(scaled)


def plot_pca_before_after(data: SyntheticCombatData, Y_after: np.ndarray) -> None:
    scores_before = pca_scores(data.Y)
    scores_after = pca_scores(Y_after)
    colors = {"Scanner_A": "#4c78a8", "Scanner_B": "#f58518", "Scanner_C": "#54a24b"}
    markers = {"STABILE": "o", "INSTABILE": "s"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, scores, title in zip(
        axes,
        [scores_before, scores_after],
        ["Prima di ComBat", "Dopo ComBat"],
        strict=True,
    ):
        for batch_name in data.batch_names:
            for label, marker in markers.items():
                mask = (data.batch == batch_name) & (data.clinical_label == label)
                ax.scatter(
                    scores[mask, 0],
                    scores[mask, 1],
                    c=colors[batch_name],
                    marker=marker,
                    s=42,
                    alpha=0.78,
                    edgecolor="white",
                    linewidth=0.5,
                    label=f"{batch_name} / {label}",
                )
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
    )
    fig.tight_layout()
    plt.show()


def plot_feature_boxplots(
    data: SyntheticCombatData,
    Y_after: np.ndarray,
    feature_to_plot: str = "feature_01",
) -> None:
    feature_idx = int(feature_to_plot.split("_")[1]) - 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, values, title in zip(
        axes,
        [data.Y, Y_after],
        ["Prima", "Dopo"],
        strict=True,
    ):
        box_data = [
            values[data.batch == batch_name, feature_idx]
            for batch_name in data.batch_names
        ]
        ax.boxplot(box_data, tick_labels=data.batch_names, patch_artist=True)
        ax.set_title(f"{title}: {feature_to_plot}")
        ax.set_xlabel("Batch")
        ax.set_ylabel("Valore feature")
        ax.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    plt.show()


def neglog10(pvalues: np.ndarray) -> np.ndarray:
    return -np.log10(np.clip(pvalues, 1e-300, 1.0))


def plot_pvalue_diagnostics(
    data: SyntheticCombatData,
    diagnostics: CombatDiagnostics,
    alpha: float = 0.05,
) -> None:
    x = np.arange(1, data.Y.shape[1] + 1)
    threshold = -np.log10(alpha)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(
        x,
        neglog10(diagnostics.p_batch_before),
        label="Prima",
        color="#d62728",
        linewidth=1.8,
    )
    axes[0].plot(
        x,
        neglog10(diagnostics.p_batch_after),
        label="Dopo",
        color="#1f77b4",
        linewidth=1.8,
    )
    axes[0].axhline(threshold, color="black", linestyle="--", linewidth=1)
    axes[0].axvspan(0.5, data.n_signal_features + 0.5, color="#dddddd", alpha=0.35)
    axes[0].set_title("Associazione feature-batch")
    axes[0].set_xlabel("Feature")
    axes[0].set_ylabel("-log10(p ANOVA batch)")
    axes[0].legend(frameon=False)

    axes[1].plot(
        x,
        neglog10(diagnostics.p_clinical_before),
        label="Prima",
        color="#d62728",
        linewidth=1.8,
    )
    axes[1].plot(
        x,
        neglog10(diagnostics.p_clinical_after),
        label="Dopo",
        color="#1f77b4",
        linewidth=1.8,
    )
    axes[1].axhline(threshold, color="black", linestyle="--", linewidth=1)
    axes[1].axvspan(0.5, data.n_signal_features + 0.5, color="#dddddd", alpha=0.35)
    axes[1].set_title("Associazione feature-clinica")
    axes[1].set_xlabel("Feature")
    axes[1].set_ylabel("-log10(p t-test tipo)")
    axes[1].legend(frameon=False)

    fig.tight_layout()
    plt.show()
