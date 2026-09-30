from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import AnalysisConfig


@dataclass(frozen=True)
class AnalysisDataset:
    input_csv: Path
    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series
    positive_label: str
    negative_label: str
    ignored_non_numeric_columns: tuple[str, ...]
    coerced_missing_values: dict[str, int]
    data_mode: str = "radiomic"
    clinical_input_csv: Path | None = None
    excluded_unmatched_patients: int = 0
    unused_clinical_patients: int = 0

    @property
    def patient_count(self) -> int:
        return int(self.groups.nunique())

    @property
    def duplicated_patient_rows(self) -> int:
        return int(self.groups.duplicated(keep=False).sum())


def _canonical_label(value: object) -> str:
    return str(value).strip().casefold()


def _merge_clinical_data(
    radiomic: pd.DataFrame,
    config: AnalysisConfig,
) -> tuple[pd.DataFrame, Path, int, int]:
    from .clinical import normalise_patient_id

    clinical_path = config.clinical_data.input_csv
    if clinical_path is None:
        raise ValueError(
            "Il dataset clinico non e' configurato. Imposta clinical_data.input_csv."
        )
    clinical_path = clinical_path.expanduser().resolve()
    if not clinical_path.is_file():
        raise FileNotFoundError(
            f"CSV clinico preprocessato non trovato: {clinical_path}. "
            "Crealo prima con: python cli.py clinical"
        )

    clinical = pd.read_csv(
        clinical_path,
        sep=config.clinical_data.separator,
        low_memory=False,
    )
    clinical_id = config.clinical_data.id_column
    if clinical_id not in clinical:
        raise ValueError(
            f"Colonna ID clinica mancante nel CSV: {clinical_id}"
        )
    clinical_feature_columns = [
        column for column in clinical.columns if column != clinical_id
    ]
    if not clinical_feature_columns:
        raise ValueError("Il CSV clinico non contiene feature")
    if config.target_column in clinical_feature_columns:
        raise ValueError(
            f"Il CSV clinico non deve contenere il target '{config.target_column}': "
            "ID e target vengono sempre presi dal CSV radiomico selezionato"
        )

    overlapping = sorted(
        set(clinical_feature_columns) & set(radiomic.columns)
    )
    if overlapping:
        raise ValueError(
            "Feature con lo stesso nome nei CSV clinico e radiomico: "
            + ", ".join(overlapping[:20])
        )

    radiomic = radiomic.copy()
    clinical = clinical.copy()
    radiomic["__patient_key"] = radiomic[config.id_column].map(
        normalise_patient_id
    )
    clinical["__patient_key"] = clinical[clinical_id].map(normalise_patient_id)
    if radiomic["__patient_key"].eq("").any():
        raise ValueError(f"La colonna '{config.id_column}' contiene ID vuoti")
    if clinical["__patient_key"].eq("").any():
        raise ValueError(f"La colonna clinica '{clinical_id}' contiene ID vuoti")

    if clinical["__patient_key"].duplicated().any():
        duplicates = clinical.loc[
            clinical["__patient_key"].duplicated(keep=False), clinical_id
        ].astype(str).unique()
        raise ValueError(
            "Il CSV clinico contiene piu' righe per lo stesso paziente: "
            + ", ".join(duplicates[:10])
        )

    radiomic_identity = radiomic[
        ["__patient_key", config.id_column]
    ].drop_duplicates()
    collisions = radiomic_identity.groupby("__patient_key")[config.id_column].nunique()
    if (collisions > 1).any():
        raise ValueError(
            "La normalizzazione crea collisioni tra ID radiomici distinti"
        )

    target_counts = radiomic.groupby("__patient_key")[config.target_column].nunique()
    if (target_counts > 1).any():
        raise ValueError("Uno stesso paziente ha target radiomici discordanti")

    radiomic_keys = set(radiomic["__patient_key"])
    clinical_keys = set(clinical["__patient_key"])
    matched_keys = radiomic_keys & clinical_keys
    if not matched_keys:
        raise ValueError("Nessun paziente in comune tra CSV clinico e radiomico")
    excluded = len(radiomic_keys - matched_keys)
    unused_clinical = len(clinical_keys - radiomic_keys)

    if config.dataset_mode == "radiomic":
        matched_radiomic = radiomic.loc[
            radiomic["__patient_key"].isin(matched_keys)
        ].copy()
        return (
            matched_radiomic.drop(columns="__patient_key"),
            clinical_path,
            excluded,
            unused_clinical,
        )

    clinical_predictors = clinical.drop(columns=clinical_id)
    if config.dataset_mode == "clinical":
        base = radiomic[
            ["__patient_key", config.id_column, config.target_column]
        ].drop_duplicates("__patient_key")
    else:
        base = radiomic
    merged = base.merge(
        clinical_predictors,
        on="__patient_key",
        how="inner",
        validate="many_to_one",
    )
    return (
        merged.drop(columns="__patient_key"),
        clinical_path,
        excluded,
        unused_clinical,
    )


def load_analysis_dataset(
    config: AnalysisConfig,
    input_csv: str | Path,
) -> AnalysisDataset:
    csv_path = Path(input_csv).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV delle feature non trovato: {csv_path}")

    frame = pd.read_csv(csv_path, sep=config.separator, low_memory=False)
    required_columns = [config.id_column, config.target_column]
    missing_columns = [column for column in required_columns if column not in frame]
    if missing_columns:
        raise ValueError(
            "Colonne obbligatorie mancanti nel CSV: " + ", ".join(missing_columns)
        )

    clinical_path = None
    excluded_unmatched_patients = 0
    unused_clinical_patients = 0
    use_clinical_file = (
        config.dataset_mode != "radiomic"
        or config.clinical_data.restrict_radiomic_to_matched_patients
    )
    if use_clinical_file:
        (
            frame,
            clinical_path,
            excluded_unmatched_patients,
            unused_clinical_patients,
        ) = _merge_clinical_data(frame, config)

    target_text = frame[config.target_column].astype("string").str.strip()
    valid_target = target_text.notna() & target_text.ne("")
    if not valid_target.all():
        frame = frame.loc[valid_target].copy()
        target_text = target_text.loc[valid_target]
    if frame.empty:
        raise ValueError("Il CSV non contiene righe con un target valido")

    canonical_target = target_text.map(_canonical_label)
    labels_by_key: dict[str, str] = {}
    for key, label in zip(canonical_target, target_text, strict=True):
        labels_by_key.setdefault(key, str(label))
    if len(labels_by_key) != 2:
        labels = ", ".join(sorted(labels_by_key.values())) or "nessuna"
        raise ValueError(
            "L'analisi di classificazione richiede esattamente due classi; "
            f"trovate: {labels}"
        )

    positive_key = _canonical_label(config.positive_class)
    if positive_key not in labels_by_key:
        raise ValueError(
            f"La classe positiva '{config.positive_class}' non e' presente. "
            "Classi disponibili: " + ", ".join(sorted(labels_by_key.values()))
        )
    negative_key = next(key for key in labels_by_key if key != positive_key)
    y = canonical_target.eq(positive_key).astype(int)
    y.name = config.target_column

    groups = frame[config.id_column].astype("string").str.strip()
    if groups.isna().any() or groups.eq("").any():
        raise ValueError(
            f"La colonna identificativa '{config.id_column}' contiene valori vuoti"
        )
    groups.name = config.id_column

    patient_targets = pd.DataFrame({"group": groups, "target": y}).groupby(
        "group", sort=False
    )["target"].nunique()
    inconsistent = patient_targets[patient_targets > 1]
    if not inconsistent.empty:
        raise ValueError(
            "Uno stesso paziente compare con target diversi: "
            + ", ".join(map(str, inconsistent.index[:10]))
        )

    excluded = {
        config.id_column,
        config.target_column,
        *config.exclude_columns,
    }
    candidate_columns = [
        column
        for column in frame.columns
        if column not in excluded
        and (config.include_metadata or not column.startswith(config.metadata_prefix))
    ]
    if not candidate_columns:
        raise ValueError("Nessuna colonna candidata all'analisi")

    numeric_features: dict[str, pd.Series] = {}
    ignored_columns: list[str] = []
    coerced_missing_values: dict[str, int] = {}
    for column in candidate_columns:
        original = frame[column]
        numeric = pd.to_numeric(original, errors="coerce")
        original_present = original.notna() & original.astype("string").str.strip().ne("")
        coerced = int((original_present & numeric.isna()).sum())
        if numeric.notna().sum() == 0:
            ignored_columns.append(column)
            continue
        if coerced:
            coerced_missing_values[column] = coerced
        numeric_features[column] = numeric.astype(float)

    if not numeric_features:
        raise ValueError("Nessuna feature numerica trovata nel CSV")

    X = pd.DataFrame(numeric_features, index=frame.index).replace(
        [np.inf, -np.inf], np.nan
    )
    return AnalysisDataset(
        input_csv=csv_path,
        X=X.reset_index(drop=True),
        y=y.reset_index(drop=True),
        groups=groups.reset_index(drop=True),
        positive_label=labels_by_key[positive_key],
        negative_label=labels_by_key[negative_key],
        ignored_non_numeric_columns=tuple(ignored_columns),
        coerced_missing_values=coerced_missing_values,
        data_mode=config.dataset_mode,
        clinical_input_csv=clinical_path,
        excluded_unmatched_patients=excluded_unmatched_patients,
        unused_clinical_patients=unused_clinical_patients,
    )
