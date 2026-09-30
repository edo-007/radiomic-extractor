from __future__ import annotations

import hashlib
import json
import re
import shutil
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from .clinical_config import ClinicalConfig, load_clinical_configuration


@dataclass(frozen=True)
class ClinicalPreparationResult:
    output_csv: Path
    report_json: Path
    patients: int
    excluded_absent_diagnoses: int
    clinical_feature_count: int
    conversion_issues: dict[str, dict[str, int]]


def normalise_patient_id(value: object) -> str:
    """Crea una chiave robusta senza modificare l'ID mostrato nei CSV."""
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(
        character for character in text if not unicodedata.combining(character)
    )
    text = re.sub(r"[^A-Za-z0-9]+", " ", text).strip().upper()
    return re.sub(r"\s+", " ", text)


def _slug(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(
        character for character in text if not unicodedata.combining(character)
    )
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return text or "value"


def _clean_missing_values(frame: pd.DataFrame, tokens: list[str]) -> pd.DataFrame:
    missing = {str(token).strip().casefold() for token in tokens}
    cleaned = frame.copy()
    for column in cleaned.columns:
        values = cleaned[column].astype("string").str.strip()
        cleaned[column] = values.mask(values.str.casefold().isin(missing), pd.NA)
    return cleaned


def _required_columns(config: ClinicalConfig) -> set[str]:
    required = {
        config.patient_column,
        config.diagnosis_column,
        *config.numeric_columns,
        *config.ordinal_columns,
        *config.categorical_columns,
    }
    if config.diagnosis_date_column:
        required.add(config.diagnosis_date_column)
    if config.lymph_node_ratio.enabled:
        required.update(
            {
                config.lymph_node_ratio.positive_column,
                config.lymph_node_ratio.examined_column,
            }
        )
    return required


def _select_diagnosis(
    frame: pd.DataFrame,
    config: ClinicalConfig,
) -> tuple[pd.DataFrame, int]:
    diagnosis = frame[config.diagnosis_column].astype("string").str.strip()
    diagnosis_number = pd.to_numeric(
        diagnosis.str.extract(r"^(\d+)", expand=False), errors="coerce"
    )
    present = ~diagnosis.str.contains(
        re.escape(config.absent_marker), case=False, na=False
    )
    requested = diagnosis_number.eq(config.diagnosis_number)
    selected = frame.loc[requested & present].copy()
    if selected.empty:
        raise ValueError(
            f"Nessuna diagnosi {config.diagnosis_number} presente nel CSV clinico"
        )

    duplicated = selected[config.patient_column].duplicated(keep=False)
    if duplicated.any():
        raise ValueError(
            "Piu' righe valide per la stessa diagnosi e lo stesso paziente: "
            + ", ".join(
                map(str, selected.loc[duplicated, config.patient_column].unique()[:10])
            )
        )
    return selected, int((requested & ~present).sum())


def _feature_name(config: ClinicalConfig, source_column: str) -> str:
    alias = config.feature_aliases.get(source_column, source_column)
    return f"clinical_{_slug(alias)}"


def _build_features(
    selected: pd.DataFrame,
    config: ClinicalConfig,
) -> tuple[pd.DataFrame, dict[str, dict[str, int]], list[dict[str, str]]]:
    features = pd.DataFrame(index=selected.index)
    issues: dict[str, dict[str, int]] = {}
    dictionary: list[dict[str, str]] = []

    for source_column in config.numeric_columns:
        output_column = _feature_name(config, source_column)
        source = selected[source_column]
        numeric = pd.to_numeric(source, errors="coerce")
        invalid = source.notna() & numeric.isna()
        if invalid.any():
            issues[source_column] = {
                str(value): int(count)
                for value, count in source.loc[invalid].value_counts().items()
            }
        features[output_column] = numeric.astype(float)
        dictionary.append(
            {
                "output_feature": output_column,
                "source_column": source_column,
                "encoding": "numeric",
                "source_value": "",
            }
        )

    for source_column, raw_mapping in config.ordinal_columns.items():
        output_column = _feature_name(config, source_column)
        mapping = {
            str(key).strip().casefold(): value for key, value in raw_mapping.items()
        }
        source = selected[source_column].astype("string").str.strip()
        canonical = source.str.casefold()
        mapped = canonical.map(mapping)
        unknown = source.notna() & ~canonical.isin(mapping)
        if unknown.any():
            issues[source_column] = {
                str(value): int(count)
                for value, count in source.loc[unknown].value_counts().items()
            }
        features[output_column] = pd.to_numeric(mapped, errors="coerce").astype(float)
        dictionary.append(
            {
                "output_feature": output_column,
                "source_column": source_column,
                "encoding": "ordinal",
                "source_value": " | ".join(
                    f"{key}={value}" for key, value in raw_mapping.items()
                ),
            }
        )

    for source_column, levels in config.categorical_columns.items():
        source = selected[source_column].astype("string").str.strip()
        canonical = source.str.casefold()
        valid_levels = {str(level).strip().casefold() for level in levels}
        unknown = source.notna() & ~canonical.isin(valid_levels)
        if unknown.any():
            issues[source_column] = {
                str(value): int(count)
                for value, count in source.loc[unknown].value_counts().items()
            }
        base_name = _feature_name(config, source_column)
        for level in levels:
            level_text = str(level).strip()
            output_column = f"{base_name}_{_slug(level_text)}"
            encoded = canonical.eq(level_text.casefold()).astype(float)
            encoded = encoded.mask(source.isna() | unknown, np.nan)
            features[output_column] = encoded
            dictionary.append(
                {
                    "output_feature": output_column,
                    "source_column": source_column,
                    "encoding": "one_hot",
                    "source_value": level_text,
                }
            )

    ratio = config.lymph_node_ratio
    if ratio.enabled:
        positive = pd.to_numeric(selected[ratio.positive_column], errors="coerce")
        examined = pd.to_numeric(selected[ratio.examined_column], errors="coerce")
        invalid = ((positive < 0) | (examined <= 0) | (positive > examined)).fillna(
            False
        )
        output_column = f"clinical_{_slug(ratio.output_name)}"
        features[output_column] = (positive / examined).mask(invalid).astype(float)
        if invalid.any():
            issues[output_column] = {"invalid_rows": int(invalid.sum())}
        dictionary.append(
            {
                "output_feature": output_column,
                "source_column": f"{ratio.positive_column}/{ratio.examined_column}",
                "encoding": "ratio",
                "source_value": "",
            }
        )

    if features.empty:
        raise ValueError("La configurazione non produce alcuna feature clinica")
    return features, issues, dictionary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_clinical_dataset(
    config: ClinicalConfig,
    *,
    config_path: str | Path | None = None,
) -> ClinicalPreparationResult:
    input_csv = config.input_csv.expanduser().resolve()
    output_csv = config.output_csv.expanduser().resolve()
    if not input_csv.is_file():
        raise FileNotFoundError(f"CSV clinico non trovato: {input_csv}")
    if output_csv.exists() and not config.overwrite:
        raise FileExistsError(
            f"Il CSV clinico preprocessato esiste gia': {output_csv}. "
            "Imposta overwrite: true per sostituirlo."
        )

    frame = pd.read_csv(
        input_csv,
        sep=config.separator,
        dtype="string",
        keep_default_na=False,
    )
    missing_columns = sorted(_required_columns(config) - set(frame.columns))
    if missing_columns:
        raise ValueError(
            "Colonne mancanti nel CSV clinico: " + ", ".join(missing_columns)
        )

    frame = _clean_missing_values(frame, config.missing_values)
    selected, absent_diagnoses = _select_diagnosis(frame, config)
    patient_keys = selected[config.patient_column].map(normalise_patient_id)
    if patient_keys.eq("").any():
        raise ValueError("La colonna paziente contiene identificativi vuoti")
    if patient_keys.duplicated().any():
        duplicates = patient_keys.loc[patient_keys.duplicated(keep=False)].unique()
        raise ValueError(
            "La normalizzazione produce ID paziente duplicati: "
            + ", ".join(map(str, duplicates[:10]))
        )

    features, issues, dictionary = _build_features(selected, config)
    output = features.copy()
    output.insert(0, config.output_id_column, selected[config.patient_column].str.strip())
    if config.diagnosis_date_column:
        diagnosis_date = pd.to_datetime(
            selected[config.diagnosis_date_column].astype("string").str.strip(),
            format="%d/%m/%Y",
            errors="coerce",
        )
        output.insert(
            1,
            "metadata_clinical_diagnosis_date",
            diagnosis_date.dt.strftime("%Y-%m-%d"),
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary_csv = output_csv.with_suffix(output_csv.suffix + ".tmp")
    output.to_csv(
        temporary_csv,
        sep=config.separator,
        index=False,
        encoding="utf-8-sig",
    )
    temporary_csv.replace(output_csv)

    report_json = output_csv.with_suffix(".report.json")
    report = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "input_csv": str(input_csv),
        "input_csv_sha256": _sha256(input_csv),
        "output_csv": str(output_csv),
        "diagnosis_number": config.diagnosis_number,
        "source_rows": int(len(frame)),
        "source_patients": int(frame[config.patient_column].nunique()),
        "output_patients": int(len(output)),
        "excluded_absent_diagnoses": absent_diagnoses,
        "clinical_features": int(len(features.columns)),
        "conversion_issues": issues,
    }
    report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(dictionary).to_csv(
        output_csv.with_suffix(".dictionary.csv"),
        sep=config.separator,
        index=False,
        encoding="utf-8-sig",
    )
    if config_path is not None:
        shutil.copy2(
            Path(config_path).resolve(), output_csv.with_suffix(".config.yaml")
        )

    return ClinicalPreparationResult(
        output_csv=output_csv,
        report_json=report_json,
        patients=len(output),
        excluded_absent_diagnoses=absent_diagnoses,
        clinical_feature_count=len(features.columns),
        conversion_issues=issues,
    )


def _print_result(result: ClinicalPreparationResult) -> None:
    console = Console()
    table = Table(title="DATASET CLINICO PREPROCESSATO", show_header=False)
    table.add_column("Elemento", style="cyan")
    table.add_column("Valore", style="green", overflow="fold")
    table.add_row("Pazienti", str(result.patients))
    table.add_row("Diagnosi assenti escluse", str(result.excluded_absent_diagnoses))
    table.add_row("Feature cliniche", str(result.clinical_feature_count))
    table.add_row("CSV", str(result.output_csv))
    table.add_row("Report", str(result.report_json))
    console.print(table)
    if result.conversion_issues:
        console.print(
            "[yellow]Valori non riconosciuti rilevati; controlla il report JSON.[/yellow]"
        )
    console.print(
        "[dim]Il dataset non viene incluso automaticamente. Imposta "
        "dataset_mode: clinical o combined in config/config_analysis.yaml.[/dim]"
    )


def run_clinical_preprocessing(
    config_path: str | Path = Path(__file__).resolve().parents[1] / "config/config_clinical.yaml",
) -> ClinicalPreparationResult:
    config = load_clinical_configuration(config_path)
    result = prepare_clinical_dataset(config, config_path=config_path)
    _print_result(result)
    return result
