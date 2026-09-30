from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LymphNodeRatioConfig(StrictConfigModel):
    enabled: bool = True
    positive_column: str = "Linfonodi positivi"
    examined_column: str = "Linfonodi esaminati"
    output_name: str = "lymph_node_ratio"


class ClinicalConfig(StrictConfigModel):
    input_csv: Path
    output_csv: Path
    overwrite: bool = True
    separator: str = ";"
    patient_column: str = "Paziente"
    output_id_column: str = "nome_cognome"
    diagnosis_column: str = "Diagnosi"
    diagnosis_number: int = Field(default=1, ge=1)
    absent_marker: str = "ASSENTE"
    diagnosis_date_column: str | None = "Data diagnosi"
    missing_values: list[str] = Field(
        default_factory=lambda: ["", "missing", "na", "n/a", "null", "none"]
    )
    numeric_columns: list[str] = Field(default_factory=list)
    ordinal_columns: dict[str, dict[str, float | None]] = Field(default_factory=dict)
    categorical_columns: dict[str, list[str]] = Field(default_factory=dict)
    feature_aliases: dict[str, str] = Field(default_factory=dict)
    lymph_node_ratio: LymphNodeRatioConfig = Field(
        default_factory=LymphNodeRatioConfig
    )

    @field_validator("separator")
    @classmethod
    def validate_separator(cls, value: str) -> str:
        if len(value) != 1:
            raise ValueError("separator deve contenere esattamente un carattere")
        return value

    @field_validator(
        "patient_column",
        "output_id_column",
        "diagnosis_column",
        "absent_marker",
    )
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("I nomi delle colonne e delle cartelle non possono essere vuoti")
        return value


class GlobalClinicalConfig(StrictConfigModel):
    clinical: ClinicalConfig


def load_clinical_configuration(
    config_path: str | Path = Path(__file__).resolve().parents[1] / "config/config_clinical.yaml",
) -> ClinicalConfig:
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"File di configurazione clinica non trovato: {path}")

    with path.open("r", encoding="utf-8") as yaml_file:
        raw_data = yaml.safe_load(yaml_file) or {}

    config = GlobalClinicalConfig.model_validate(raw_data).clinical
    input_csv = config.input_csv.expanduser()
    output_csv = config.output_csv.expanduser()
    if not input_csv.is_absolute():
        input_csv = path.parent / input_csv
    if not output_csv.is_absolute():
        output_csv = path.parent / output_csv
    return config.model_copy(
        update={
            "input_csv": input_csv.resolve(),
            "output_csv": output_csv.resolve(),
        }
    )
