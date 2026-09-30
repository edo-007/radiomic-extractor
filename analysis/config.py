from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator


class StrictConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(StrictConfigModel):
    enabled: bool = True
    params: dict[str, Any] = Field(default_factory=dict)


class ModelsConfig(StrictConfigModel):
    logistic_l1: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            params={
                "C": 1.0,
                "solver": "liblinear",
                "class_weight": "balanced",
                "max_iter": 5000,
            }
        )
    )
    logistic_l2: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            params={
                "C": 1.0,
                "solver": "liblinear",
                "class_weight": "balanced",
                "max_iter": 5000,
            }
        )
    )
    random_forest: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            params={
                "n_estimators": 500,
                "class_weight": "balanced_subsample",
                "n_jobs": -1,
            }
        )
    )
    svm_linear: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            params={"C": 1.0, "class_weight": "balanced"}
        )
    )
    svm_rbf: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            params={"C": 1.0, "gamma": "scale", "class_weight": "balanced"}
        )
    )
    svm_poly: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            enabled=False,
            params={
                "C": 1.0,
                "degree": 3,
                "gamma": "scale",
                "coef0": 0.0,
                "class_weight": "balanced",
            },
        )
    )
    svm_sigmoid: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            enabled=False,
            params={
                "C": 1.0,
                "gamma": "scale",
                "coef0": 0.0,
                "class_weight": "balanced",
            },
        )
    )

    def enabled_models(self) -> dict[str, ModelConfig]:
        return {
            name: model
            for name, model in self
            if isinstance(model, ModelConfig) and model.enabled
        }


class FeatureSelectionConfig(StrictConfigModel):
    enabled: bool = True
    method: Literal["f_classif", "mutual_info"] = "f_classif"
    k: int = Field(default=20, ge=1)


class PreprocessingConfig(StrictConfigModel):
    impute_strategy: Literal["median", "mean", "most_frequent"] = "median"
    remove_constant_features: bool = True
    correlation_threshold: float | None = Field(default=0.90, gt=0.0, lt=1.0)
    scaling: Literal["standard", "robust", "none"] = "standard"
    feature_selection: FeatureSelectionConfig = Field(
        default_factory=FeatureSelectionConfig
    )


class UnivariateConfig(StrictConfigModel):
    enabled: bool = True
    test: Literal["mannwhitney", "welch_ttest"] = "mannwhitney"
    correction: Literal["fdr_bh", "none"] = "fdr_bh"


class CrossValidationConfig(StrictConfigModel):
    method: Literal[
        "stratified_kfold",
        "repeated_stratified_kfold",
        "leave_one_out",
    ] = "repeated_stratified_kfold"
    n_splits: int = Field(default=5, ge=2)
    n_repeats: int = Field(default=5, ge=1)
    random_state: int = 42

    @field_validator("method", mode="before")
    @classmethod
    def normalise_method(cls, value: str) -> str:
        if not isinstance(value, str):
            return value
        value = value.strip().lower().replace("-", "_")
        aliases = {
            "loo": "leave_one_out",
            "leave_one_group_out": "leave_one_out",
        }
        return aliases.get(value, value)


class ImportanceConfig(StrictConfigModel):
    enabled: bool = True
    scoring: Literal["roc_auc", "balanced_accuracy", "accuracy", "f1"] = "roc_auc"
    permutation_repeats: int = Field(default=5, ge=1)
    n_jobs: int | None = None


class ReportingConfig(StrictConfigModel):
    top_n_features: int = Field(default=20, ge=1)
    terminal_width: int = Field(default=180, ge=80)


class ClinicalDataConfig(StrictConfigModel):
    input_csv: Path | None = None
    separator: str = ";"
    id_column: str = "nome_cognome"
    restrict_radiomic_to_matched_patients: bool = False

    @field_validator("separator")
    @classmethod
    def validate_separator(cls, value: str) -> str:
        if len(value) != 1:
            raise ValueError(
                "clinical_data.separator deve contenere esattamente un carattere"
            )
        return value


class AnalysisConfig(StrictConfigModel):
    _source_bytes: bytes = PrivateAttr(default=b"")
    _source_path: Path | None = PrivateAttr(default=None)
    experiments_dir: Path = Path("../radiomic-output")
    dataset_mode: Literal["radiomic", "clinical", "combined"] = "radiomic"
    clinical_data: ClinicalDataConfig = Field(default_factory=ClinicalDataConfig)
    separator: str = ";"
    id_column: str = "nome_cognome"
    target_column: str = "stato_microsatellitare"
    positive_class: str | int | float | bool = "INSTABILE"
    exclude_columns: list[str] = Field(default_factory=list)
    metadata_prefix: str = "metadata_"
    include_metadata: bool = False
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    univariate: UnivariateConfig = Field(default_factory=UnivariateConfig)
    cross_validation: CrossValidationConfig = Field(
        default_factory=CrossValidationConfig
    )
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    importance: ImportanceConfig = Field(default_factory=ImportanceConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)

    @field_validator("separator")
    @classmethod
    def validate_separator(cls, value: str) -> str:
        if len(value) != 1:
            raise ValueError("separator deve contenere esattamente un carattere")
        return value

    @field_validator("id_column", "target_column")
    @classmethod
    def validate_column_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Il nome di una colonna non puo' essere vuoto")
        return value


class GlobalAnalysisConfig(StrictConfigModel):
    analysis: AnalysisConfig


def load_analysis_configuration(
    config_path: str | Path = Path(__file__).resolve().parents[1] / "config/config_analysis.yaml",
) -> AnalysisConfig:
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"File di configurazione non trovato: {path}")

    source_bytes = path.read_bytes()
    raw_data = yaml.safe_load(source_bytes.decode("utf-8-sig")) or {}

    config = GlobalAnalysisConfig.model_validate(raw_data).analysis
    config._source_bytes = source_bytes
    config._source_path = path
    experiments_dir = config.experiments_dir.expanduser()
    if not experiments_dir.is_absolute():
        experiments_dir = path.parent / experiments_dir
    clinical_data = config.clinical_data
    clinical_csv = clinical_data.input_csv
    if clinical_csv is not None:
        clinical_csv = clinical_csv.expanduser()
        if not clinical_csv.is_absolute():
            clinical_csv = path.parent / clinical_csv
        clinical_data = clinical_data.model_copy(
            update={"input_csv": clinical_csv.resolve()}
        )
    clinical_file_required = (
        config.dataset_mode != "radiomic"
        or config.clinical_data.restrict_radiomic_to_matched_patients
    )
    if clinical_file_required and clinical_csv is None:
        raise ValueError(
            "La modalita' clinica richiesta richiede clinical_data.input_csv"
        )
    return config.model_copy(
        update={
            "experiments_dir": experiments_dir.resolve(),
            "clinical_data": clinical_data,
        }
    )
