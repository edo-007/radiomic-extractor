from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator


SUPPORTED_TUNING_MODELS = {
    "logistic_l1",
    "logistic_l2",
    "random_forest",
    "svm_linear",
    "svm_rbf",
    "svm_poly",
    "svm_sigmoid",
}


class StrictTuningModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OuterCVConfig(StrictTuningModel):
    n_splits: int = Field(default=4, ge=2)
    n_repeats: int = Field(default=5, ge=1)


class InnerCVConfig(StrictTuningModel):
    n_splits: int = Field(default=3, ge=2)


class TuningModelConfig(StrictTuningModel):
    enabled: bool = True
    parameters: dict[str, list[Any]] = Field(default_factory=dict)

    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls, value: dict[str, list[Any]]) -> dict[str, list[Any]]:
        for parameter, choices in value.items():
            if not parameter.strip():
                raise ValueError("Il nome di un iperparametro non può essere vuoto")
            if not isinstance(choices, list) or not choices:
                raise ValueError(
                    f"L'iperparametro '{parameter}' deve avere una lista non vuota"
                )
        return value


TuningScore = Literal["roc_auc", "balanced_accuracy", "accuracy", "f1"]


class TuningConfig(StrictTuningModel):
    _source_bytes: bytes = PrivateAttr(default=b"")
    _source_path: Path | None = PrivateAttr(default=None)
    output_dir: Path = Path("../../radiomic-output/tuning")
    strategy: Literal["randomized", "grid"] = "randomized"
    scoring: TuningScore | list[TuningScore] = "roc_auc"
    refit: TuningScore | None = None
    n_iter: int = Field(default=20, ge=1)
    n_jobs: int | None = 1
    random_state: int = 42
    outer_cv: OuterCVConfig = Field(default_factory=OuterCVConfig)
    inner_cv: InnerCVConfig = Field(default_factory=InnerCVConfig)
    preprocessing_parameters: dict[str, list[Any]] = Field(default_factory=dict)
    models: dict[str, TuningModelConfig]

    @field_validator("n_jobs")
    @classmethod
    def validate_n_jobs(cls, value: int | None) -> int | None:
        if value is not None and value != -1 and value < 1:
            raise ValueError("n_jobs deve essere -1, null oppure un intero >= 1")
        return value

    @field_validator("preprocessing_parameters")
    @classmethod
    def validate_preprocessing_parameters(
        cls,
        value: dict[str, list[Any]],
    ) -> dict[str, list[Any]]:
        for parameter, choices in value.items():
            if not parameter.strip() or not isinstance(choices, list) or not choices:
                raise ValueError(
                    "Ogni parametro di preprocessing deve avere un nome e una "
                    "lista non vuota"
                )
        return value

    @field_validator("models")
    @classmethod
    def validate_model_names(
        cls,
        value: dict[str, TuningModelConfig],
    ) -> dict[str, TuningModelConfig]:
        unknown = sorted(set(value) - SUPPORTED_TUNING_MODELS)
        if unknown:
            raise ValueError("Modelli di tuning non supportati: " + ", ".join(unknown))
        return value

    @model_validator(mode="after")
    def validate_enabled_models(self) -> "TuningConfig":
        if not self.enabled_models():
            raise ValueError("Abilita almeno un modello in config/config_tuning.yaml")
        metrics = self.score_names()
        if not metrics or len(metrics) != len(set(metrics)):
            raise ValueError("scoring deve contenere almeno una metrica, senza duplicati")
        if self.refit is None:
            self.refit = metrics[0]
        if self.refit not in metrics:
            raise ValueError("refit deve essere una delle metriche elencate in scoring")
        return self

    def score_names(self) -> list[str]:
        return [self.scoring] if isinstance(self.scoring, str) else list(self.scoring)

    def enabled_models(self) -> dict[str, TuningModelConfig]:
        return {
            name: model_config
            for name, model_config in self.models.items()
            if model_config.enabled
        }


class GlobalTuningConfig(StrictTuningModel):
    tuning: TuningConfig


def load_tuning_configuration(
    config_path: str | Path = Path(__file__).resolve().parents[1] / "config/config_tuning.yaml",
) -> TuningConfig:
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"File di configurazione tuning non trovato: {path}")

    source_bytes = path.read_bytes()
    raw_data = yaml.safe_load(source_bytes.decode("utf-8-sig")) or {}

    config = GlobalTuningConfig.model_validate(raw_data).tuning
    config._source_bytes = source_bytes
    config._source_path = path
    output_dir = config.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = path.parent / output_dir
    return config.model_copy(update={"output_dir": output_dir.resolve()})
