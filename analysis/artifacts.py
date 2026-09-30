from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import joblib
import yaml

from .config import AnalysisConfig
from .tuning import TuningResult
from .tuning_config import TuningConfig


def prepare_tuning_artifacts(
    *,
    analysis_config: AnalysisConfig,
    tuning_config: TuningConfig,
    input_csv: Path,
    clinical_input_csv: Path | None = None,
) -> Path:
    """Congela la provenienza prima dei fit, usando gli stessi byte dei loader."""
    run_dir = _create_run_dir(tuning_config.output_dir)
    hashes: dict[str, str] = {}

    def save_bytes(relative_path: str | Path, content: bytes) -> None:
        destination = run_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        hashes[Path(relative_path).as_posix()] = hashlib.sha256(content).hexdigest()

    for name, config in (("analysis", analysis_config), ("tuning", tuning_config)):
        resolved = yaml.safe_dump(
            {name: config.model_dump(mode="json")},
            allow_unicode=True,
            sort_keys=False,
        ).encode("utf-8")
        save_bytes(f"config_{name}.yaml", config._source_bytes or resolved)
        save_bytes(f"config_{name}_resolved.yaml", resolved)

    input_csv = input_csv.resolve()
    extraction_files = [
        path for path in input_csv.parent.iterdir()
        if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}
    ]
    sources_dir = input_csv.parent / "config_sources"
    if sources_dir.is_dir():
        extraction_files.extend(path for path in sources_dir.rglob("*") if path.is_file())
    for path in sorted(extraction_files):
        content = path.read_bytes()
        save_bytes(Path("extraction_config") / path.relative_to(input_csv.parent), content)
        if path.name == "config_resolved.yaml" and path.parent == input_csv.parent:
            save_bytes("config_extraction.yaml", content)

    metadata = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "input_csv": str(input_csv.resolve()),
        "experiment_dir": str(input_csv.resolve().parent),
        "input_csv_sha256": _sha256(input_csv),
        "analysis_config_source": str(analysis_config._source_path) if analysis_config._source_path else None,
        "tuning_config_source": str(tuning_config._source_path) if tuning_config._source_path else None,
        "extraction_config_available": (run_dir / "config_extraction.yaml").is_file(),
        "preprocessing_scaling_by_model": {
            name: "none" if name == "random_forest" else analysis_config.preprocessing.scaling
            for name in tuning_config.enabled_models()
        },
    }
    if clinical_input_csv is not None:
        clinical_input_csv = clinical_input_csv.resolve()
        metadata["clinical_input_csv"] = str(clinical_input_csv)
        metadata["clinical_input_csv_sha256"] = _sha256(clinical_input_csv)
        clinical_config = clinical_input_csv.with_suffix(".config.yaml")
        if clinical_config.is_file():
            save_bytes("config_clinical.yaml", clinical_config.read_bytes())
    metadata["configuration_sha256"] = hashes
    (run_dir / "source.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return run_dir


def save_tuning_artifacts(result: TuningResult, *, run_dir: Path) -> Path:
    """Salva solo i risultati; le configurazioni iniziali non vengono rilette."""
    (run_dir / "best_parameters.json").write_text(
        json.dumps(result.final_parameters, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    result.summary.to_csv(run_dir / "summary.csv", sep=";", index=False)
    result.outer_folds.to_csv(run_dir / "outer_fold_metrics.csv", sep=";", index=False)
    result.search_results.to_csv(run_dir / "cv_results.csv", sep=";", index=False)
    result.selected_features.to_csv(
        run_dir / "selected_features.csv", sep=";", index=False
    )

    models_dir = run_dir / "models"
    models_dir.mkdir()
    for model_name, model in result.final_models.items():
        joblib.dump(model, models_dir / f"{model_name}.joblib")

    return run_dir


def _create_run_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / f"tuning-{datetime.now():%Y-%m-%d_%H-%M-%S}"
    for counter in range(1, 1000):
        candidate = base if counter == 1 else base.with_name(f"{base.name}_{counter}")
        try:
            candidate.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        return candidate
    raise FileExistsError(f"Non trovo un nome libero per il tuning in: {output_dir}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
