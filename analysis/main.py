from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from .artifacts import prepare_tuning_artifacts, save_tuning_artifacts
from .benchmark import run_benchmark
from .config import load_analysis_configuration
from .data import load_analysis_dataset
from .importance import run_cross_validated_importance
from .models import MODEL_LABELS, build_estimators
from .reporting import TerminalReporter
from .selector import select_analysis_input
from .tuning import run_nested_tuning
from .tuning_config import load_tuning_configuration
from .univariate import run_univariate_analysis


ANALYSIS_COMMANDS = ("univariate", "benchmark", "importance", "all", "tune")


def _run_importance_with_progress(dataset, config, estimators, reporter):
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=reporter.console,
    )
    with progress:
        task_id = progress.add_task("Permutation importance", total=None)

        def update_progress(model_name: str, completed: int, total: int) -> None:
            progress.update(
                task_id,
                description=(
                    "Permutation importance - "
                    + MODEL_LABELS.get(model_name, model_name)
                ),
                completed=completed,
                total=total,
            )

        return run_cross_validated_importance(
            dataset,
            config,
            estimators,
            progress_callback=update_progress,
        )


def _run_tuning_with_progress(dataset, config, tuning_config, reporter):
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TextColumn("trascorso:"),
        TimeElapsedColumn(),
        TextColumn("rimanente stimato:"),
        TimeRemainingColumn(),
        # La progress bar non entra nel report statico delle tabelle.
        console=Console(width=reporter.console.width),
    )
    with progress:
        task_id = progress.add_task("Preparazione nested CV", total=None)

        def update_progress(
            model_name: str,
            phase: str,
            completed: int,
            total: int,
        ) -> None:
            progress.update(
                task_id,
                description=f"Tuning {MODEL_LABELS.get(model_name, model_name)} - {phase}",
                completed=completed,
                total=total,
            )

        return run_nested_tuning(
            dataset,
            config,
            tuning_config,
            progress_callback=update_progress,
        )


def run_analysis(
    command: str,
    config_path: str | Path,
    tuning_config_path: str | Path = Path(__file__).resolve().parents[1] / "config/config_tuning.yaml",
) -> int:
    if command not in ANALYSIS_COMMANDS:
        raise ValueError(f"Comando di analisi non supportato: {command}")

    config = load_analysis_configuration(config_path)
    tuning_config = (
        load_tuning_configuration(tuning_config_path)
        if command == "tune"
        else None
    )
    input_csv = select_analysis_input(config.experiments_dir)
    dataset = load_analysis_dataset(config, input_csv)
    reporter = TerminalReporter(config)
    reporter.print_dataset(dataset)

    if command == "tune":
        output_dir = prepare_tuning_artifacts(
            analysis_config=config,
            tuning_config=tuning_config,
            input_csv=input_csv,
            clinical_input_csv=dataset.clinical_input_csv,
        )
        try:
            if not (output_dir / "config_extraction.yaml").is_file():
                reporter.console.print(
                    "[yellow]Attenzione: nell'estrazione manca config_resolved.yaml; "
                    "sono stati conservati soltanto i file di configurazione disponibili.[/yellow]"
                )
            reporter.print_tuning_plan(tuning_config)
            result = _run_tuning_with_progress(dataset, config, tuning_config, reporter)
            save_tuning_artifacts(result, run_dir=output_dir)
            reporter.print_tuning(result, tuning_config, output_dir)
        except Exception as error:
            reporter.console.print(f"Tuning non completato: {error}", markup=False)
            raise
        finally:
            reporter.save_report(output_dir)
        return 0

    if command in {"univariate", "all"}:
        if command == "univariate" or config.univariate.enabled:
            reporter.print_univariate(
                run_univariate_analysis(dataset, config.univariate)
            )

    if command in {"benchmark", "importance", "all"}:
        estimators = build_estimators(
            config.models,
            random_state=config.cross_validation.random_state,
        )
        reporter.print_models()

        if command in {"benchmark", "all"}:
            reporter.print_benchmark(run_benchmark(dataset, config, estimators))

        if command in {"importance", "all"} and (
            command == "importance" or config.importance.enabled
        ):
            if config.cross_validation.method == "leave_one_out":
                if command == "importance":
                    raise ValueError(
                        "Permutation importance non disponibile con LOO: ogni "
                        "validation fold contiene un solo paziente. Usa "
                        "stratified_kfold o repeated_stratified_kfold."
                    )
                reporter.print_loo_importance_notice()
            else:
                reporter.print_importance(
                    _run_importance_with_progress(
                        dataset,
                        config,
                        estimators,
                        reporter,
                    )
                )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analisi statistica e machine learning di feature radiomiche."
    )
    parser.add_argument("command", choices=ANALYSIS_COMMANDS)
    parser.add_argument(
        "--config",
        type=Path,
        default=(Path(__file__).resolve().parents[1] / "config/config_analysis.yaml"),
        help="File YAML dell'analisi (default: config/config_analysis.yaml).",
    )
    parser.add_argument(
        "--tuning-config",
        type=Path,
        default=(Path(__file__).resolve().parents[1] / "config/config_tuning.yaml"),
        help="File YAML del tuning (usato dal comando tune).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run_analysis(args.command, args.config, args.tuning_config)
    except (FileNotFoundError, OSError, ValueError, RuntimeError, ValidationError) as error:
        print(f"Errore: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
