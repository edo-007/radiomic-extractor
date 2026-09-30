from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from prompt_toolkit.shortcuts import choice
from prompt_toolkit.styles import Style


@dataclass(frozen=True)
class ExperimentChoice:
    path: Path
    feature_csvs: tuple[Path, ...]
    modified_ns: int


SELECTOR_STYLE = Style.from_dict(
    {
        "question": "bold fg:#5fd7ff",
        "pointer": "bold fg:#00d787",
        "selected-option": "bold fg:#00d787",
        "option": "fg:#d0d0d0",
        "bottom-toolbar": "bg:#303030 fg:#ffffff",
        "frame.border": "fg:#5fd7ff",
    }
)


def list_feature_csvs(experiment_dir: Path) -> tuple[Path, ...]:
    """Elenca i CSV di dati, escludendo la preview delle colonne."""
    csv_files = [
        path
        for path in experiment_dir.iterdir()
        if path.is_file()
        and path.suffix.casefold() == ".csv"
        and path.name.casefold() != "feature_preview.csv"
    ]
    return tuple(sorted(csv_files, key=lambda path: path.name.casefold()))


def discover_experiments(experiments_dir: str | Path) -> list[ExperimentChoice]:
    """Trova gli esperimenti con feature, dal più recente al più vecchio."""
    root = Path(experiments_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Cartella degli esperimenti non trovata: {root}")

    experiments: list[ExperimentChoice] = []
    for directory in root.iterdir():
        if not directory.is_dir() or directory.name == "tuning" or directory.name.startswith("tuning-"):
            continue
        feature_csvs = list_feature_csvs(directory)
        if not feature_csvs:
            continue
        experiments.append(
            ExperimentChoice(
                path=directory,
                feature_csvs=feature_csvs,
                modified_ns=directory.stat().st_mtime_ns,
            )
        )

    experiments.sort(
        key=lambda experiment: (
            experiment.modified_ns,
            experiment.path.name.casefold(),
        ),
        reverse=True,
    )
    if not experiments:
        raise FileNotFoundError(
            f"Nessun esperimento contenente CSV di feature trovato in: {root}"
        )
    return experiments


def select_analysis_input(experiments_dir: str | Path) -> Path:
    """Fa scegliere interattivamente esperimento e CSV tramite tastiera."""
    experiments = discover_experiments(experiments_dir)
    toolbar = "  ↑/↓ cambia opzione  •  Invio conferma  •  Ctrl+C annulla  "

    experiment_options = [
        (
            experiment,
            _experiment_label(experiment),
        )
        for experiment in experiments
    ]
    try:
        selected_experiment = choice(
            message="Scegli l'esperimento (il più recente è il primo):",
            options=experiment_options,
            default=experiments[0],
            symbol="❯",
            bottom_toolbar=toolbar,
            show_frame=True,
            mouse_support=True,
            style=SELECTOR_STYLE,
        )

        selected_csv = choice(
            message=f"Scegli il CSV delle feature in {selected_experiment.path.name}:",
            options=[
                (csv_path, _csv_label(csv_path))
                for csv_path in selected_experiment.feature_csvs
            ],
            default=selected_experiment.feature_csvs[0],
            symbol="❯",
            bottom_toolbar=toolbar,
            show_frame=True,
            mouse_support=True,
            style=SELECTOR_STYLE,
        )
    except (EOFError, KeyboardInterrupt) as error:
        raise RuntimeError("Selezione del CSV annullata") from error

    print(f"CSV selezionato: {selected_csv}")
    return selected_csv


def _experiment_label(experiment: ExperimentChoice) -> str:
    modified = datetime.fromtimestamp(experiment.modified_ns / 1_000_000_000)
    csv_label = "CSV" if len(experiment.feature_csvs) == 1 else "CSV"
    return (
        f"{experiment.path.name}  ·  {len(experiment.feature_csvs)} {csv_label}  ·  "
        f"aggiornato {modified:%d/%m/%Y %H:%M}"
    )


def _csv_label(csv_path: Path) -> str:
    size_mb = csv_path.stat().st_size / (1024 * 1024)
    return f"{csv_path.name}  ·  {size_mb:.1f} MB"
