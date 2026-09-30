from __future__ import annotations

import math
from math import prod
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from .benchmark import BenchmarkResult
from .config import AnalysisConfig
from .data import AnalysisDataset
from .models import MODEL_LABELS
from .tuning import TuningResult
from .tuning_config import TuningConfig


def _number(value: object, digits: int = 3) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "-"
    return f"{numeric:.{digits}f}"


def _mean_and_std(mean: object, standard_deviation: object) -> str:
    formatted_mean = _number(mean)
    formatted_std = _number(standard_deviation)
    if formatted_std == "-":
        return formatted_mean
    return f"{formatted_mean} +/- {formatted_std}"


class TerminalReporter:
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.console = Console(width=config.reporting.terminal_width, record=True)

    def save_report(self, output_dir: Path) -> None:
        """Esporta le stesse tabelle: testo UTF-8 e HTML con colori e layout."""
        self.console.save_text(output_dir / "report.txt", clear=False)
        self.console.save_html(output_dir / "report.html", clear=False)

    def print_dataset(self, dataset: AnalysisDataset) -> None:
        class_counts = dataset.y.value_counts()
        table = Table(title="DATI CARICATI", show_header=False)
        table.add_column("Elemento", style="cyan")
        table.add_column("Valore", style="green", overflow="fold")
        table.add_row("CSV", str(dataset.input_csv))
        table.add_row("Modalita'", dataset.data_mode)
        if dataset.clinical_input_csv is not None:
            table.add_row("CSV clinico", str(dataset.clinical_input_csv))
            table.add_row(
                "Pazienti radiomici senza dati clinici esclusi",
                str(dataset.excluded_unmatched_patients),
            )
            table.add_row(
                "Pazienti clinici senza riga radiomica non usati",
                str(dataset.unused_clinical_patients),
            )
        table.add_row("Righe", str(len(dataset.X)))
        table.add_row("Pazienti", str(dataset.patient_count))
        table.add_row("Feature numeriche", str(dataset.X.shape[1]))
        table.add_row(
            dataset.negative_label,
            str(int(class_counts.get(0, 0))),
        )
        table.add_row(
            f"{dataset.positive_label} (classe positiva)",
            str(int(class_counts.get(1, 0))),
        )
        if dataset.duplicated_patient_rows:
            table.add_row(
                "Righe con ID paziente ripetuto",
                str(dataset.duplicated_patient_rows),
            )
        self.console.print(table)

        if dataset.ignored_non_numeric_columns:
            self.console.print(
                "[yellow]Colonne non numeriche ignorate:[/yellow] "
                + ", ".join(dataset.ignored_non_numeric_columns)
            )
        if dataset.coerced_missing_values:
            details = ", ".join(
                f"{name}={count}"
                for name, count in dataset.coerced_missing_values.items()
            )
            self.console.print(
                "[yellow]Valori non numerici trattati come mancanti:[/yellow] "
                + details
            )

    def print_models(self) -> None:
        table = Table(title="MODELLI ABILITATI")
        table.add_column("Modello", style="cyan")
        table.add_column("Parametri", style="green", overflow="fold")
        for model_name, model_config in self.config.models.enabled_models().items():
            parameters = ", ".join(
                f"{key}={value}" for key, value in model_config.params.items()
            )
            table.add_row(MODEL_LABELS.get(model_name, model_name), parameters or "-")
        self.console.print(table)

    def print_univariate(self, results: pd.DataFrame) -> None:
        table = Table(
            title=(
                "ANALISI UNIVARIATA - prime "
                f"{min(len(results), self.config.reporting.top_n_features)} feature"
            )
        )
        table.add_column("Feature", style="cyan", overflow="fold")
        table.add_column("AUC discr.", justify="right")
        table.add_column("Effetto", justify="right")
        table.add_column("p-value", justify="right")
        table.add_column("p FDR", justify="right")
        table.add_column("Mediana -", justify="right")
        table.add_column("Mediana +", justify="right")
        for row in results.head(self.config.reporting.top_n_features).itertuples():
            table.add_row(
                str(row.feature),
                _number(row.discriminative_auc),
                _number(row.effect_size),
                _number(row.p_value, 4),
                _number(row.adjusted_p_value, 4),
                _number(row.median_negative),
                _number(row.median_positive),
            )
        self.console.print(table)
        self.console.print(
            "[dim]L'analisi univariata e' descrittiva; non viene usata fuori "
            "dai fold per selezionare le feature del benchmark.[/dim]"
        )

    def print_benchmark(self, result: BenchmarkResult) -> None:
        table = Table(title="BENCHMARK IN CROSS-VALIDATION")
        table.add_column("Modello", style="cyan")
        table.add_column("Fold", justify="right")
        table.add_column("ROC-AUC", justify="right")
        table.add_column("Bal. accuracy", justify="right")
        table.add_column("Accuracy", justify="right")
        table.add_column("Sensibilita'", justify="right")
        table.add_column("Specificita'", justify="right")
        table.add_column("F1", justify="right")
        for row in result.summary.itertuples():
            table.add_row(
                MODEL_LABELS.get(row.model, row.model),
                str(row.n_folds),
                _mean_and_std(row.roc_auc_mean, row.roc_auc_std),
                _mean_and_std(
                    row.balanced_accuracy_mean,
                    row.balanced_accuracy_std,
                ),
                _mean_and_std(row.accuracy_mean, row.accuracy_std),
                _mean_and_std(row.sensitivity_mean, row.sensitivity_std),
                _mean_and_std(row.specificity_mean, row.specificity_std),
                _mean_and_std(row.f1_mean, row.f1_std),
            )
        self.console.print(table)
        if self.config.cross_validation.method == "leave_one_out":
            self.console.print(
                "[dim]LOO: le metriche sono calcolate aggregando tutte le "
                "predizioni out-of-fold, una per paziente.[/dim]"
            )

    def print_loo_importance_notice(self) -> None:
        self.console.print(
            "[yellow]Permutation importance non eseguita: con un solo paziente "
            "nel validation fold la permutazione non e' definita. Per questa "
            "analisi usa stratified_kfold o repeated_stratified_kfold.[/yellow]"
        )

    def print_importance(self, results: pd.DataFrame) -> None:
        top_n = self.config.reporting.top_n_features
        for model_name, model_results in results.groupby("model", sort=False):
            table = Table(
                title=(
                    "PERMUTATION IMPORTANCE CV - "
                    + MODEL_LABELS.get(model_name, model_name)
                )
            )
            table.add_column("Feature", style="cyan", overflow="fold")
            table.add_column("Importanza", justify="right")
            table.add_column("Dev. std.", justify="right")
            table.add_column("Fold selez.", justify="right")
            table.add_column("% positiva", justify="right")
            for row in model_results.head(top_n).itertuples():
                table.add_row(
                    str(row.feature),
                    _number(row.importance_mean, 4),
                    _number(row.importance_std, 4),
                    f"{100.0 * row.selection_fraction:.1f}%",
                    f"{100.0 * row.positive_fraction:.1f}%",
                )
            self.console.print(table)

        self.console.print(
            "[dim]Le importance sono misurate sui validation fold. Valori vicini "
            "a zero indicano un contributo predittivo non stabile; 'Fold selez.' "
            "indica quanto spesso la feature ha superato la selezione interna.[/dim]"
        )

    def print_tuning(
        self,
        result: TuningResult,
        tuning_config: TuningConfig,
        output_dir,
    ) -> None:
        table = Table(title="VALUTAZIONE FINALE - OUTER NESTED CV")
        table.add_column("Modello", style="cyan")
        table.add_column("Outer fold", justify="right")
        table.add_column("ROC-AUC", justify="right")
        table.add_column("IC 95% AUC", justify="right")
        table.add_column("Bal. accuracy", justify="right")
        table.add_column("Accuracy", justify="right")
        table.add_column("Sensibilita'", justify="right")
        table.add_column("Specificita'", justify="right")
        table.add_column("F1", justify="right")
        for row in result.summary.itertuples():
            table.add_row(
                MODEL_LABELS.get(row.model, row.model),
                str(row.n_outer_folds),
                _mean_and_std(row.roc_auc_mean, row.roc_auc_std),
                f"+/- {_number(row.roc_auc_ci95)}",
                _mean_and_std(
                    row.balanced_accuracy_mean,
                    row.balanced_accuracy_std,
                ),
                _mean_and_std(row.accuracy_mean, row.accuracy_std),
                _mean_and_std(row.sensitivity_mean, row.sensitivity_std),
                _mean_and_std(row.specificity_mean, row.specificity_std),
                _mean_and_std(row.f1_mean, row.f1_std),
            )
        self.console.print(table)

        parameters = Table(title="REFIT FINALE SULL'INTERO DATASET")
        parameters.add_column("Modello", style="cyan")
        for metric in tuning_config.score_names():
            suffix = " (scelta)" if metric == tuning_config.refit else ""
            parameters.add_column(f"Inner {metric}{suffix}", justify="right")
        parameters.add_column("Parametri", style="green", overflow="fold")
        for model_name, details in result.final_parameters.items():
            formatted = ", ".join(
                f"{name}={value}"
                for name, value in details["parameters"].items()
            )
            parameters.add_row(
                MODEL_LABELS.get(model_name, model_name),
                *[_number(details["inner_cv_metrics"][metric]) for metric in tuning_config.score_names()],
                formatted,
            )
        self.console.print(parameters)
        self.console.print(
            "[dim]Le sei metriche della outer CV sono la valutazione da riportare. "
            "I punteggi inner si riferiscono tutti alla stessa configurazione scelta "
            f"tramite {tuning_config.refit}; non sono una valutazione indipendente. "
            f"La tabella outer e' ordinata per {tuning_config.refit}.[/dim]"
        )
        self.console.print(f"[green]Risultati e modelli salvati in: {output_dir}[/green]")
        self.console.print("Report del terminale: report.txt e report.html")

    def print_tuning_plan(self, tuning_config: TuningConfig) -> None:
        if "random_forest" in tuning_config.enabled_models():
            self.console.print("Random Forest: scaling disabilitato; gli altri step di preprocessing restano attivi.")
        self.console.print(
            f"Metriche inner: {', '.join(tuning_config.score_names())} | "
            f"Metrica per scegliere gli iperparametri (refit): {tuning_config.refit}"
        )
        outer_folds = (
            tuning_config.outer_cv.n_splits * tuning_config.outer_cv.n_repeats
        )
        table = Table(title="PIANO DEL TUNING")
        table.add_column("Modello", style="cyan")
        table.add_column("Configurazioni/ricerca", justify="right")
        table.add_column("Fit stimati", justify="right")
        total_fits = 0
        for model_name, model_config in tuning_config.enabled_models().items():
            space = {
                **tuning_config.preprocessing_parameters,
                **model_config.parameters,
            }
            combinations = prod(len(values) for values in space.values())
            candidates = (
                combinations
                if tuning_config.strategy == "grid"
                else min(tuning_config.n_iter, combinations)
            )
            fits = (
                candidates
                * tuning_config.inner_cv.n_splits
                * (outer_folds + 1)
            )
            total_fits += fits
            table.add_row(
                MODEL_LABELS.get(model_name, model_name),
                str(candidates),
                str(fits),
            )
        table.caption = (
            f"Outer fold: {outer_folds} · Inner fold: "
            f"{tuning_config.inner_cv.n_splits} · Fit totali stimati: {total_fits}"
        )
        self.console.print(table)
