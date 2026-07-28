import csv
from collections import Counter

try:
    from .feature_preview_service import (
        RadiomicFeaturePreview,
        RadiomicFeaturePreviewService,
    )
    from .logger_conf import logger
    from .models import (
        DicomMetadataConfig,
        MirpExtractor,
        carica_pazienti,
        load_configuration,
    )
except ImportError:
    from feature_preview_service import (
        RadiomicFeaturePreview,
        RadiomicFeaturePreviewService,
    )
    from logger_conf import logger
    from models import (
        DicomMetadataConfig,
        MirpExtractor,
        carica_pazienti,
        load_configuration,
    )

from rich.console import Console
from rich.table import Table


MAX_FEATURE_NAMES_IN_CONSOLE = 500


def print_dicom_metadata_summary(metadata_config: DicomMetadataConfig) -> None:
    tags = metadata_config.active_tags()
    if not tags:
        logger.info("Metadati DICOM: nessun tag configurato per il CSV finale.")
        return

    logger.info("Metadati DICOM configurati per il CSV finale: %d tag", len(tags))
    table = Table(
        title="[bold]METADATI DICOM - TAG CONFIGURATI[cyan]",
        show_header=True,
        header_style="bold",
    )
    table.add_column("Colonna CSV", style="cyan", overflow="fold")
    table.add_column("Sorgente", style="green", no_wrap=True)
    table.add_column("Tag", style="magenta", no_wrap=True)
    table.add_column("Keyword", style="yellow", overflow="fold")

    for tag in tags:
        table.add_row(
            tag.output_column,
            tag.source.upper(),
            tag.tag_text,
            tag.keyword or "-",
        )

    Console(width=140).print(table)


def print_radiomic_feature_preview(
    preview: RadiomicFeaturePreview,
    *,
    max_feature_names: int = MAX_FEATURE_NAMES_IN_CONSOLE,
) -> None:
    logger.info(
        "Feature radiomiche previste: %d per riga/ROI, %d gruppi",
        preview.radiomic_feature_count,
        len(preview.groups),
    )

    console = Console(width=160)
    summary_table = Table(
        title="[bold]RECAP OUTPUT FEATURE - PRIMA DELL'ESTRAZIONE[cyan]",
        show_header=True,
        header_style="bold",
    )
    summary_table.add_column("Elemento", style="cyan", no_wrap=True)
    summary_table.add_column("Valore", style="green", overflow="fold")
    summary_table.add_row("Pazienti", str(preview.patient_count))
    summary_table.add_row("Gruppi radiomici", str(len(preview.groups)))
    summary_table.add_row("Feature radiomiche per riga/ROI", str(preview.radiomic_feature_count))
    summary_table.add_row("Metadati DICOM", str(len(preview.metadata_columns)))
    summary_table.add_row("Colonne CSV previste", str(preview.csv_column_count))
    console.print(summary_table)

    group_table = Table(
        title="[bold]GRUPPI DI FEATURE RADIOMICHE[cyan]",
        show_header=True,
        header_style="bold",
    )
    group_table.add_column("Sorgente", style="cyan", overflow="fold")
    group_table.add_column("Famiglie", style="yellow", overflow="fold")
    group_table.add_column("Feature", justify="right", style="green", no_wrap=True)
    group_table.add_column("Parametri", style="magenta", overflow="fold")

    for group in preview.groups:
        group_table.add_row(
            group.source,
            ", ".join(group.feature_families),
            str(group.count),
            "\n".join(group.parameters) or "-",
        )
    console.print(group_table)

    feature_table = Table(
        title="[bold]NOMI FEATURE RADIOMICHE PREVISTE[cyan]",
        show_header=True,
        header_style="bold",
    )
    feature_table.add_column("#", justify="right", style="cyan", no_wrap=True)
    feature_table.add_column("Sorgente", style="magenta", overflow="fold")
    feature_table.add_column("Feature", style="green", overflow="fold")

    rendered = 0
    for group in preview.groups:
        for feature_name in group.feature_names:
            if rendered >= max_feature_names:
                break
            rendered += 1
            feature_table.add_row(str(rendered), group.source, feature_name)
        if rendered >= max_feature_names:
            break
    console.print(feature_table)

    if preview.radiomic_feature_count > rendered:
        print(
            "Mostrate le prime "
            f"{rendered} feature su {preview.radiomic_feature_count}. "
            "La lista completa e' nel file di preview."
        )


def write_radiomic_feature_preview_csv(
    preview: RadiomicFeaturePreview,
    csv_path,
) -> None:
    with open(csv_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=";")
        writer.writerow(["tipo", "sorgente", "feature_family", "nome_colonna"])
        writer.writerow(["paziente", "csv", "-", "nome_cognome"])
        writer.writerow(["paziente", "csv", "-", "stato_microsatellitare"])
        for metadata_column in preview.metadata_columns:
            writer.writerow(["metadata_dicom", "dicom", "-", metadata_column])
        for group in preview.groups:
            for feature_name in group.feature_names:
                writer.writerow([
                    "radiomica",
                    group.source,
                    ", ".join(group.feature_families),
                    feature_name,
                ])


def ask_to_continue() -> bool:
    while True:
        try:
            answer = input("Continuare con l'estrazione radiomica? [s/N]: ")
        except EOFError:
            logger.warning("Input non disponibile: estrazione annullata.")
            return False

        answer = answer.strip().lower()
        if answer in {"s", "si", "sì", "y", "yes"}:
            return True
        if answer in {"", "n", "no"}:
            return False

        print("Rispondi con 's' per continuare oppure 'n' per annullare.")


def main() -> None:
    # Carica e valida i parametri scritti in config_extractor.yaml.
    config = load_configuration()

    # Cerca le coppie CT + RTStruct e associa lo stato microsatellitare dal CSV.
    pazienti = carica_pazienti(config)
    print(f"Pazienti caricati: {len(pazienti)}")

    if not pazienti:
        print("Nessun paziente valido trovato.")
        return

    if config.n_test is not False:
        numero_pazienti = len(pazienti)
        pazienti = pazienti[: config.n_test]
        print(
            "Modalità n-test attiva: "
            f"analizzo i primi {len(pazienti)} pazienti su {numero_pazienti}."
        )

    conteggio_stati = Counter(
        paziente.stato_microsatellitare.value
        for paziente in pazienti
        if paziente.stato_microsatellitare is not None
    )
    print(
        "Stato microsatellitare caricato: "
        + ", ".join(
            f"{stato}={quantita}"
            for stato, quantita in sorted(conteggio_stati.items())
        )
    )

    pazienti_senza_stato = [
        paziente.nome
        for paziente in pazienti
        if paziente.stato_microsatellitare is None
    ]
    if pazienti_senza_stato:
        print(
            "Attenzione: stato microsatellitare non disponibile per "
            f"{len(pazienti_senza_stato)} pazienti."
        )

    # Crea l'estrattore (Wrapper di MIRP) usando i parametri della configurazione.
    mirp_config = config.mirp
    if not mirp_config.export_features:
        print(
            "Nota: export_features era false; lo imposto a true per scrivere "
            "il CSV finale aggregato."
        )
        mirp_config = mirp_config.model_copy(update={"export_features": True})

    extractor = MirpExtractor(config=mirp_config)
    output_dir = config.data.ensure_output_dir()

    print_dicom_metadata_summary(config.dicom_metadata)
    feature_preview = RadiomicFeaturePreviewService(
        mirp_config=mirp_config,
        metadata_config=config.dicom_metadata,
        patient_count=len(pazienti),
    ).build()
    feature_preview_path = output_dir / "feature_preview.csv"
    write_radiomic_feature_preview_csv(feature_preview, feature_preview_path)
    print_radiomic_feature_preview(feature_preview)
    print(f"Lista completa delle colonne previste salvata in: {feature_preview_path}")

    if not ask_to_continue():
        print("Estrazione annullata.")
        return

    print(
        f"Avvio l'estrazione radiomica per {len(pazienti)} pazienti "
        "in modalità sequenziale."
    )
    risultati = extractor.extract_batch(
        pazienti,
        output_dir=output_dir,
        verbose=True,
    )

    numero_tabelle = sum(
        len(feature_tables or [])
        for feature_tables in risultati.values()
    )
    print(
        f"Estrazione completata: {len(risultati)} pazienti, "
        f"{numero_tabelle} tabelle restituite."
    )

    csv_path = config.data.ensure_features_csv_path()
    numero_righe = extractor.write_feature_csv(
        risultati=risultati,
        pazienti=pazienti,
        csv_path=csv_path,
        metadata_config=config.dicom_metadata,
    )
    print(f"Feature scritte in: {csv_path} ({numero_righe} righe).")


if __name__ == "__main__":
    main()
