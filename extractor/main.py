import csv
import os
import sys
import time
from collections import Counter
from math import ceil
from pathlib import Path

try:
    from .config_profiles import save_configuration_snapshot
    from .feature_preview_service import (
        RadiomicFeaturePreview,
        RadiomicFeaturePreviewService,
    )
    from .logger_conf import logger
    from .models import (
        CONFIG_FILE,
        DicomMetadataConfig,
        MirpExtractor,
        carica_pazienti,
        load_configuration,
    )
except ImportError:
    from config_profiles import save_configuration_snapshot
    from feature_preview_service import (
        RadiomicFeaturePreview,
        RadiomicFeaturePreviewService,
    )
    from logger_conf import logger
    from models import (
        CONFIG_FILE,
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


def ask_to_continue(timeout_seconds: int = 60) -> bool:
    """Attende un eventuale annullamento, poi avvia automaticamente."""
    if timeout_seconds <= 0:
        return True

    print(
        "Premi N per annullare oppure S/Invio per iniziare subito. "
        f"Senza risposta l'estrazione inizierà tra {timeout_seconds} secondi."
    )

    if os.name == "nt":
        return _ask_to_continue_windows(timeout_seconds)
    return _ask_to_continue_posix(timeout_seconds)


def _ask_to_continue_windows(timeout_seconds: int) -> bool:
    import msvcrt

    deadline = time.monotonic() + timeout_seconds
    last_remaining: int | None = None
    while time.monotonic() < deadline:
        remaining = max(0, ceil(deadline - time.monotonic()))
        if remaining != last_remaining:
            print(f"\rAvvio automatico tra {remaining:2d} secondi...", end="", flush=True)
            last_remaining = remaining

        if msvcrt.kbhit():
            key = msvcrt.getwch().lower()
            if key in {"\x00", "\xe0"}:
                msvcrt.getwch()  # Secondo carattere dei tasti speciali Windows.
                continue
            if key in {"n", "\x03", "\x1b"}:
                print("\rEstrazione annullata.                         ")
                return False
            if key in {"s", "y", "\r"}:
                print("\rAvvio immediato dell'estrazione.             ")
                return True
        time.sleep(0.1)

    print("\rTempo scaduto: avvio automatico dell'estrazione.    ")
    return True


def _ask_to_continue_posix(timeout_seconds: int) -> bool:
    """Fallback per terminali non Windows."""
    import select

    ready, _, _ = select.select([sys.stdin], [], [], timeout_seconds)
    if not ready:
        print("Tempo scaduto: avvio automatico dell'estrazione.")
        return True

    answer = sys.stdin.readline().strip().lower()
    if answer in {"n", "no"}:
        print("Estrazione annullata.")
        return False
    return True


def main(config_path: str | Path = CONFIG_FILE) -> None:
    # Carica e valida i parametri scritti in config/config_extractor.yaml.
    config = load_configuration(config_path)

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
            "Modalità n_test attiva: "
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

    # MIRP riceve un solo spacing alla volta, mentre lo YAML può richiederne
    # diversi per lo stesso esperimento.
    mirp_config = config.mirp
    if not mirp_config.export_features:
        print(
            "Nota: export_features era false; lo imposto a true per scrivere "
            "il CSV finale aggregato."
        )
        mirp_config = mirp_config.model_copy(update={"export_features": True})

    voxel_spacings = mirp_config.voxel_spacings()
    preview_config = mirp_config.for_voxel_spacing(voxel_spacings[0])

    print_dicom_metadata_summary(config.dicom_metadata)
    feature_preview = RadiomicFeaturePreviewService(
        mirp_config=preview_config,
        metadata_config=config.dicom_metadata,
        patient_count=len(pazienti),
    ).build()
    print_radiomic_feature_preview(feature_preview)
    print(
        "Voxel spacing isotropici richiesti: "
        + ", ".join(f"{spacing[0]:g} mm" for spacing in voxel_spacings)
    )

    if not ask_to_continue():
        print("Estrazione annullata.")
        return

    experiment_dir = config.data.create_experiment_dir()
    saved_config_path = save_configuration_snapshot(
        config, experiment_dir, mirp_config=mirp_config,
    )
    feature_preview_path = experiment_dir / "feature_preview.csv"
    write_radiomic_feature_preview_csv(feature_preview, feature_preview_path)

    print(f"Cartella dell'esperimento: {experiment_dir}")
    print(f"Configurazione completa salvata in: {saved_config_path}")
    print(f"Lista completa delle colonne previste salvata in: {feature_preview_path}")
    print(
        f"Avvio l'estrazione radiomica per {len(pazienti)} pazienti "
        f"e {len(voxel_spacings)} voxel spacing, in modalità sequenziale."
    )

    csv_summaries: list[tuple[Path, int]] = []
    for spacing_index, voxel_spacing in enumerate(voxel_spacings, start=1):
        active_config = mirp_config.for_voxel_spacing(voxel_spacing)
        extractor = MirpExtractor(config=active_config)
        csv_path = config.data.ensure_features_csv_path(
            output_dir=experiment_dir,
            voxel_spacing=voxel_spacing,
        )
        print(
            f"[{spacing_index}/{len(voxel_spacings)}] "
            f"Voxel spacing {voxel_spacing[0]:g} mm: {csv_path}"
        )

        numero_righe = 0
        csv_con_righe = False

        def write_completed_patient(paziente, feature_tables) -> None:
            nonlocal numero_righe, csv_con_righe

            righe_scritte = extractor.write_feature_csv(
                risultati={paziente.nome: feature_tables},
                pazienti=[paziente],
                csv_path=csv_path,
                metadata_config=config.dicom_metadata,
                append=csv_con_righe,
            )
            numero_righe += righe_scritte
            csv_con_righe = csv_con_righe or righe_scritte > 0

        mirp_output_dir = experiment_dir
        if active_config.write_features:
            spacing_label = format(voxel_spacing[0], ".12g").replace(".", "p")
            mirp_output_dir = experiment_dir / f"mirp_voxel-spacing-{spacing_label}mm"
            mirp_output_dir.mkdir(parents=True, exist_ok=True)

        risultati = extractor.extract_batch(
            pazienti,
            output_dir=mirp_output_dir,
            verbose=True,
            on_patient_completed=write_completed_patient,
        )

        numero_tabelle = sum(
            len(feature_tables or [])
            for feature_tables in risultati.values()
        )
        print(
            f"Spacing {voxel_spacing[0]:g} mm completato: "
            f"{len(risultati)} pazienti, {numero_tabelle} tabelle restituite."
        )
        csv_summaries.append((csv_path, numero_righe))

    print("Estrazione completata. CSV prodotti:")
    for csv_path, numero_righe in csv_summaries:
        print(f"- {csv_path} ({numero_righe} righe)")


if __name__ == "__main__":
    main()
