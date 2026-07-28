from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError
from pydicom.dataset import Dataset
from pydicom.errors import InvalidDicomError

from extractor.dicom_metadata_service import (
    DicomMetadataRecord,
    DicomMetadataService,
)
from extractor.models import ClinicalDataLoader, PatientScanner, Paziente


DEFAULT_DATA_ROOT = Path("../radiomic-data")
DEFAULT_PATIENTS_CSV_NAME = "pazienti.csv"
DICOM_METADATA_SERVICE = DicomMetadataService()


@dataclass(frozen=True)
class PatientListRow:
    nome: str
    dicom_id: str
    data_studio: str
    tipo: str


@dataclass(frozen=True)
class CsvPatientRow:
    nome: str
    dicom_id: str
    tipo: str


def read_dicom_metadata(dicom_path: Path) -> Dataset:
    return DICOM_METADATA_SERVICE.read_dataset(dicom_path)


def print_dicom_info(dicom_path: Path, dataset: Dataset) -> None:
    print(f"File: {dicom_path.resolve()}")
    print()
    print(dataset)


def print_dicom_metadata_record(record: DicomMetadataRecord) -> None:
    print(f"File: {record.file_path}")
    print_table(
        headers=("Key", "Tag", "Keyword", "Valore"),
        table_rows=[
            (
                entry.key,
                entry.tag,
                entry.keyword or "-",
                str(entry.value),
            )
            for entry in record.entries.values()
        ],
    )


def load_patients_from_data(data_root: Path) -> list[Paziente]:
    logging.getLogger("extractor.logger_conf").setLevel(logging.ERROR)

    return PatientScanner(patients_folder=data_root).load_patients()


def read_csv_patient_rows(patients_csv: Path) -> dict[str, CsvPatientRow]:
    if not patients_csv.exists():
        return {}

    records = ClinicalDataLoader(csv_path=patients_csv)._read_records()
    return {
        dicom_id: CsvPatientRow(
            nome=patient_name,
            dicom_id=dicom_id,
            tipo=status.value,
        )
        for dicom_id, (patient_name, status, _row_number) in records.items()
    }


def load_found_patients(data_root: Path, patients_csv: Path) -> list[Paziente]:
    patients = load_patients_from_data(data_root)
    if not patients_csv.exists():
        return patients

    return ClinicalDataLoader(csv_path=patients_csv).assign_status(patients)


def build_patient_rows(patients: list[Paziente]) -> list[PatientListRow]:
    rows: list[PatientListRow] = []

    for patient in patients:
        status = patient.stato_microsatellitare
        rows.append(
            PatientListRow(
                nome=patient.nome,
                dicom_id=patient.get_patient_id(),
                data_studio=patient.get_study_date(),
                tipo=status.value if status is not None else "NON DISPONIBILE",
            )
        )

    return rows


def format_study_date(value: str) -> str:
    if len(value) == 8 and value.isdigit():
        return f"{value[0:4]}-{value[4:6]}-{value[6:8]}"
    return value


def normalise_dicom_id(value: str) -> str:
    return value.strip()


def sort_patient_rows(
    rows: list[PatientListRow],
    sort_by: str,
    descending: bool = False,
) -> list[PatientListRow]:
    if sort_by == "data":
        missing_dates = {"", "Non Disponibile"}
        dated_rows = [
            row for row in rows if row.data_studio not in missing_dates
        ]
        undated_rows = [
            row for row in rows if row.data_studio in missing_dates
        ]
        dated_rows = sorted(
            dated_rows,
            key=lambda row: (row.data_studio, row.nome.casefold()),
            reverse=descending,
        )
        undated_rows = sorted(undated_rows, key=lambda row: row.nome.casefold())
        return dated_rows + undated_rows

    return sorted(rows, key=lambda row: row.nome.casefold(), reverse=descending)


def print_table(headers: tuple[str, ...], table_rows: list[tuple[str, ...]]) -> None:
    widths = [
        max(len(str(value)) for value in column)
        for column in zip(headers, *table_rows, strict=False)
    ]

    header_line = "  ".join(
        header.ljust(width) for header, width in zip(headers, widths, strict=True)
    )
    separator_line = "  ".join("-" * width for width in widths)

    print(header_line)
    print(separator_line)
    for table_row in table_rows:
        print(
            "  ".join(
                value.ljust(width)
                for value, width in zip(table_row, widths, strict=True)
            )
        )


def print_patient_rows(rows: list[PatientListRow]) -> None:
    print_table(
        headers=("Nome", "Data studio", "DicomID", "Tipo"),
        table_rows=[
            (row.nome, format_study_date(row.data_studio), row.dicom_id, row.tipo)
            for row in rows
        ],
    )


def print_csv_rows(rows: list[CsvPatientRow]) -> None:
    print_table(
        headers=("Nome CSV", "DicomID", "Tipo"),
        table_rows=[
            (row.nome, row.dicom_id, row.tipo)
            for row in rows
        ],
    )


def print_data_only_rows(rows: list[PatientListRow]) -> None:
    print_table(
        headers=("Nome", "Data studio", "DicomID"),
        table_rows=[
            (row.nome, format_study_date(row.data_studio), row.dicom_id)
            for row in rows
        ],
    )


def print_patient_recap(rows: list[PatientListRow]) -> None:
    total = len(rows)
    stable_count = sum(row.tipo == "STABILE" for row in rows)
    unstable_count = sum(row.tipo == "INSTABILE" for row in rows)
    unknown_count = total - stable_count - unstable_count

    print()
    print("Recap")
    print(f"Totale pazienti: {total}")
    print(f"Stabili: {stable_count}")
    print(f"Instabili: {unstable_count}")
    if unknown_count:
        print(f"Tipo non disponibile: {unknown_count}")


def print_patient_differences(
    rows: list[PatientListRow],
    csv_rows_by_id: dict[str, CsvPatientRow],
    csv_path: Path,
) -> None:
    data_dicom_ids = {
        normalise_dicom_id(row.dicom_id)
        for row in rows
        if normalise_dicom_id(row.dicom_id) != "Non Disponibile"
    }
    csv_only_rows = sorted(
        (
            csv_row
            for dicom_id, csv_row in csv_rows_by_id.items()
            if normalise_dicom_id(dicom_id) not in data_dicom_ids
        ),
        key=lambda row: row.nome.casefold(),
    )
    data_only_rows = [
        row
        for row in rows
        if normalise_dicom_id(row.dicom_id) not in csv_rows_by_id
    ]

    print()
    print(f"Trovati nel CSV ma non nella cartella dati ({len(csv_only_rows)})")
    if csv_rows_by_id:
        if csv_only_rows:
            print_csv_rows(csv_only_rows)
        else:
            print("Nessuno.")
    else:
        print(f"CSV non trovato o vuoto: {csv_path}")

    print()
    print(f"Trovati nella cartella dati ma non nel CSV ({len(data_only_rows)})")
    if data_only_rows:
        print_data_only_rows(data_only_rows)
    else:
        print("Nessuno.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CLI per utility radiomiche e ispezione dei file DICOM."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--infodicom",
        metavar="PATH",
        type=Path,
        help="Mostra tutti i metadati contenuti in un file DICOM.",
    )
    group.add_argument(
        "--pazienti",
        action="store_true",
        help=(
            "Mostra i pazienti trovati nella cartella dati; associa il tipo "
            "dal CSV quando disponibile."
        ),
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        help=(
            "Tag DICOM da leggere con --infodicom. Accetta keyword "
            "come PatientID oppure tag come (0010,0020). Ripetibile."
        ),
    )
    parser.add_argument(
        "--cartella-dati",
        metavar="PATH",
        type=Path,
        help=(
            "Cartella radice dei dati radiomici. Usato senza --pazienti, "
            "mostra comunque tutti i pazienti trovati nella cartella "
            f"(default con --pazienti: {DEFAULT_DATA_ROOT})."
        ),
    )
    parser.add_argument(
        "--csv-pazienti",
        metavar="PATH",
        type=Path,
        help=(
            "CSV opzionale con DicomID, NomeCognome e Tipo "
            f"(default: <cartella-dati>/{DEFAULT_PATIENTS_CSV_NAME})."
        ),
    )
    parser.add_argument(
        "--ordina",
        choices=("nome", "data"),
        default="nome",
        help="Ordinamento della lista pazienti (default: nome).",
    )
    parser.add_argument(
        "--discendente",
        action="store_true",
        help="Usa ordinamento discendente per la lista pazienti.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    patient_listing_requested = (
        args.pazienti
        or args.cartella_dati is not None
        or args.csv_pazienti is not None
    )

    if args.tag and args.infodicom is None:
        parser.error("--tag puo' essere usato solo insieme a --infodicom")

    if args.infodicom is None and not patient_listing_requested:
        parser.print_help()
        return 0

    if args.infodicom is not None:
        try:
            if args.tag:
                record = DICOM_METADATA_SERVICE.extract_from_file(
                    args.infodicom,
                    args.tag,
                )
            else:
                dataset = read_dicom_metadata(args.infodicom)
        except (
            FileNotFoundError,
            IsADirectoryError,
            ValueError,
            InvalidDicomError,
        ) as error:
            print(f"Errore: {error}", file=sys.stderr)
            return 1

        if args.tag:
            print_dicom_metadata_record(record)
        else:
            print_dicom_info(args.infodicom, dataset)
        return 0

    data_root = (
        args.cartella_dati.expanduser()
        if args.cartella_dati is not None
        else DEFAULT_DATA_ROOT
    )
    patients_csv = (
        args.csv_pazienti.expanduser()
        if args.csv_pazienti is not None
        else data_root / DEFAULT_PATIENTS_CSV_NAME
    )

    try:
        csv_rows_by_id = read_csv_patient_rows(patients_csv)
        patients = load_found_patients(data_root, patients_csv)
        rows = sort_patient_rows(
            build_patient_rows(patients),
            sort_by=args.ordina,
            descending=args.discendente,
        )
    except (
        FileNotFoundError,
        IsADirectoryError,
        NotADirectoryError,
        ValueError,
        ValidationError,
        InvalidDicomError,
        OSError,
    ) as error:
        print(f"Errore: {error}", file=sys.stderr)
        return 1

    if rows:
        print_patient_rows(rows)
    else:
        print("Nessun paziente trovato.")

    print_patient_recap(rows)
    print_patient_differences(rows, csv_rows_by_id, patients_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
