from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pydicom
from pydicom.datadict import keyword_for_tag, tag_for_keyword
from pydicom.dataset import Dataset
from pydicom.errors import InvalidDicomError
from pydicom.multival import MultiValue
from pydicom.sequence import Sequence
from pydicom.tag import BaseTag, Tag


DEFAULT_MISSING_VALUE = "Non Disponibile"


DicomTagReference = str | int | tuple[int, int] | BaseTag


class EmptyDicomFileError(ValueError):
    """Il file DICOM esiste ma e' vuoto."""


def safe_resolve(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path


def filesystem_path(path: Path) -> Path:
    r"""Restituisce un path adatto alle operazioni sul filesystem.

    Su Windows usa il prefisso extended-length ``\\?\`` per evitare falsi
    negativi quando i file DICOM hanno percorsi piu' lunghi di 260 caratteri.
    """
    if os.name != "nt":
        return path

    path = path.expanduser()
    try:
        absolute_path = path.resolve(strict=False)
    except OSError:
        absolute_path = path.absolute()

    path_text = str(absolute_path)
    if path_text.startswith("\\\\?\\"):
        return Path(path_text)
    if path_text.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + path_text.lstrip("\\"))
    return Path("\\\\?\\" + path_text)


def windows_path_hint(path: Path) -> str:
    if os.name != "nt":
        return ""

    return (
        " Su Windows puo' succedere con percorsi molto lunghi: abilita "
        "LongPathsEnabled oppure sposta i dati in una cartella piu' corta, "
        "ad esempio C:\\radiomic-data."
    )


def candidate_dicom_files(folder: Path) -> list[Path]:
    filesystem_folder = filesystem_path(folder)
    try:
        files = sorted(
            (file for file in filesystem_folder.iterdir() if file.is_file()),
            key=lambda file: file.name.lower(),
        )
    except OSError as exc:
        raise OSError(
            f"Impossibile leggere la cartella DICOM '{safe_resolve(folder)}': "
            f"{exc}.{windows_path_hint(folder)}"
        ) from exc

    dcm_files = [file for file in files if file.suffix.lower() == ".dcm"]
    return dcm_files or files


@dataclass(frozen=True)
class DicomTagSpec:
    """Descrive un metadato DICOM da leggere."""

    tag: DicomTagReference
    key: str | None = None
    missing_value: Any = DEFAULT_MISSING_VALUE


@dataclass(frozen=True)
class DicomMetadataEntry:
    """Valore letto da un singolo tag DICOM."""

    key: str
    tag: str
    keyword: str
    name: str
    vr: str
    value: Any
    is_missing: bool = False


@dataclass(frozen=True)
class DicomMetadataRecord:
    """Metadati estratti da un file DICOM."""

    file_path: Path
    entries: dict[str, DicomMetadataEntry]

    def as_dict(self) -> dict[str, Any]:
        return {key: entry.value for key, entry in self.entries.items()}

    def as_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "file_path": str(self.file_path),
                "key": entry.key,
                "tag": entry.tag,
                "keyword": entry.keyword,
                "name": entry.name,
                "vr": entry.vr,
                "value": entry.value,
                "is_missing": entry.is_missing,
            }
            for entry in self.entries.values()
        ]


@dataclass(frozen=True)
class DicomDatasetReadResult:
    """Dataset DICOM valido letto da una cartella."""

    file_path: Path
    dataset: Dataset


class DicomMetadataService:
    """Legge e memorizza metadati DICOM a partire dai tag richiesti."""

    def __init__(self, missing_value: Any = DEFAULT_MISSING_VALUE) -> None:
        self.missing_value = missing_value
        self._dataset_cache: dict[Path, Dataset] = {}
        self._metadata_cache: dict[Path, DicomMetadataRecord] = {}

    def read_dataset(self, dicom_path: Path | str) -> Dataset:
        path = Path(dicom_path).expanduser()
        cache_key = safe_resolve(path)
        cached_dataset = self._dataset_cache.get(cache_key)
        if cached_dataset is not None:
            return cached_dataset

        self._validate_dicom_file(path)
        dataset = pydicom.dcmread(
            filesystem_path(path),
            stop_before_pixels=True,
            force=False,
        )
        self._dataset_cache[cache_key] = dataset
        return dataset

    def read_first_valid_dataset(self, folder: Path | str) -> DicomDatasetReadResult:
        cartella = Path(folder).expanduser()
        cartella_filesystem = filesystem_path(cartella)

        if not cartella_filesystem.exists():
            raise FileNotFoundError(
                f"La cartella DICOM non esiste: {safe_resolve(cartella)}"
                f"{windows_path_hint(cartella)}"
            )

        if not cartella_filesystem.is_dir():
            raise NotADirectoryError(
                f"Il percorso DICOM non e' una cartella: {safe_resolve(cartella)}"
            )

        file_dicom = candidate_dicom_files(cartella)
        if not file_dicom:
            raise FileNotFoundError(
                f"Nessun file trovato in: {safe_resolve(cartella)}"
                f"{windows_path_hint(cartella)}"
            )

        file_vuoti: list[Path] = []
        file_non_validi: list[tuple[Path, str]] = []

        for file_path in file_dicom:
            try:
                dataset = self.read_dataset(file_path)
            except EmptyDicomFileError:
                file_vuoti.append(file_path)
                continue
            except ValueError as exc:
                file_non_validi.append((file_path, str(exc)))
                continue
            except (InvalidDicomError, OSError) as exc:
                file_non_validi.append((file_path, str(exc)))
                continue

            if not self._has_main_dicom_identifiers(dataset):
                file_non_validi.append(
                    (file_path, "mancano i principali identificativi DICOM")
                )
                continue

            return DicomDatasetReadResult(file_path=file_path, dataset=dataset)

        dettagli: list[str] = []
        if file_vuoti:
            dettagli.append(
                "File vuoti ignorati:\n"
                + "\n".join(f"- {file.name}" for file in file_vuoti)
            )

        if file_non_validi:
            dettagli.append(
                "File non validi ignorati:\n"
                + "\n".join(
                    f"- {file.name}: {errore}"
                    for file, errore in file_non_validi
                )
            )

        descrizione_errori = "\n\n".join(dettagli)
        raise InvalidDicomError(
            f"Nessun file DICOM valido trovato in {safe_resolve(cartella)}."
            + (f"\n\n{descrizione_errori}" if descrizione_errori else "")
        )

    def extract_from_file(
        self,
        dicom_path: Path | str,
        tags: Iterable[DicomTagSpec | DicomTagReference],
    ) -> DicomMetadataRecord:
        path = Path(dicom_path).expanduser()
        dataset = self.read_dataset(path)
        record = DicomMetadataRecord(
            file_path=safe_resolve(path),
            entries=self.extract_from_dataset(dataset, tags),
        )
        self._metadata_cache[record.file_path] = record
        return record

    def extract_from_folder(
        self,
        folder: Path | str,
        tags: Iterable[DicomTagSpec | DicomTagReference],
    ) -> DicomMetadataRecord:
        result = self.read_first_valid_dataset(folder)
        record = DicomMetadataRecord(
            file_path=safe_resolve(result.file_path),
            entries=self.extract_from_dataset(result.dataset, tags),
        )
        self._metadata_cache[record.file_path] = record
        return record

    def extract_from_dataset(
        self,
        dataset: Dataset,
        tags: Iterable[DicomTagSpec | DicomTagReference],
    ) -> dict[str, DicomMetadataEntry]:
        entries: dict[str, DicomMetadataEntry] = {}
        for raw_spec in tags:
            spec = self._normalise_spec(raw_spec)
            tag = self.resolve_tag(spec.tag)
            keyword = keyword_for_tag(tag) or ""
            key = spec.key or keyword or self.format_tag(tag)
            entries[key] = self._build_entry(dataset, tag, key, spec.missing_value)

        return entries

    def get_tag_value(
        self,
        dataset: Dataset,
        tag: DicomTagReference,
        default: Any | None = None,
    ) -> Any:
        missing_value = self.missing_value if default is None else default
        entry = self._build_entry(
            dataset=dataset,
            tag=self.resolve_tag(tag),
            key=str(tag),
            missing_value=missing_value,
        )
        return entry.value

    def get_cached_metadata(
        self,
        dicom_path: Path | str,
    ) -> DicomMetadataRecord | None:
        return self._metadata_cache.get(safe_resolve(Path(dicom_path).expanduser()))

    def clear_cache(self) -> None:
        self._dataset_cache.clear()
        self._metadata_cache.clear()

    @staticmethod
    def resolve_tag(tag_reference: DicomTagReference) -> BaseTag:
        if isinstance(tag_reference, BaseTag):
            return tag_reference

        if isinstance(tag_reference, tuple):
            return Tag(tag_reference)

        if isinstance(tag_reference, int):
            return Tag(tag_reference)

        tag_text = str(tag_reference).strip()
        keyword_tag = tag_for_keyword(tag_text)
        if keyword_tag is not None:
            return Tag(keyword_tag)

        cleaned = tag_text.lower().replace("0x", "")
        cleaned = re.sub(r"[^0-9a-f]", "", cleaned)
        if len(cleaned) != 8:
            raise ValueError(f"Tag DICOM non riconosciuto: {tag_reference}")

        return Tag(int(cleaned, 16))

    @staticmethod
    def format_tag(tag: BaseTag) -> str:
        return f"({tag.group:04X},{tag.element:04X})"

    @staticmethod
    def _normalise_spec(
        spec: DicomTagSpec | DicomTagReference,
    ) -> DicomTagSpec:
        if isinstance(spec, DicomTagSpec):
            return spec
        return DicomTagSpec(tag=spec)

    def _build_entry(
        self,
        dataset: Dataset,
        tag: BaseTag,
        key: str,
        missing_value: Any,
    ) -> DicomMetadataEntry:
        element = dataset.get(tag)
        keyword = keyword_for_tag(tag) or ""

        if element is None:
            return DicomMetadataEntry(
                key=key,
                tag=self.format_tag(tag),
                keyword=keyword,
                name="",
                vr="",
                value=missing_value,
                is_missing=True,
            )

        return DicomMetadataEntry(
            key=key,
            tag=self.format_tag(tag),
            keyword=element.keyword or keyword,
            name=element.name,
            vr=element.VR,
            value=self._serialise_value(element.value),
            is_missing=False,
        )

    @staticmethod
    def _serialise_value(value: Any) -> Any:
        if isinstance(value, Sequence):
            return f"[Sequence: {len(value)} item(s)]"

        if isinstance(value, MultiValue):
            return [DicomMetadataService._serialise_value(item) for item in value]

        if isinstance(value, bytes):
            return value.hex()

        if isinstance(value, Mapping):
            return dict(value)

        return str(value)

    @staticmethod
    def _validate_dicom_file(path: Path) -> None:
        filesystem_file = filesystem_path(path)
        if not filesystem_file.exists():
            raise FileNotFoundError(f"File DICOM non trovato: {safe_resolve(path)}")
        if not filesystem_file.is_file():
            raise IsADirectoryError(f"Il percorso non e' un file: {safe_resolve(path)}")
        if filesystem_file.stat().st_size == 0:
            raise EmptyDicomFileError(f"Il file DICOM e' vuoto: {safe_resolve(path)}")

    @staticmethod
    def _has_main_dicom_identifiers(dataset: Dataset) -> bool:
        identifiers = (
            "SOPClassUID",
            "SOPInstanceUID",
            "StudyInstanceUID",
            "SeriesInstanceUID",
        )
        return any(hasattr(dataset, name) for name in identifiers)
