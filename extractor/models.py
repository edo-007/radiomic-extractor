from __future__ import annotations

import csv
import logging
import re
from collections import defaultdict
from enum import Enum
from math import isnan
from pathlib import Path
from typing import Any, Iterable, Literal

import yaml

from pydantic import (
    AliasChoices,
    BaseModel, 
    ConfigDict,
    DirectoryPath, 
    Field, 
    FilePath, 
    field_validator
)
from pydicom.datadict import keyword_for_tag
from pydicom.dataset import Dataset

try:
    from .dicom_metadata_service import (
        DEFAULT_MISSING_VALUE,
        DicomMetadataRecord,
        DicomMetadataService,
        DicomTagReference,
        DicomTagSpec,
        candidate_dicom_files,
        filesystem_path,
        safe_resolve,
        windows_path_hint,
    )
    from .logger_conf import logger
except ImportError:
    from dicom_metadata_service import (
        DEFAULT_MISSING_VALUE,
        DicomMetadataRecord,
        DicomMetadataService,
        DicomTagReference,
        DicomTagSpec,
        candidate_dicom_files,
        filesystem_path,
        safe_resolve,
        windows_path_hint,
    )
    from logger_conf import logger

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_ROOT / "config_extractor.yaml"
_DICOM_METADATA_SERVICE = DicomMetadataService()
METADATA_COLUMN_PREFIX = "metadata_"


def _safe_resolve(path: Path) -> Path:
    return safe_resolve(path)


def _filesystem_path(path: Path) -> Path:
    return filesystem_path(path)


def _windows_path_hint(path: Path) -> str:
    return windows_path_hint(path)


def _candidate_dicom_files(folder: Path) -> list[Path]:
    return candidate_dicom_files(folder)


def _normalise_metadata_name(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("Il nome del metadato DICOM non puo' essere vuoto")

    value = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", value)
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    if not value:
        raise ValueError("Il nome del metadato DICOM non contiene caratteri validi")
    return value

class DataConfig(BaseModel):
    """Percorsi dei dati in input e output."""

    patients_folder: DirectoryPath
    patients_csv: FilePath | None = None
    output_dir: Path = Path("./results")
    features_csv: Path = Path("features.csv")

    def ensure_output_dir(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir

    def ensure_features_csv_path(self) -> Path:
        output_dir = self.ensure_output_dir()
        csv_path = self.features_csv.expanduser()
        if not csv_path.is_absolute():
            csv_path = output_dir / csv_path
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        return csv_path


class MirpConfig(BaseModel):
    """Parametri necessari per l'estrazione radiomica con MIRP."""

    ibsi_compliant: bool = True
    bin_width: float = Field(25.0, gt=0)
    voxel_spacing: list[float] = Field(default_factory=lambda: [1.0, 1.0, 1.0])
    roi_names: list[str] | None = None
    resegmentation_intensity_range: list[float] | None = Field(
        default_factory=lambda: [-1000.0, float("nan")]
    )
    feature_families: list[str] = Field(
        default_factory=lambda: ["statistics", "intensity_histogram"]
    )
    by_slice: bool = False
    num_processes: int | None = Field(
        1,
        validation_alias=AliasChoices("num_processes", "num_cpus"),
    )
    write_features: bool = False
    export_features: bool = True

    @field_validator("voxel_spacing")
    @classmethod
    def validate_voxel_spacing(cls, 
        value: list[float]
    ) -> list[float]:
        if len(value) != 3:
            raise ValueError("voxel_spacing deve contenere tre valori: z, y, x")
        if any(spacing <= 0 for spacing in value):
            raise ValueError("voxel_spacing deve contenere solo valori positivi")
        return value

    @field_validator("num_processes")
    @classmethod
    def validate_num_processes(cls, value: int | None) -> int | None:
        if value is not None and (value < -1 or value == 0):
            raise ValueError(
                "num_processes deve essere -1 (tutti i core) oppure un "
                "numero intero >= 1"
            )
        return value


    @field_validator("resegmentation_intensity_range")
    @classmethod
    def validate_intensity_range(cls, 
        value: list[float] | None
    ) -> list[float] | None:
        if value is None:
            return value
        if len(value) != 2:
            raise ValueError("resegmentation_intensity_range deve contenere due valori")
        lower, upper = value
        if not isnan(upper) and lower >= upper:
            raise ValueError(
                "resegmentation_intensity_range deve avere limite inferiore minore del superiore"
            )
        return value

    def to_mirp_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "image_file_type": "dicom",
            "mask_file_type": "dicom",
            "image_modality": "ct",
            "mask_modality": "rtstruct",
            "association_strategy": "frame_of_reference",
            "ibsi_compliant": self.ibsi_compliant,
            "by_slice": self.by_slice,
            "new_spacing": self.voxel_spacing,
            "base_feature_families": self.feature_families,
            "base_discretisation_method": "fixed_bin_size",
            "base_discretisation_bin_width": float(self.bin_width),
        }

        if self.roi_names:
            kwargs["roi_name"] = self.roi_names

        if self.resegmentation_intensity_range:
            kwargs["resegmentation_intensity_range"] = self.resegmentation_intensity_range

        return kwargs


class DicomMetadataTagConfig(BaseModel):
    """Singolo tag DICOM da aggiungere al CSV delle feature."""

    model_config = ConfigDict(extra="forbid")

    tag: str
    source: Literal["ct", "rt"] = "ct"
    missing_value: Any = DEFAULT_MISSING_VALUE

    @field_validator("tag")
    @classmethod
    def validate_tag(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Il tag DICOM non puo' essere vuoto")

        DicomMetadataService.resolve_tag(value)
        return value

    @field_validator("source", mode="before")
    @classmethod
    def normalise_source(cls, value: Any) -> str:
        if value is None:
            return "ct"

        source = str(value).strip().lower()
        if source in {"rtstruct", "rt_struct"}:
            return "rt"
        return source

    @property
    def resolved_tag(self):
        return DicomMetadataService.resolve_tag(self.tag)

    @property
    def tag_text(self) -> str:
        return DicomMetadataService.format_tag(self.resolved_tag)

    @property
    def keyword(self) -> str:
        return keyword_for_tag(self.resolved_tag) or ""

    @property
    def output_column(self) -> str:
        base_name = self.keyword or self.tag
        column = _normalise_metadata_name(base_name)
        if column.startswith(METADATA_COLUMN_PREFIX):
            return column
        return f"{METADATA_COLUMN_PREFIX}{column}"

    def to_dicom_tag_spec(self) -> DicomTagSpec:
        return DicomTagSpec(
            tag=self.tag,
            key=self.output_column,
            missing_value=self.missing_value,
        )


class DicomMetadataConfig(BaseModel):
    """Configurazione dei metadati DICOM da aggiungere al CSV."""

    enabled: bool = True
    tags: list[DicomMetadataTagConfig] = Field(default_factory=list)

    @field_validator("tags", mode="before")
    @classmethod
    def normalise_tags(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, str):
            return [{"tag": value}]
        if not isinstance(value, list):
            raise TypeError("dicom_metadata.tags deve essere una lista")

        normalised_tags: list[Any] = []
        for item in value:
            if isinstance(item, str):
                normalised_tags.append({"tag": item})
            else:
                normalised_tags.append(item)
        return normalised_tags

    @field_validator("tags")
    @classmethod
    def validate_unique_columns(
        cls,
        value: list[DicomMetadataTagConfig],
    ) -> list[DicomMetadataTagConfig]:
        columns = [tag.output_column for tag in value]
        duplicates = sorted(
            column for column in set(columns) if columns.count(column) > 1
        )
        if duplicates:
            raise ValueError(
                "Colonne metadata DICOM duplicate: " + ", ".join(duplicates)
            )
        return value

    def active_tags(self) -> list[DicomMetadataTagConfig]:
        if not self.enabled:
            return []
        return self.tags

    def has_tags(self) -> bool:
        return bool(self.active_tags())


class GlobalConfig(BaseModel):
    """Schema completo del file YAML."""

    data: DataConfig
    dicom_metadata: DicomMetadataConfig = Field(
        default_factory=DicomMetadataConfig
    )
    mirp: MirpConfig = Field(default_factory=MirpConfig)
    n_test: int | Literal[False] = Field(
        default=False,
        validation_alias=AliasChoices("n-test", "n_test"),
    )

    @field_validator("n_test", mode="before")
    @classmethod
    def validate_n_test(cls, value: Any) -> Any:
        if value is False:
            return value
        if value is True:
            raise ValueError("n-test deve essere false oppure un intero >= 1")
        if isinstance(value, int) and value >= 1:
            return value
        raise ValueError("n-test deve essere false oppure un intero >= 1")


class ConfigManager(BaseModel):
    config_path: FilePath = Path(CONFIG_FILE)

    def load_configuration(self) -> GlobalConfig:
        """Carica lo YAML e lo convalida tramite i modelli Pydantic."""
        with open(self.config_path, "r", encoding="utf-8") as yaml_file:
            raw_data = yaml.safe_load(yaml_file) or {}

        raw_data = self._normalise_legacy_config(raw_data)
        raw_data = self._resolve_relative_paths(raw_data)
        return GlobalConfig(**raw_data)

    def _normalise_legacy_config(self, raw_data: dict[str, Any]) -> dict[str, Any]:
        if "data" not in raw_data and "patients_folder" in raw_data:
            raw_data = {
                "data": {
                    "patients_folder": raw_data.pop("patients_folder"),
                    "output_dir": raw_data.pop("output_dir", "./results"),
                },
                **raw_data,
            }
        return raw_data

    def _resolve_relative_paths(self, raw_data: dict[str, Any]) -> dict[str, Any]:
        config_dir = Path(self.config_path).resolve().parent
        data = raw_data.get("data", {})

        for key in ("patients_folder", "patients_csv", "output_dir"):
            value = data.get(key)
            if value is None:
                continue

            path = Path(value).expanduser()
            if not path.is_absolute():
                data[key] = config_dir / path

        raw_data["data"] = data
        return raw_data


class StatoMicrosatellitare(str, Enum):
    STABILE = "STABILE"
    INSTABILE = "INSTABILE"


class Paziente(BaseModel):
    nome: str
    path_ct: Path
    path_rt: Path
    stato_microsatellitare: StatoMicrosatellitare | None = None

    @field_validator("path_ct", "path_rt")
    @classmethod
    def ensure_path_contains_dicoms(cls, value: Path) -> Path:
        filesystem_path = _filesystem_path(value)

        if not filesystem_path.exists():
            raise ValueError(
                f"La cartella DICOM non esiste: {_safe_resolve(value)}"
                f"{_windows_path_hint(value)}"
            )

        if not filesystem_path.is_dir():
            raise ValueError(
                f"Il percorso DICOM non e' una cartella: {_safe_resolve(value)}"
            )

        if not _candidate_dicom_files(value):
            raise ValueError(
                f"La cartella '{value}' non contiene file DICOM leggibili."
                f"{_windows_path_hint(value)}"
            )
        return value
        



    def _get_dicom_dataset(self, use_rt: bool = False) -> Dataset:
        """Legge i metadati del primo file DICOM valido trovato.

        Seleziona la cartella CT o RT, ignora file vuoti e file non validi,
        senza caricare i Pixel Data in memoria.

        Args:
            use_rt: Se True usa ``self.path_rt``, altrimenti ``self.path_ct``.

        Returns:
            Il primo dataset DICOM valido trovato.

        Raises:
            FileNotFoundError: Se la cartella non esiste o non contiene file
                con estensione ``.dcm``.
            InvalidDicomError: Se nessun file ``.dcm`` non vuoto è un DICOM
                valido.
        """
        cartella = Path(self.path_rt if use_rt else self.path_ct)
        return _DICOM_METADATA_SERVICE.read_first_valid_dataset(cartella).dataset

    def get_dicom_metadata_record(
        self,
        tags: Iterable[DicomTagSpec | DicomTagReference],
        use_rt: bool = False,
    ) -> DicomMetadataRecord:
        """Estrae e memorizza metadati DICOM per i tag richiesti."""
        cartella = Path(self.path_rt if use_rt else self.path_ct)
        return _DICOM_METADATA_SERVICE.extract_from_folder(cartella, tags)

    def get_dicom_metadata(
        self,
        tags: Iterable[DicomTagSpec | DicomTagReference],
        use_rt: bool = False,
    ) -> dict[str, Any]:
        """Restituisce i metadati DICOM come dizionario key -> valore."""
        return self.get_dicom_metadata_record(tags=tags, use_rt=use_rt).as_dict()

    def get_dicom_tag_value(
        self,
        tag: DicomTagReference,
        use_rt: bool = False,
        default: Any = DEFAULT_MISSING_VALUE,
    ) -> Any:
        """Legge un singolo valore DICOM da keyword o tag numerico."""
        dataset = self._get_dicom_dataset(use_rt=use_rt)
        return _DICOM_METADATA_SERVICE.get_tag_value(
            dataset=dataset,
            tag=tag,
            default=default,
        )

    # tag: (0010,0020)
    def get_patient_id(self, use_rt: bool = False) -> str:
        """Estrae il Patient ID.

        Di default legge da path_ct, se use_rt=True legge da path_rt.
        """
        return str(self.get_dicom_tag_value("PatientID", use_rt=use_rt))

    # tag: (0010,0010)
    def get_patient_name(self, use_rt: bool = False) -> str:
        """Estrae il Patient Name dal DICOM CT o RTStruct."""
        return str(self.get_dicom_tag_value("PatientName", use_rt=use_rt))

    # tag: (0008,0020)
    def get_study_date(self, use_rt: bool = False) -> str:
        """Estrae lo Study Date.

        Di default legge da path_ct, se use_rt=True legge da path_rt.
        """
        return str(self.get_dicom_tag_value("StudyDate", use_rt=use_rt))


class PatientScanner(BaseModel):
    patients_folder: DirectoryPath

    def load_patients(self) -> list[Paziente]:
        patients_data: dict[str, dict[str, Path]] = defaultdict(dict)
        logger.info("Inizio scansione cartella pazienti: %s", self.patients_folder)

        for path in sorted(self.patients_folder.rglob("*")):
            if not path.is_dir():
                continue

            patient_name = self._extract_patient_name(path.name)
            if patient_name is None:
                continue

            if "_CT_" in path.name:
                self._store_path(patients_data, patient_name, "path_ct", path)
            elif "_RTst_" in path.name:
                self._store_path(patients_data, patient_name, "path_rt", path)

        patients = self._build_patients(patients_data)
        logger.info(
            "Creati %d oggetti Paziente validi su %d trovati",
            len(patients),
            len(patients_data),
        )
        return patients

    def _extract_patient_name(self, folder_name: str) -> str | None:
        match = re.match(r"^(.*?)_", folder_name)
        if not match:
            return None
        return match.group(1).replace("^", "_").strip()

    def _store_path(
        self,
        patients_data: dict[str, dict[str, Path]],
        patient_name: str,
        key: str,
        path: Path,
    ) -> None:
        if key in patients_data[patient_name]:
            logger.warning(
                "Percorso %s duplicato per il paziente '%s': uso '%s'",
                key,
                patient_name,
                path,
            )
        patients_data[patient_name][key] = path

    def _build_patients(
        self, patients_data: dict[str, dict[str, Path]]
    ) -> list[Paziente]:
        patients: list[Paziente] = []

        for name, data in sorted(patients_data.items()):
            if "path_ct" not in data:
                logger.warning("CT mancante per il paziente '%s'", name)
                continue

            if "path_rt" not in data:
                logger.warning("RTStruct mancante per il paziente '%s'", name)
                continue

            patients.append(Paziente(nome=name, **data))

        return patients


class ClinicalDataLoader(BaseModel):
    """Legge il target clinico dal CSV e lo associa ai pazienti DICOM."""

    csv_path: FilePath

    @staticmethod
    def _normalise_name(value: str) -> str:
        """Uniforma solo separatori, spazi e maiuscole per il confronto."""
        value = re.sub(r"[\^_]+", " ", value.strip())
        return " ".join(value.split()).casefold()

    @staticmethod
    def _normalise_id(value: str) -> str:
        # L'ID resta una stringa: gli eventuali zeri iniziali sono significativi.
        return value.strip()

    def _read_records(self) -> dict[str, tuple[str, StatoMicrosatellitare, int]]:
        records: dict[str, tuple[str, StatoMicrosatellitare, int]] = {}

        with open(self.csv_path, "r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file, delimiter=";", skipinitialspace=True)
            if reader.fieldnames is None:
                raise ValueError(f"Il file clinico '{self.csv_path}' non ha un'intestazione")

            reader.fieldnames = [field.strip() for field in reader.fieldnames]
            required_fields = {"DicomID", "NomeCognome", "Tipo"}
            missing_fields = required_fields.difference(reader.fieldnames)
            if missing_fields:
                missing = ", ".join(sorted(missing_fields))
                raise ValueError(
                    f"Colonne mancanti in '{self.csv_path}': {missing}"
                )

            for row_number, row in enumerate(reader, start=2):
                dicom_id = self._normalise_id(row.get("DicomID") or "")
                patient_name = (row.get("NomeCognome") or "").strip()
                raw_status = (row.get("Tipo") or "").strip().upper()

                if not dicom_id and not patient_name and not raw_status:
                    continue
                if not dicom_id or not patient_name or not raw_status:
                    raise ValueError(
                        f"Riga {row_number} incompleta in '{self.csv_path}'"
                    )
                if dicom_id in records:
                    raise ValueError(
                        f"DicomID duplicato '{dicom_id}' alla riga {row_number} "
                        f"di '{self.csv_path}'"
                    )

                try:
                    status = StatoMicrosatellitare(raw_status)
                except ValueError as error:
                    valid_values = ", ".join(item.value for item in StatoMicrosatellitare)
                    raise ValueError(
                        f"Stato microsatellitare non valido '{raw_status}' alla "
                        f"riga {row_number}; valori ammessi: {valid_values}"
                    ) from error

                records[dicom_id] = (patient_name, status, row_number)

        return records

    def assign_status(self, patients: list[Paziente]) -> list[Paziente]:
        """Restituisce copie dei pazienti arricchite con il relativo target clinico."""
        records = self._read_records()
        enriched_patients: list[Paziente] = []
        seen_dicom_ids: set[str] = set()

        for patient in patients:
            ct_id = self._normalise_id(patient.get_patient_id())
            rt_id = self._normalise_id(patient.get_patient_id(use_rt=True))
            if ct_id == "Non Disponibile" or rt_id == "Non Disponibile":
                raise ValueError(
                    f"PatientID DICOM mancante per il paziente '{patient.nome}'"
                )
            if ct_id != rt_id:
                raise ValueError(
                    f"PatientID incoerente per '{patient.nome}': CT='{ct_id}', RT='{rt_id}'"
                )
            if ct_id in seen_dicom_ids:
                raise ValueError(f"PatientID DICOM duplicato tra i pazienti: '{ct_id}'")
            seen_dicom_ids.add(ct_id)

            record = records.get(ct_id)
            if record is None:
                logger.warning(
                    "Nessun dato clinico nel CSV per '%s' (PatientID=%s)",
                    patient.nome,
                    ct_id,
                )
                enriched_patients.append(patient)
                continue

            csv_name, status, csv_row_number = record
            ct_name = patient.get_patient_name()
            rt_name = patient.get_patient_name(use_rt=True)
            expected_name = self._normalise_name(csv_name)
            names_to_check = {
                "cartella": patient.nome,
                "DICOM CT": ct_name,
                "DICOM RT": rt_name,
            }
            inconsistent_names = {
                source: name
                for source, name in names_to_check.items()
                if self._normalise_name(name) != expected_name
            }
            if inconsistent_names:
                found = ", ".join(
                    f"{source}='{name}'" for source, name in inconsistent_names.items()
                )
                raise ValueError(
                    f"Nome incoerente per PatientID '{ct_id}' alla riga "
                    f"{csv_row_number} del CSV: "
                    f"CSV='{csv_name}', {found}"
                )

            enriched_patients.append(
                patient.model_copy(update={"stato_microsatellitare": status})
            )

        unused_ids = set(records).difference(seen_dicom_ids)
        if unused_ids:
            logger.warning(
                "%d record del CSV non corrispondono a un paziente DICOM caricato",
                len(unused_ids),
            )

        logger.info(
            "Stato microsatellitare associato a %d pazienti su %d",
            sum(patient.stato_microsatellitare is not None for patient in enriched_patients),
            len(enriched_patients),
        )
        return enriched_patients


class MirpExtractor(BaseModel):
    nome: str = "standardExtractor"
    config: MirpConfig = Field(default_factory=MirpConfig)

    def extract(self, 
        paziente: Paziente, 
        output_dir: Path | None = None,
        verbose: bool = False

    ) -> list[Any] | None:
        """Estrae le feature radiomiche per un singolo paziente."""
        import mirp as _mirp

        kwargs = {
            "image": str(paziente.path_ct),
            "mask": str(paziente.path_rt),
            **self.config.to_mirp_kwargs(),
        }

        # Stampa un recap dei parametri:
        # Sono sicuro che arrivano correttamente qui dalla configurazione .yaml

        write_dir = str(output_dir) if output_dir is not None else None

        if verbose: 
            self._print_summary()
            logger.info(
                "Estrazione feature MIRP per paziente '%s'", 
                paziente.nome
            )


        return _mirp.extract_features(
            write_features=self.config.write_features,
            export_features=self.config.export_features,
            write_dir=write_dir,

            **kwargs, # Sono qui i dati del paziente
        )

    # Versione parallela disattivata. Viene conservata come riferimento.
    #
    # def extract_batch_parallel(
    #     self,
    #     pazienti: list[Paziente],
    #     output_dir: Path | None = None,
    #     verbose: bool = False,
    # ) -> dict[str, list[Any] | None]:
    #     """Estrae più pazienti in processi separati."""
    #     from joblib import Parallel, delayed
    #
    #     num_processes = self.config.num_processes or 1
    #     completed_jobs = Parallel(
    #         n_jobs=num_processes,
    #         backend="loky",
    #         return_as="generator_unordered",
    #         batch_size=1,
    #     )(
    #         delayed(self._extract_patient)(paziente, output_dir)
    #         for paziente in pazienti
    #     )
    #     return dict(completed_jobs)

    def extract_batch(
        self,
        pazienti: list[Paziente],
        output_dir: Path | None = None,
        verbose: bool = False,
    ) -> dict[str, list[Any] | None]:
        """Estrae più pazienti in sequenza mostrando l'avanzamento."""
        if not pazienti:
            return {}

        patient_names = [paziente.nome for paziente in pazienti]
        duplicate_names = sorted(
            name for name in set(patient_names) if patient_names.count(name) > 1
        )
        if duplicate_names:
            raise ValueError(
                "Nomi paziente duplicati nel batch: " + ", ".join(duplicate_names)
            )

        if verbose:
            self._print_summary()

        logger.info(
            "Avvio estrazione sequenziale di %d pazienti",
            len(pazienti),
        )

        # Evita che i messaggi delle librerie di terze parti interferiscano
        # con la barra di avanzamento.
        # logging.basicConfig(level=logging.CRITICAL, force=True)

        completed_results: dict[str, list[Any] | None] = {}
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("paziente: [cyan]{task.fields[current_patient]}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=Console(),
        )

        with progress:
            task_id = progress.add_task(
                "Estrazione radiomica",
                total=len(pazienti),
                current_patient="-",
            )

            for paziente in pazienti:
                progress.update(
                    task_id,
                    current_patient=paziente.nome,
                    refresh=True,
                )
                patient_name, features = self._extract_patient(
                    paziente,
                    output_dir,
                )
                completed_results[patient_name] = features
                progress.update(task_id, advance=1)

        logger.info(
            "Estrazione batch completata per %d pazienti", len(completed_results)
        )
        return {
            paziente.nome: completed_results[paziente.nome]
            for paziente in pazienti
        }

    def write_feature_csv(
        self,
        risultati: dict[str, list[Any] | None],
        pazienti: list[Paziente],
        csv_path: Path,
        metadata_config: DicomMetadataConfig | None = None,
    ) -> int:
        """Scrive metadati paziente, DICOM e feature radiomiche in un CSV."""
        import pandas as pd

        frames: list[pd.DataFrame] = []
        metadata_columns = self._metadata_columns(metadata_config)
        metadata_by_patient = self._extract_dicom_metadata_by_patient(
            pazienti=pazienti,
            metadata_config=metadata_config,
        )

        for paziente in pazienti:
            feature_tables = risultati.get(paziente.nome)
            for feature_table in self._normalise_feature_tables(feature_tables):
                table = self._keep_radiomic_feature_columns(feature_table)
                metadata_values = metadata_by_patient.get(paziente.nome, {})
                table.insert(
                    0,
                    "stato_microsatellitare",
                    (
                        paziente.stato_microsatellitare.value
                        if paziente.stato_microsatellitare is not None
                        else ""
                    ),
                )
                table.insert(
                    0,
                    "nome_cognome",
                    self._format_nome_cognome(paziente.nome),
                )
                for index, column in enumerate(metadata_columns, start=2):
                    if column in table.columns:
                        raise ValueError(
                            f"La colonna metadata DICOM '{column}' esiste gia' "
                            "nella tabella delle feature."
                        )
                    table.insert(
                        index,
                        column,
                        metadata_values.get(column, DEFAULT_MISSING_VALUE),
                    )
                frames.append(table)

        if frames:
            features = pd.concat(frames, ignore_index=True, sort=False)
        else:
            logger.warning("Nessuna tabella di feature restituita da MIRP")
            features = pd.DataFrame(
                columns=[
                    "nome_cognome",
                    "stato_microsatellitare",
                    *metadata_columns,
                ]
            )

        features.to_csv(csv_path, sep=";", na_rep="", index=False)
        logger.info("Feature scritte in %s", csv_path)
        return len(features)

    @staticmethod
    def _metadata_columns(
        metadata_config: DicomMetadataConfig | None,
    ) -> list[str]:
        if metadata_config is None:
            return []
        return [tag.output_column for tag in metadata_config.active_tags()]

    def _extract_dicom_metadata_by_patient(
        self,
        pazienti: list[Paziente],
        metadata_config: DicomMetadataConfig | None,
    ) -> dict[str, dict[str, Any]]:
        if metadata_config is None or not metadata_config.has_tags():
            return {}

        return {
            paziente.nome: self._extract_dicom_metadata_for_patient(
                paziente=paziente,
                metadata_config=metadata_config,
            )
            for paziente in pazienti
        }

    @staticmethod
    def _extract_dicom_metadata_for_patient(
        paziente: Paziente,
        metadata_config: DicomMetadataConfig,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        for source in ("ct", "rt"):
            tag_configs = [
                tag
                for tag in metadata_config.active_tags()
                if tag.source == source
            ]
            if not tag_configs:
                continue

            record = paziente.get_dicom_metadata_record(
                tags=[tag.to_dicom_tag_spec() for tag in tag_configs],
                use_rt=source == "rt",
            )
            metadata.update(record.as_dict())

        return metadata

    @staticmethod
    def _normalise_feature_tables(feature_tables: Any) -> list[Any]:
        import pandas as pd

        if feature_tables is None:
            return []

        if isinstance(feature_tables, pd.DataFrame):
            return [feature_tables]

        if not isinstance(feature_tables, list):
            raise TypeError(
                "MIRP ha restituito un tipo inatteso per le feature: "
                f"{type(feature_tables).__name__}"
            )

        normalised_tables: list[pd.DataFrame] = []
        for item in feature_tables:
            if item is None:
                continue
            if isinstance(item, pd.DataFrame):
                normalised_tables.append(item)
                continue
            raise TypeError(
                "MIRP ha restituito una tabella feature inattesa: "
                f"{type(item).__name__}"
            )

        return normalised_tables

    @staticmethod
    def _keep_radiomic_feature_columns(feature_table: Any) -> Any:
        mirp_metadata_columns = {
            "sample_name",
            "image_file_name",
            "image_directory",
            "image_study_date",
            "image_study_description",
            "image_series_description",
            "image_series_instance_uid",
            "image_modality",
            "image_pet_suv_type",
            "image_mask_label",
            "image_mask_file_name",
            "image_mask_directory",
            "image_mask_series_description",
            "image_mask_series_instance_uid",
            "image_settings_id",
            "image_voxel_size_x",
            "image_voxel_size_y",
            "image_voxel_size_z",
            "image_noise_level",
            "image_noise_iteration_id",
            "image_rotation_angle",
            "image_translation_x",
            "image_translation_y",
            "image_translation_z",
            "image_mask_randomise_id",
            "image_mask_adapt_size",
        }
        feature_columns = [
            column
            for column in feature_table.columns
            if column not in mirp_metadata_columns
        ]
        return feature_table.loc[:, feature_columns].copy()

    @staticmethod
    def _format_nome_cognome(value: str) -> str:
        return " ".join(re.sub(r"[\^_]+", " ", value).split())

    def _extract_patient(
        self,
        paziente: Paziente,
        output_dir: Path | None,
    ) -> tuple[str, list[Any] | None]:
        """Estrae le feature di un paziente aggiungendo il suo nome al risultato."""
        try:
            features = self.extract(paziente, output_dir, verbose=False)
        except Exception as error:
            raise RuntimeError(
                f"Estrazione radiomica fallita per '{paziente.nome}'"
            ) from error
        return paziente.nome, features


    def _print_summary(self):
        """Stampa una tabella riassuntiva dei parametri ricevuti utilizzando rich."""
        console = Console()
        
        print()
        # Creazione della tabella stilizzata
        table = Table(
            title="[bold]MIRP EXTRACTOR - CONFIGURAZIONE ATTIVA[cyan]",
            show_header=True, 
            header_style="bold"
        )
        
        # Aggiunta delle colonne
        table.add_column("Parametro", style="cyan", width=25)
        table.add_column("Valore", style="green")

        # .model_dump() trasforma lo schema in un dict
        for key, value in self.config.model_dump().items():
            table.add_row(key, str(value))
            
        console.print(table)
        print() 



def load_configuration(config_path: Path | str = CONFIG_FILE) -> GlobalConfig:
    return ConfigManager(config_path=Path(config_path)).load_configuration()


def carica_pazienti(config: GlobalConfig) -> list[Paziente]:
    patients = PatientScanner(
        patients_folder=config.data.patients_folder
    ).load_patients()
    if config.data.patients_csv is None:
        return patients

    return ClinicalDataLoader(
        csv_path=config.data.patients_csv
    ).assign_status(patients)
