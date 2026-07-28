# radiomic-extractor
orchestration and extraction of radiomic features via mirp

## Struttura

- `extractor/`: moduli Python per scansione pazienti ed estrazione feature.
- `config_extractor.yaml`: configurazione dell'estrattore, mantenuta nella root del progetto.

Avvio:

```bash
python -m extractor.main
```

Info DICOM:

```bash
python cli.py --infodicom path/to/file.dcm
python cli.py --infodicom path/to/file.dcm --tag PatientID --tag "(0008,0020)"
```

Uso da codice:

```python
from extractor import DicomMetadataService

service = DicomMetadataService()
record = service.extract_from_file(
    "path/to/file.dcm",
    ["PatientID", "PatientName", "(0008,0020)"],
)
metadata = record.as_dict()
```

Metadati DICOM nel CSV delle feature:

```yaml
dicom_metadata:
  enabled: true
  tags:
    # Data dello studio CT nel formato DICOM YYYYMMDD.
    - StudyDate
```

La colonna generata nel CSV sara' `metadata_study_date`.
