# CLI e utility DICOM

## Guida generale

```powershell
.\.venv\Scripts\python.exe .\cli.py --help
```

La CLI comprende estrazione, analisi, ispezione DICOM e riepilogo pazienti.

Per creare esplicitamente il dataset clinico preprocessato:

```powershell
.\.venv\Scripts\python.exe .\cli.py clinical --config config/config_clinical.yaml
```

La sua inclusione nelle analisi e' controllata separatamente da `dataset_mode`
in `config/config_analysis.yaml`.

## Ispezionare un DICOM

Per mostrare tutti i metadati:

```powershell
.\.venv\Scripts\python.exe .\cli.py --infodicom C:\dati\file.dcm
```

Per mostrare tag specifici, ripetere `--tag`:

```powershell
.\.venv\Scripts\python.exe .\cli.py `
  --infodicom C:\dati\file.dcm `
  --tag PatientID `
  --tag "(0008,0020)"
```

Sono accettate keyword DICOM e coppie numeriche `(gruppo,elemento)`.

## Elencare e confrontare i pazienti

```powershell
.\.venv\Scripts\python.exe .\cli.py --pazienti
```

Con percorsi espliciti:

```powershell
.\.venv\Scripts\python.exe .\cli.py `
  --pazienti `
  --cartella-dati C:\percorso\radiomic-data `
  --csv-pazienti C:\percorso\radiomic-data\pazienti.csv
```

Ordinamento:

```powershell
# Per nome
.\.venv\Scripts\python.exe .\cli.py --pazienti --ordina nome

# Per data, dalla più recente
.\.venv\Scripts\python.exe .\cli.py --pazienti --ordina data --discendente
```

Il riepilogo evidenzia pazienti presenti nei DICOM, record presenti nel CSV e
differenze tra le due sorgenti.

## Uso del servizio da Python

```python
from extractor import DicomMetadataService

service = DicomMetadataService()
record = service.extract_from_file(
    "C:/dati/file.dcm",
    ["PatientID", "PatientName", "(0008,0020)"],
)

print(record.as_dict())
```

Il servizio supporta file singoli e cartelle, riconosce file DICOM anche senza
estensione e restituisce errori espliciti per file vuoti o non validi.
