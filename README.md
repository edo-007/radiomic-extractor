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
```
