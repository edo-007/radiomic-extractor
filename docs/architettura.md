# Architettura del codice

## Moduli principali

```text
cli.py
├── extractor/
│   ├── main.py
│   ├── models.py
│   ├── dicom_metadata_service.py
│   ├── feature_preview_service.py
│   ├── logger_conf.py
│   └── utils.py
└── analysis/
    ├── main.py
    ├── config.py
    ├── selector.py
    ├── data.py
    ├── preprocessing.py
    ├── models.py
    ├── univariate.py
    ├── benchmark.py
    ├── importance.py
    ├── tuning_config.py
    ├── tuning.py
    ├── artifacts.py
    └── reporting.py
```

## Pacchetto extractor

- `main.py`: orchestration del run, conferma, cartella esperimento e ciclo sugli
  spacing.
- `models.py`: schemi Pydantic, scansione pazienti, dati clinici, wrapper MIRP e
  scrittura incrementale dei CSV.
- `dicom_metadata_service.py`: lettura robusta dei tag CT/RTStruct.
- `feature_preview_service.py`: costruzione preventiva dei nomi delle feature.
- `logger_conf.py`: configurazione dei log.
- `utils.py`: utility di ordinamento.

`MirpConfig.to_mirp_kwargs()` è il confine tra lo YAML del progetto e gli
argomenti passati alla libreria MIRP. Per un batch multi-spacing viene creata
una copia della configurazione con un solo spacing alla volta.

## Pacchetto analysis

- `config.py`: schema di `config/config_analysis.yaml`.
- `clinical_config.py`: schema di `config/config_clinical.yaml`.
- `clinical.py`: pulizia e creazione esplicita del singolo CSV clinico.
- `selector.py`: scoperta e menu esperimenti/CSV.
- `data.py`: validazione, join clinico opzionale e separazione `X`, `y`, gruppi.
- `preprocessing.py`: trasformatori e pipeline anti-leakage.
- `models.py`: factory degli estimatori.
- `univariate.py`: test statistici e correzione FDR.
- `benchmark.py`: split, metriche e benchmark.
- `importance.py`: permutation importance sui validation fold.
- `tuning_config.py`: schema di `config/config_tuning.yaml`.
- `tuning.py`: ricerca inner, valutazione outer e refit finale.
- `artifacts.py`: tracciamento e serializzazione dei risultati.
- `reporting.py`: tabelle Rich mostrate nel terminale.

## Flusso dell'analisi

```text
config/config_analysis.yaml ──────────────┐
                                   ▼
radiomic-output ──► selector ──► data loader
                                   │
                                   ▼
                           AnalysisDataset
                           ├── X numerica
                           ├── y binaria
                           └── gruppi paziente
                                   │
                 ┌─────────────────┼─────────────────┐
                 ▼                 ▼                 ▼
             univariate        benchmark          tuning
                                   │                 │
                                   ▼                 ▼
                              reporting       artifacts + reporting
```

## Validazione e sicurezza metodologica

- Gli schemi usano `extra="forbid"`: chiavi YAML sconosciute sono errori.
- Tutti gli split sono raggruppati per paziente.
- Preprocessing e selezione vengono adattati dentro i fold.
- La nested CV separa scelta e valutazione degli iperparametri.
- I run conservano configurazioni e hash del dataset.

## Test

Esecuzione completa:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

I test coprono pipeline di analisi, selector, spacing, cartelle esperimento,
nested tuning e salvataggio degli artefatti. Per intercettare deprecazioni:

```powershell
.\.venv\Scripts\python.exe -W error::FutureWarning -m unittest discover -s tests -v
```
