# Output e riproducibilità

## Cartella di estrazione

Ogni estrazione confermata crea una directory indipendente:

```text
radiomic-output/
└── extraction-YYYY-MM-DD_HH-MM-SS/
    ├── config_resolved.yaml
    ├── config_sources/
    ├── feature_preview.csv
    ├── features_voxel-spacing-1p25mm.csv
    ├── features_voxel-spacing-1mm.csv
    └── features_voxel-spacing-2mm.csv
```

- `config_resolved.yaml` contiene configurazione unita, default e impostazioni
  effettive MIRP per spacing; e' ricaricabile senza i profili originali.
- `config_sources/` conserva i file originali (commenti inclusi); hash e versioni
  software sono registrati in `provenance` nello snapshot risolto.
- `feature_preview.csv` elenca le colonne previste prima dell'estrazione.
- Ogni `features_voxel-spacing-*.csv` contiene solo le feature ottenute con lo
  spacing dichiarato nel nome.

Se due run iniziano nello stesso secondo viene aggiunto un suffisso `_2`, `_3`
e così via. I nomi decimali usano `p` per non confondere il punto con
l'estensione: `1.25 mm` diventa `1p25mm`.

## Contenuto dei CSV

Le prime colonne identificano paziente e classe:

```text
nome_cognome
stato_microsatellitare
metadata_...
feature radiomiche...
```

I metadati tecnici interni restituiti da MIRP che non rappresentano feature
radiomiche vengono rimossi dal CSV finale. Le liste DICOM, come PixelSpacing,
sono serializzate in una singola cella usando `\` come separatore.

## Riproducibilità del tuning

Ogni tuning crea:

```text
radiomic-output/tuning/
└── tuning-YYYY-MM-DD_HH-MM-SS/
    ├── config_analysis.yaml
    ├── config_analysis_resolved.yaml
    ├── config_tuning.yaml
    ├── config_tuning_resolved.yaml
    ├── config_extraction.yaml
    ├── extraction_config/
    │   ├── config_resolved.yaml
    │   └── config_sources/
    ├── report.txt
    ├── report.html
    ├── source.json
    ├── summary.csv
    ├── outer_fold_metrics.csv
    ├── cv_results.csv
    ├── best_parameters.json
    ├── selected_features.csv
    └── models/
        ├── logistic_l2.joblib
        ├── random_forest.joblib
        └── svm_rbf.joblib
```

`source.json` registra il percorso del CSV, la directory dell'esperimento,
l'orario e lo SHA-256 del file. L'hash consente di controllare che il dataset
non sia cambiato dopo il tuning.

Le configurazioni vengono congelate prima dei fit: gli YAML originali sono
gli stessi byte letti all'avvio, mentre quelli `_resolved` includono i default
e i percorsi assoluti. `source.json` include anche gli hash degli snapshot.
`extraction_config/` conserva gli YAML dell'estrazione selezionata e i suoi
`config_sources/`, senza modificarli; `config_extraction.yaml` e' una copia
identica del suo `config_resolved.yaml`. Se il file risolto manca, viene
mostrato un avviso e non viene inventata una configurazione sostitutiva.
Quando sono usati dati clinici, viene conservato anche il relativo sidecar
`config_clinical.yaml`, se disponibile.

`report.txt` e `report.html` contengono le tabelle effettivamente stampate, con
la stessa larghezza del terminale configurata. L'HTML conserva anche i colori.

I file `.joblib` contengono l'intera pipeline finale già addestrata:
imputazione, filtri, scaling, selezione delle feature e modello. Devono essere
caricati solo se provengono da una fonte fidata e in un ambiente con versioni
software compatibili.

## Cosa riportare

Per una valutazione standard riportare media e deviazione standard dei fold.
Per il tuning riportare la metrica della outer CV contenuta in `summary.csv`.
Il punteggio inner presente in `best_parameters.json` serve a scegliere gli
iperparametri e non è una stima imparziale delle prestazioni future.
