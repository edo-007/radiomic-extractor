# radiomic-extractor
orchestration and extraction of radiomic features via mirp

La [documentazione completa](docs/README.md) è disponibile nella cartella `docs/`.

## Struttura

- `extractor/`: moduli Python per scansione pazienti ed estrazione feature.
- `analysis/`: analisi univariata, benchmark ML e permutation importance.
- `config/config_extractor.yaml`: configurazione principale; dettagli nei quattro profili
  in `config/extraction/`. [Guida ai profili](docs/configurazione-estrazione.md).
- `config/config_analysis.yaml`: dati, preprocessing, modelli e cross-validation.
- `config/config_clinical.yaml`: pulizia e codifica del singolo dataset clinico.
- `config/config_tuning.yaml`: nested CV e spazi di ricerca degli iperparametri.

Avvio:

```bash
python -m extractor.main
# equivalente tramite CLI unificata
python cli.py extract --config config/config_extractor.yaml
```

Prima dell'estrazione il comando stampa un recap con numero e nomi delle feature
radiomiche previste e chiede conferma con
`Continuare con l'estrazione radiomica? [s/N]`. Dopo la conferma crea dentro
`data.output_dir` una cartella `extraction-YYYY-MM-DD_HH-MM-SS` contenente:

- `config_resolved.yaml`, configurazione completa risolta con default MIRP e provenienza;
- `config_sources/`, copie dei file YAML originali, commenti inclusi;
- `feature_preview.csv`, lista completa delle colonne previste;
- un CSV delle feature per ogni voxel spacing richiesto.

I voxel spacing sono isotropici: specificare un numero per ciascuno spacing:

```yaml
mirp:
  voxel_spacing: [1.25, 1.0, 2.0]
```

Per un solo spacing usare `voxel_spacing: [1.25]` oppure `voxel_spacing: 1.25`.
Triplette dimensionali e valori duplicati non sono accettati. I CSV hanno nomi come
`features_voxel-spacing-1p25mm.csv` e `features_voxel-spacing-2mm.csv`.

Se `patients_csv` e' configurato, i pazienti non presenti nel CSV clinico
vengono ignorati.

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

## Analisi delle feature

Il dataset clinico viene generato soltanto quando richiesto:

```bash
python cli.py clinical --config config/config_clinical.yaml
```

In `config/config_analysis.yaml`, `dataset_mode` sceglie `radiomic`, `clinical` oppure
`combined`. Nelle ultime due modalita' il CSV clinico viene unito per paziente
al CSV radiomico scelto nel menu; non viene scritto un dataset combinato.

I comandi di analisi mostrano due menu interattivi navigabili con le frecce
↑/↓ e Invio: prima si sceglie l'esperimento, quindi uno dei suoi CSV di
feature. Gli esperimenti sono ordinati dal più recente al più vecchio. La
cartella che li contiene è configurata tramite `analysis.experiments_dir` in
`config/config_analysis.yaml`; il singolo `input_csv` non è salvato nella
configurazione. I risultati vengono stampati nel terminale e non creano file
di output.

```bash
python cli.py analysis univariate --config config/config_analysis.yaml
python cli.py analysis benchmark --config config/config_analysis.yaml
python cli.py analysis importance --config config/config_analysis.yaml
python cli.py analysis all --config config/config_analysis.yaml
python cli.py analysis tune --config config/config_analysis.yaml --tuning-config config/config_tuning.yaml
```

I parametri contenuti in `models.<nome>.params` vengono passati al relativo
estimatore scikit-learn.

Imputazione, filtro delle correlazioni, scaling e selezione delle feature sono
inseriti nella pipeline e appresi esclusivamente sul training fold. La
cross-validation usa inoltre l'ID paziente come gruppo, evitando che righe
dello stesso paziente siano divise tra training e validation.

Per usare la Leave-One-Out cross-validation per paziente:

```yaml
cross_validation:
  method: leave_one_out  # e' accettato anche: loo
  n_splits: 5            # ignorato in modalita' LOO
  n_repeats: 5           # ignorato in modalita' LOO
  random_state: 42
```

Con LOO le metriche vengono calcolate sulle predizioni out-of-fold aggregate.
La permutation importance richiede validation fold con almeno due osservazioni
e viene quindi saltata dal comando `analysis all` quando LOO e' attiva.

### Tuning dei modelli

Per ispezionare una Random Forest salvata senza riaddestrarla:

```powershell
.\.venv\Scripts\python.exe cli.py analysis inspect-rf
```

Seleziona un tuning con le frecce e apri il report HTML generato: contiene
diagrammi degli alberi, profondita', nodi, foglie e importanze delle feature.
Dettagli: [Ispezione Random Forest](docs/ispezione-random-forest.md).
Nei nuovi addestramenti RF lo scaling e' disabilitato; i modelli storici
rimangono invariati e le loro soglie vengono riconvertite solo nel report.

Il comando `analysis tune` usa la configurazione separata
`config/config_tuning.yaml`. Per impostazione predefinita confronta regressione
logistica L2, Random Forest e SVM RBF con nested cross-validation:

- outer CV: 4 fold ripetuti 5 volte, usati esclusivamente per la valutazione;
- inner CV: 3 fold, usati per scegliere gli iperparametri;
- selezione del numero di feature eseguita dentro ogni fold;
- ricerca configurabile: `grid` oppure `randomized` limitata da `n_iter`.

Ogni esecuzione crea `../radiomic-output/tuning/tuning-YYYY-MM-DD_HH-MM-SS` con le copie
delle configurazioni congelate all'avvio (originali e risolte), l'intera
configurazione dell'estrazione selezionata, l'origine e l'hash del CSV, le metriche outer,
tutti i risultati della ricerca, le feature finali e i modelli serializzati.
`report.txt` e `report.html` conservano le stesse tabelle del terminale.
`scoring: [f1, accuracy]` calcola entrambe le metriche inner della configurazione
scelta con `refit: f1`. La metrica
da riportare è quella della outer CV; il punteggio inner serve solo a scegliere
gli iperparametri.

Filtri MIRP IBSI-compliant:

```yaml
mirp:
  # Il codice forza sempre ibsi_compliant=true.
  filter_kernels:
    - laplacian_of_gaussian
    - gabor
  response_map_feature_families:
    - statistics
    # - intensity_histogram
    # - glcm
  response_map_discretisation_n_bins: 16
  laplacian_of_gaussian_sigma: [5.0, 10.0]
  gabor_sigma: [2.0, 4.0]
  gabor_lambda: [1.0, 2.0]
```

Sono esposti solo i filtri IBSI-compliant: `mean`,
`laplacian_of_gaussian`/`log`, `laws`, `gabor`, `separable_wavelet` e
`nonseparable_wavelet`. Gaussian, Laplace semplice (`laplace`/`laplacian`),
Sobel, Prewitt, LBP e le varianti Riesz non sono configurabili perche'
richiedono `ibsi_compliant=false` o non hanno reference values IBSI.
