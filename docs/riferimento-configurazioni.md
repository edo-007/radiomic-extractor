# Riferimento delle configurazioni YAML

## Regole comuni

- I file usano YAML e indentazione a spazi.
- I decimali usano il punto: `1.25`.
- `null` disabilita o lascia non specificato un valore opzionale.
- `.nan` rappresenta un limite numerico aperto dove previsto.
- I percorsi relativi sono risolti rispetto alla directory del relativo YAML.
- Le chiavi sconosciute nella configurazione di analisi e tuning generano un
  errore, per intercettare refusi.

## `config/config_extractor.yaml`

Il file principale include quattro profili tematici in `config/extraction/`.
Vedere [struttura, regole e snapshot](configurazione-estrazione.md).
Le chiavi elencate nelle tabelle restano nella sezione `mirp`, ma si modificano
nel relativo profilo, senza duplicarle nel file principale. Anche nell'estrazione
chiavi sconosciute e duplicate vengono rifiutate.

### Sezione `data`

| Chiave | Tipo | Significato |
|---|---|---|
| `patients_folder` | percorso | Directory radice degli studi DICOM |
| `patients_csv` | percorso o `null` | Anagrafica clinica e fonte ufficiale di inclusione |
| `output_dir` | percorso | Directory che conterrà gli esperimenti |
| `features_csv` | percorso, opzionale | Nome base dei CSV; default `features.csv` |

### Sezione `dicom_metadata`

| Chiave | Tipo | Significato |
|---|---|---|
| `enabled` | booleano | Abilita l'aggiunta di metadati al CSV |
| `tags` | lista | Tag da estrarre |
| `tags[].tag` | stringa | Keyword o coppia numerica DICOM |
| `tags[].source` | `ct` o `rt` | Dataset da cui leggere il tag |
| `tags[].missing_value` | valore opzionale | Valore usato quando il tag non è disponibile |

Una voce stringa, per esempio `- StudyDate`, equivale a una voce con
`tag: StudyDate` e sorgente CT.

### Sezione `mirp`: base

| Chiave | Tipo | Significato |
|---|---|---|
| `bin_width` | float positivo | Bin width delle immagini originali |
| `voxel_spacing` | numero/lista | Uno o più spacing isotropici |
| `by_slice` | booleano | `false` per 3D, `true` per 2D slice-by-slice |
| `roi_names` | lista o `null` | Nomi delle ROI da includere |
| `resegmentation_intensity_range` | `[min,max]` o `null` | Filtro delle intensità HU |
| `feature_families` | lista | Famiglie sull'immagine originale |
| `filter_kernels` | lista o `null` | Filtri delle response map |
| `response_map_feature_families` | lista | Famiglie sulle immagini filtrate |
| `response_map_discretisation_n_bins` | intero/lista | Numero di bin delle response map |
| `boundary_condition` | stringa | Gestione dei bordi dei filtri |

Condizioni del bordo supportate: `reflect`, `constant`, `nearest`, `mirror` e
`wrap`.

### Mean e Laplacian-of-Gaussian

| Chiave | Significato |
|---|---|
| `mean_filter_kernel_size` | Dimensione del kernel mean |
| `laplacian_of_gaussian_sigma` | Uno o più sigma LoG |
| `laplacian_of_gaussian_kernel_truncate` | Troncamento del kernel |
| `laplacian_of_gaussian_pooling_method` | Pooling tra risposte |

Il pooling LoG accetta `none`, `max`, `min`, `mean` o `sum`.

### Gabor

| Chiave | Significato |
|---|---|
| `gabor_sigma` | Scala/e del filtro |
| `gabor_lambda` | Lunghezza/e d'onda |
| `gabor_gamma` | Rapporto d'aspetto |
| `gabor_theta` | Angolo/i espliciti |
| `gabor_theta_step` | Passo angolare alternativo |
| `gabor_response` | Componente della risposta |
| `gabor_rotation_invariance` | Abilita invarianza alla rotazione |
| `gabor_pooling_method` | Pooling delle orientazioni |

Le risposte comprendono `modulus`, `magnitude`, `angle`, `phase`, `real` e
`imaginary`, inclusi gli alias validati dal codice.

### Laws

| Chiave | Significato |
|---|---|
| `laws_kernel` | Uno o più kernel Laws |
| `laws_delta` | Distanza usata nel calcolo dell'energia |
| `laws_compute_energy` | Calcola la mappa di energia |
| `laws_rotation_invariance` | Abilita invarianza alla rotazione |
| `laws_pooling_method` | Pooling delle risposte |

### Wavelet separabili

| Chiave | Significato |
|---|---|
| `separable_wavelet_families` | Famiglie wavelet, per esempio `coif4` |
| `separable_wavelet_set` | Combinazioni, per esempio `hhh`, `lll` |
| `separable_wavelet_stationary` | Trasformata stazionaria |
| `separable_wavelet_decomposition_level` | Livello/i di decomposizione |
| `separable_wavelet_rotation_invariance` | Invarianza alla rotazione |
| `separable_wavelet_pooling_method` | Pooling delle risposte |

### Wavelet non separabili

| Chiave | Significato |
|---|---|
| `nonseparable_wavelet_families` | Famiglie, per esempio `simoncelli` |
| `nonseparable_wavelet_decomposition_level` | Livello/i di decomposizione |
| `nonseparable_wavelet_response` | Componente della risposta |

### Esecuzione

| Chiave | Tipo | Significato |
|---|---|---|
| `num_processes` | `-1`, `null` o intero positivo | Processi richiesti; il batch attuale è sequenziale |
| `write_features` | booleano | Scrive anche gli output nativi MIRP |
| `export_features` | booleano | Restituisce le tabelle per il CSV aggregato |
| `n_test` | `false` o intero positivo | Limita l'estrazione ai primi N pazienti |

`n_test` è una chiave globale, allo stesso livello di `data`, `dicom_metadata`
e `mirp`; l'alias storico `n-test` non e' supportato.

## `config/config_analysis.yaml`

### Dataset e colonne

| Chiave | Significato |
|---|---|
| `experiments_dir` | Directory contenente gli esperimenti selezionabili |
| `dataset_mode` | `radiomic`, `clinical` oppure `combined` |
| `clinical_data.input_csv` | CSV creato esplicitamente con `cli.py clinical` |
| `clinical_data.separator` | Separatore del CSV clinico |
| `clinical_data.id_column` | Chiave paziente nel CSV clinico |
| `clinical_data.restrict_radiomic_to_matched_patients` | Limita il baseline radiomico agli ID presenti anche nel CSV clinico |
| `separator` | Separatore CSV, esattamente un carattere |
| `id_column` | Identificatore del paziente |
| `target_column` | Classe da predire |
| `positive_class` | Valore considerato classe positiva |
| `exclude_columns` | Colonne escluse dai predittori |
| `metadata_prefix` | Prefisso delle colonne metadata |
| `include_metadata` | Include i metadata tra i predittori |

### Preprocessing

| Chiave | Valori |
|---|---|
| `impute_strategy` | `median`, `mean`, `most_frequent` |
| `remove_constant_features` | booleano |
| `correlation_threshold` | float tra 0 e 1 oppure `null` |
| `scaling` | `standard`, `robust`, `none` |
| `feature_selection.enabled` | booleano |
| `feature_selection.method` | `f_classif`, `mutual_info` |
| `feature_selection.k` | intero positivo |

### Univariata e cross-validation

| Chiave | Valori |
|---|---|
| `univariate.enabled` | booleano |
| `univariate.test` | `mannwhitney`, `welch_ttest` |
| `univariate.correction` | `fdr_bh`, `none` |
| `cross_validation.method` | `stratified_kfold`, `repeated_stratified_kfold`, `leave_one_out`/`loo` |
| `cross_validation.n_splits` | intero almeno 2 |
| `cross_validation.n_repeats` | intero almeno 1 |
| `cross_validation.random_state` | seed intero |

### Modelli

Ogni modello espone:

| Chiave | Significato |
|---|---|
| `enabled` | Inclusione nel benchmark standard |
| `params` | Parametri inoltrati all'estimatore scikit-learn |

Modelli: `logistic_l1`, `logistic_l2`, `random_forest`, `svm_linear`,
`svm_rbf`, `svm_poly`, `svm_sigmoid`.

### Importance e reporting

| Chiave | Significato |
|---|---|
| `importance.enabled` | Esecuzione dentro `analysis all` |
| `importance.scoring` | `roc_auc`, `balanced_accuracy`, `accuracy`, `f1` |
| `importance.permutation_repeats` | Ripetizioni di ogni permutazione |
| `importance.n_jobs` | Parallelismo scikit-learn |
| `reporting.top_n_features` | Numero di feature mostrate |
| `reporting.terminal_width` | Larghezza delle tabelle terminale |

## `config/config_clinical.yaml`

| Chiave | Significato |
|---|---|
| `input_csv` | CSV clinico originale, mai modificato |
| `output_csv` | Unico dataset clinico preprocessato prodotto su richiesta |
| `overwrite` | Consente di aggiornare il CSV di output esistente |
| `diagnosis_number` | Numero di diagnosi da selezionare per paziente |
| `absent_marker` | Testo che identifica una diagnosi assente |
| `numeric_columns` | Colonne convertite direttamente in numeri |
| `ordinal_columns` | Mapping espliciti valore -> rango numerico |
| `categorical_columns` | Livelli ammessi per il one-hot encoding |
| `feature_aliases` | Nomi brevi usati nelle colonne `clinical_*` |
| `lymph_node_ratio` | Configurazione del rapporto linfonodale derivato |

Il comando `cli.py clinical` deve essere lanciato esplicitamente. Le normali
analisi non rigenerano e non sovrascrivono il dataset clinico.

## `config/config_tuning.yaml`

| Chiave | Significato |
|---|---|
| `output_dir` | Directory dei run di tuning |
| `strategy` | `randomized` oppure `grid` |
| `scoring` | Metrica ottimizzata nella inner CV |
| `n_iter` | Massimo di configurazioni con `randomized` |
| `n_jobs` | `-1`, `null` o intero positivo |
| `random_state` | Seed della ricerca e degli split |
| `outer_cv.n_splits` | Fold della valutazione esterna |
| `outer_cv.n_repeats` | Ripetizioni della valutazione esterna |
| `inner_cv.n_splits` | Fold della ricerca interna |
| `preprocessing_parameters` | Griglia degli step prima del modello |
| `models.<nome>.enabled` | Inclusione del modello nel tuning |
| `models.<nome>.parameters` | Spazio degli iperparametri del modello |

Tutti i valori degli spazi devono essere liste non vuote. I nomi devono
corrispondere ai parametri della pipeline, per esempio `model__C` o
`feature_selection__k`.
