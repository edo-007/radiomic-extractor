# Analisi delle feature

## Selezione del dataset

`config/config_analysis.yaml` non contiene un `input_csv` fisso. Contiene solamente la
directory generale:

```yaml
analysis:
  experiments_dir: "../radiomic-output"
```

Ogni comando mostra due menu:

1. selezione dell'esperimento, dal più recente al più vecchio;
2. selezione del CSV delle feature dentro l'esperimento.

Usare `↑` e `↓` per cambiare opzione, `Invio` per confermare e `Ctrl+C` per
annullare. È disponibile anche il mouse. `feature_preview.csv` viene escluso
dalla lista dei dataset.

## Sorgenti delle feature

`dataset_mode` controlla il contenuto di `X` senza creare automaticamente
nuovi CSV:

```yaml
dataset_mode: radiomic  # radiomic, clinical, combined
clinical_data:
  input_csv: "../radiomic-data/DATI_CLINICI/clinical_preprocessed.csv"
  separator: ";"
  id_column: nome_cognome
  restrict_radiomic_to_matched_patients: false
```

Il dataset clinico viene creato separatamente con `cli.py clinical`. In
modalita' `clinical` o `combined`, il collegamento e' un inner join sull'ID
normalizzato; ID e target provengono sempre dal CSV radiomico scelto. Vedere
[Dati clinici](dati-clinici.md).

## Comandi

```powershell
# Test univariati
.\.venv\Scripts\python.exe .\cli.py analysis univariate

# Confronto dei modelli configurati
.\.venv\Scripts\python.exe .\cli.py analysis benchmark

# Permutation importance sui validation fold
.\.venv\Scripts\python.exe .\cli.py analysis importance

# Univariata + benchmark + importance
.\.venv\Scripts\python.exe .\cli.py analysis all
```

## Colonne e target

```yaml
analysis:
  separator: ";"
  id_column: nome_cognome
  target_column: stato_microsatellitare
  positive_class: INSTABILE
  exclude_columns: []
  metadata_prefix: "metadata_"
  include_metadata: false
```

Il target deve contenere esattamente due classi. Tutte le righe dello stesso
paziente devono avere lo stesso target. Le colonne non numeriche vengono
ignorate; conversioni fallite sono trattate come valori mancanti e segnalate.
Per impostazione predefinita le colonne `metadata_*` non sono predittori.

## Pipeline anti-leakage

```text
Training fold
  └── imputazione
      └── rimozione feature costanti
          └── filtro di correlazione
              └── scaling
                  └── selezione feature
                      └── modello
```

Ogni passaggio viene appreso esclusivamente sul training fold. Il validation o
test fold viene solamente trasformato con le regole già apprese. Questo evita
che informazione esterna entri nella selezione delle feature.

Configurazione:

```yaml
preprocessing:
  impute_strategy: median
  remove_constant_features: true
  correlation_threshold: 0.90
  scaling: standard
  feature_selection:
    enabled: true
    method: f_classif
    k: 20
```

- Imputazione: `median`, `mean` o `most_frequent`.
- Scaling: `standard`, `robust` o `none`; sempre escluso nei nuovi fit Random Forest.
- Selezione: `f_classif` o `mutual_info`.
- `k` viene ridotto automaticamente se nel fold restano meno feature.
- `correlation_threshold: null` disabilita il filtro di correlazione.

## Analisi univariata

```yaml
univariate:
  enabled: true
  test: mannwhitney
  correction: fdr_bh
```

Sono disponibili Mann-Whitney e Welch t-test. La correzione multipla può essere
FDR Benjamini-Hochberg oppure `none`. Questa analisi è descrittiva: i risultati
non vengono usati fuori dai fold per selezionare le feature del benchmark.

## Cross-validation

```yaml
cross_validation:
  method: repeated_stratified_kfold
  n_splits: 5
  n_repeats: 5
  random_state: 42
```

Metodi disponibili:

- `stratified_kfold`;
- `repeated_stratified_kfold`;
- `leave_one_out`, con alias `loo`.

L'ID paziente viene sempre usato come gruppo: ROI o righe dello stesso paziente
non possono finire contemporaneamente in training e validation.

Con LOO viene escluso un paziente alla volta e `n_splits`/`n_repeats` vengono
ignorati. La permutation importance non è disponibile con LOO perché ogni
validation fold contiene un solo paziente.

## Modelli

Sono configurabili:

- regressione logistica L1 e L2;
- Random Forest;
- SVM lineare, RBF, polinomiale e sigmoid.

Ogni blocco contiene `enabled` e i parametri scikit-learn:

```yaml
models:
  svm_rbf:
    enabled: true
    params:
      C: 1.0
      gamma: scale
      class_weight: balanced
```

Il parametro deprecato `SVC.probability` non serve: per ROC-AUC viene usata la
`decision_function`.

## Metriche e importance

Il benchmark produce ROC-AUC, balanced accuracy, accuracy, sensibilità,
specificità e F1. Con classi bilanciate ma dataset piccolo è comunque utile
osservare sia ROC-AUC sia balanced accuracy e la variabilità tra fold.

La permutation importance viene calcolata sui validation fold, mai sul training
usato per adattare il modello. `selection_fraction` indica quanto spesso una
feature ha superato la selezione interna.
