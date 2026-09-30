# Tuning con nested cross-validation

Per una spiegazione con un esempio su 48 pazienti, refit finale e confronto
con LOO, leggi [Nested CV spiegata](nested-cv-spiegata.md).

## Perché è annidato

La inner CV sceglie gli iperparametri; la outer CV valuta il modello senza aver
partecipato alla scelta:

```text
Outer fold
├── Outer training
│   └── Inner CV
│       ├── prova configurazione 1
│       ├── prova configurazione 2
│       └── sceglie la migliore
└── Outer test
    └── valuta una volta la configurazione scelta
```

Usare gli stessi fold per scelta e valutazione produrrebbe una stima troppo
ottimistica. Con pochi pazienti la nested CV non elimina l'incertezza, ma evita
questa specifica forma di leakage.

## Comando

```powershell
.\.venv\Scripts\python.exe .\cli.py analysis tune
```

Percorsi espliciti:

```powershell
.\.venv\Scripts\python.exe .\cli.py analysis tune `
  --config .\config\config_analysis.yaml `
  --tuning-config .\config\config_tuning.yaml
```

Dopo la selezione di esperimento e CSV viene mostrato il numero stimato di fit.
La progress bar avanza al completamento di ciascuna ricerca outer e mostra il
tempo rimanente stimato.

## Configurazione

```yaml
tuning:
  output_dir: "../../radiomic-output/tuning"
  strategy: randomized
  scoring: [f1, accuracy]
  refit: f1
  n_iter: 20
  n_jobs: 1
  random_state: 42

  outer_cv:
    n_splits: 4
    n_repeats: 5

  inner_cv:
    n_splits: 3
```

- `randomized` estrae al massimo `n_iter` combinazioni.
- `grid` prova tutte le combinazioni e ignora `n_iter`.
- `output_dir` e' relativo alla cartella dello YAML, non al terminale.
- `scoring` accetta una metrica o una lista senza duplicati tra `roc_auc`,
  `balanced_accuracy`, `accuracy`, `f1`.
- `refit` sceglie gli iperparametri: deve essere in `scoring`; se omesso, usa
  la prima metrica. Aggiungere metriche non moltiplica i fit: vengono valutate
  sulle stesse configurazioni e sugli stessi fold.
- `n_jobs: 1` limita i picchi di RAM dovuti alle oltre 10.000 feature.
- La Random Forest può continuare a parallelizzare internamente tramite il suo
  parametro `n_jobs` in `config/config_analysis.yaml`.

Configurazione consigliata per il dataset attuale: outer 4-fold ripetuta 5
volte, inner 3-fold e ricerca limitata. Il preset corrente produce 20 outer
fold e circa 3.465 fit complessivi, quindi può richiedere diverse ore.

## Spazi di ricerca

I nomi seguono la sintassi delle pipeline scikit-learn:

```yaml
preprocessing_parameters:
  feature_selection__k: [5, 10, 20]

models:
  logistic_l2:
    enabled: true
    parameters:
      model__C: [0.01, 0.1, 1.0, 10.0, 100.0]

  random_forest:
    enabled: true
    parameters:
      model__n_estimators: [100, 300, 500]
      model__max_depth: [null, 5, 10, 20]
      model__min_samples_leaf: [1, 2, 4]
      model__max_features: [sqrt, log2, 0.3]

  svm_rbf:
    enabled: true
    parameters:
      model__C: [0.01, 0.1, 1.0, 10.0, 100.0]
      model__gamma: [scale, 0.0001, 0.001, 0.01, 0.1]
```

`model__` indirizza il parametro al modello; `feature_selection__` allo step di
selezione. Un nome inesistente produce un errore prima del fit.

I parametri base non inclusi nella ricerca provengono dal blocco del modello in
`config/config_analysis.yaml`, anche quando quel modello è disabilitato nel benchmark
standard. L'abilitazione del tuning è controllata da `config/config_tuning.yaml`.

## Procedura completa

Per ogni modello:

1. vengono generati gli outer fold raggruppati per paziente;
2. in ogni outer training viene eseguita la ricerca inner;
3. la configurazione migliore viene valutata sull'outer test;
4. vengono aggregate media, deviazione standard e IC 95% approssimativo;
5. una ricerca finale sull'intero dataset sceglie i parametri operativi;
6. la pipeline finale viene addestrata e serializzata.

La cache temporanea riusa le trasformazioni tra configurazioni e viene rimossa
al termine. Non viene incorporata nei modelli salvati.

## Interpretazione

- `summary.csv`: confronto imparziale basato sugli outer fold.
- `outer_fold_metrics.csv`: risultato di ogni outer fold e parametri scelti.
- `cv_results.csv`: tutte le configurazioni provate nelle inner CV.
- `best_parameters.json`: ricerca finale sull'intero dataset.
- `selected_features.csv`: feature del modello finale, non frequenze outer.

Il riepilogo finale nel terminale e `summary.csv` riportano, per la outer nested
CV, ROC-AUC, balanced accuracy, accuracy, sensibilita', specificita' e F1. Per
ogni metrica sono disponibili media e deviazione standard; il report mostra
anche l'intervallo al 95% approssimato della ROC-AUC.

La tabella del refit sull'intero dataset mostra una colonna `Inner <metrica>`
per ogni voce di `scoring`. Tutti i punteggi appartengono alla stessa
configurazione vincente per `refit`, non ai massimi separati delle metriche.
Per esempio, con `[f1, accuracy]` e `refit: f1` viene mostrata anche l'accuracy
del candidato scelto per F1. Non rappresentano una valutazione indipendente
e non vanno sostituiti alle metriche outer. La tabella outer e' ordinata per
la metrica `refit`; scegliere tra modelli dopo averla consultata puo' introdurre
ulteriore ottimismo nella prestazione del vincitore.

## Configurazioni e report salvati

Random Forest non applica scaling nei nuovi fit; gli altri modelli rispettano
`preprocessing.scaling`. La scelta effettiva e' registrata in `source.json`.
Per visualizzare gli alberi dei modelli salvati usa `analysis inspect-rf`:
vedi [Ispezione Random Forest](ispezione-random-forest.md).

Prima dei fit viene creata `radiomic-output/tuning/tuning-.../` e vengono
salvati gli stessi byte degli YAML letti dai loader, con commenti e formattazione.
`config_analysis_resolved.yaml` e `config_tuning_resolved.yaml` contengono anche
default e percorsi assoluti e possono essere ricaricati con i rispettivi flag CLI.
Cambiare gli YAML mentre il tuning gira non modifica gli snapshot.

`extraction_config/` conserva byte per byte gli YAML trovati nell'estrazione
selezionata e tutti i file in `config_sources/`, mantenendo nomi e struttura.
`config_extraction.yaml` e' inoltre una copia identica del suo `config_resolved.yaml`.
Non viene usata la configurazione di estrazione corrente del progetto. Se manca
il file risolto, viene segnalato e si archiviano soltanto i file disponibili.

`report.txt` conserva le tabelle del terminale in testo UTF-8; `report.html`
mantiene anche colori e impaginazione. La progress bar animata non viene registrata.
`best_parameters.json` include la metrica di selezione e `inner_cv_metrics`;
`cv_results.csv` include media, deviazione standard e rango per ogni score.

Le nuove estrazioni si chiamano `extraction-...`; i nomi delle estrazioni
storiche restano invariati e sono ancora selezionabili. Lo spostamento dei report
storici nella nuova directory non ne rigenera il contenuto: i nuovi artefatti
sono disponibili soltanto per i nuovi tuning.

Differenze piccole, per esempio ROC-AUC 0,71 contro 0,73, non vanno considerate
conclusive senza osservare deviazione standard e intervallo di confidenza.
