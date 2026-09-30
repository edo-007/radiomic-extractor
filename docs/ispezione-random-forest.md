# Ispezione delle Random Forest salvate

Non richiede di ripetere tuning o estrazione. Carica soltanto file `.joblib`
fidati, prodotti dai tuoi run: il formato pickle/joblib puo' eseguire codice.
Usa lo stesso ambiente Python del training.

## Comando

```powershell
.\.venv\Scripts\python.exe cli.py analysis inspect-rf
```

Scegli con le frecce uno dei tuning contenenti `models/random_forest.joblib`,
dal piu' recente al piu' vecchio (data del modello salvato). La cartella di
ricerca viene letta da `output_dir` in `config/config_tuning.yaml`.

Oppure indica direttamente un tuning; in questo caso non serve lo YAML:

```powershell
.\.venv\Scripts\python.exe cli.py analysis inspect-rf --run-dir "../radiomic-output/tuning/tuning-2026-09-30_15-00-51"
```

Il comando crea una nuova cartella `rf-inspection-YYYY-MM-DD_HH-MM-SS` dentro
il tuning e stampa il percorso di `index.html`, da aprire nel browser.
Non modifica i modelli ne' sovrascrive le ispezioni precedenti.
`--output-dir PATH` permette di creare la cartella di ispezione altrove.

## Report

- `index.html`: riepilogo, selettore dell'albero, diagramma, statistiche e prime
  30 feature ordinate per importanza.
- `trees/tree-0001.svg`, ...: diagrammi vettoriali apribili singolarmente.
- `trees.csv`: profondita', numero di nodi e foglie per ogni albero.
- `nodes.csv`: tutti i nodi, collegamenti e soglie, sia originali sia del modello.
- `feature_importances.csv`: importanze MDI di tutte le feature selezionate.
- `summary.json`: statistiche, percorso e SHA-256 del modello ispezionato.

La radice ha profondita' zero. `max_depth` del modello e' un limite configurato,
non la profondita' effettivamente raggiunta. `None` significa nessun limite
esplicito, non profondita' infinita.

I diagrammi mostrano per default quattro livelli di decisione sotto la radice.
I sottoalberi nascosti sono segnalati: `--max-depth 6` aumenta la profondita'
visualizzata (valori da 0 a 10). Statistiche e CSV restano sempre completi.
Per alberi larghi usa lo scorrimento o apri il relativo SVG separatamente.
Le feature molto lunghe sono abbreviate nel nodo; passando il mouse sul nodo
si legge il nome completo, disponibile anche nei CSV.

## Scaling e significato delle soglie

I nuovi addestramenti Random Forest, sia benchmark sia tuning e importance,
non applicano scaling, anche se `preprocessing.scaling` e' `standard` o `robust`.
Imputazione, filtri e selezione delle feature restano attivi secondo configurazione.
SVM e regressione logistica continuano a rispettare lo scaling configurato.
La scelta effettiva per modello e' registrata in `source.json` dei nuovi tuning.

I vecchi modelli non vengono alterati. Quando contengono StandardScaler o
RobustScaler, il report riconverte le soglie con i parametri appresi durante
il training e mostra anche la soglia interna. Per valori mancanti il confronto
avviene sul valore imputato. Trasformazioni non supportate producono un errore,
non una conversione approssimata o inventata.

Le classi sono quelle codificate nel modello (nel progetto: 1 = classe positiva
specificata nello YAML di analisi). I campioni dei nodi sono di training/bootstrap,
non pazienti di test. Le importanze MDI sono descrittive, possono essere distorte
e non misurano causalita' o generalizzazione: per un controllo su validation fold
esiste `analysis importance`. Un singolo albero non rappresenta da solo la decisione
complessiva della foresta.
