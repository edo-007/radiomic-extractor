# Risoluzione dei problemi

## La CLI non trova Python

Verificare:

```powershell
Test-Path .\.venv\Scripts\python.exe
```

Se manca, creare l'ambiente e installare le dipendenze come descritto in
[Installazione e avvio](installazione-e-avvio.md).

## Configurazione non trovata

Avviare dalla root del progetto oppure passare il percorso assoluto:

```powershell
.\.venv\Scripts\python.exe .\cli.py extract --config C:\percorso\config_extractor.yaml
```

I percorsi interni relativi allo YAML vengono risolti rispetto alla cartella
dello YAML, non necessariamente rispetto alla directory del terminale.

## RecursionError durante l'importazione DICOM su Windows

Se il traceback ripete `load_metadata`, `describe_self` e
`_get_export_attributes` dentro MIRP, la ricorsione puo' nascondere un errore
di accesso a un file. Con percorsi vicini o superiori a 260 caratteri, Windows
puo' segnalare come assente un DICOM che esiste realmente.

L'estrattore passa a MIRP CT, RTStruct e directory di output tramite
`filesystem_path()`, che aggiunge automaticamente il prefisso Windows `\\?\`.
Non occorre inserirlo negli YAML, spostare i dati o aumentare il limite di
ricorsione. Non viene modificato il codice installato di MIRP.

Se il problema persiste, verificare che i file esistano e siano accessibili e
conservare il traceback completo: la stessa ricorsione puo' mascherare anche
un file realmente mancante. Rilanciare `cli.py extract` crea un nuovo esperimento;
non riprende automaticamente il CSV parziale del run interrotto.

## Nessun esperimento disponibile nell'analisi

Controllare `analysis.experiments_dir` in `config/config_analysis.yaml`. Il selettore
mostra solo sottocartelle contenenti almeno un CSV diverso da
`feature_preview.csv`.

## CSV senza target o feature

Il CSV deve contenere le colonne configurate come `id_column` e
`target_column`, due classi non vuote e almeno una feature numerica. Verificare
anche il separatore, normalmente `;`.

## Pazienti esclusi

Quando `patients_csv` è presente, è la fonte ufficiale di inclusione. Un
paziente DICOM assente dal CSV clinico viene ignorato. Usare `cli.py --pazienti`
per vedere corrispondenze e differenze.

## Errore sul voxel spacing

Tutti i valori devono essere positivi e isotropici:

```yaml
# Valido
voxel_spacing: [[1.25, 1.25, 1.25], [2.0, 2.0, 2.0]]

# Non valido: anisotropico
voxel_spacing: [1.0, 1.0, 2.0]
```

Non ripetere lo stesso spacing nello stesso run.

## Il tempo rimanente non compare subito

Rich deve osservare almeno un'attività completata prima di stimare la velocità.
Durante l'estrazione significa almeno un paziente; durante il tuning significa
almeno una ricerca inner completa.

## Il tuning sembra fermo

Una singola ricerca comprende più configurazioni e più inner fold. Con il
preset corrente il totale è circa 3.465 fit. La barra avanza al termine della
ricerca corrente, quindi può restare sullo stesso valore per parecchio tempo.

Per un test più rapido ridurre temporaneamente:

```yaml
tuning:
  n_iter: 5
  outer_cv:
    n_splits: 3
    n_repeats: 1
  inner_cv:
    n_splits: 2
```

Questa configurazione è utile come smoke test, non come valutazione finale.

## Memoria elevata durante il tuning

Il filtro di correlazione su oltre 10.000 feature può usare molta RAM. Lasciare
`tuning.n_jobs: 1`; aumentarlo solo dopo aver verificato il consumo. La cache
della pipeline evita di ripetere inutilmente parte del preprocessing.

## FutureWarning relativo a SVC probability

Non inserire `probability` nei parametri SVM. ROC-AUC usa direttamente
`decision_function`, quindi `probability=True` non è necessario.

## Permutation importance non disponibile con LOO

Ogni fold LOO contiene un solo paziente e non permette una permutation
importance significativa. Usare `stratified_kfold` o
`repeated_stratified_kfold`.

## Interruzione di un processo

Nel terminale usare `Ctrl+C`. I CSV di estrazione vengono aggiornati dopo ogni
paziente e possono quindi contenere un risultato parziale. Gli artefatti del
tuning vengono consolidati al termine del run: un tuning interrotto prima della
fine non produce il pacchetto completo in `radiomic-output/tuning`.
Le configurazioni e la provenienza sono invece salvate prima dei fit; il report
testuale/HTML viene salvato anche se il tuning fallisce. La presenza della
cartella da sola non indica quindi che il run sia terminato.
