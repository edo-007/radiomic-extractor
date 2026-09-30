# Nested CV: seguiamo un'esecuzione dall'inizio alla fine

La nested cross-validation serve a **scegliere una configurazione senza usare
per questa scelta i pazienti sui quali la valuteremo**.

Seguiamo un solo esempio, dall'inizio al modello salvato: **48 pazienti e una
Random Forest**. Dopo sarà più facile capire che cosa succede con più modelli.

I pazienti, le configurazioni e i punteggi dell'esempio sono illustrativi:
non sono i risultati del tuo dataset. Assumiamo una riga per paziente e gruppi
della stessa dimensione. Nel codice gli split rispettano i gruppi-paziente e
cercano di bilanciare le classi, quindi le dimensioni possono variare.

## Prima di iniziare: quattro parole utili

- **Addestrare, o fare fit:** usare dei pazienti con esito noto per apprendere
  una pipeline, per esempio le mediane di imputazione e gli alberi della foresta.
- **Configurazione:** le impostazioni da provare, per esempio profondità massima
  degli alberi e numero di feature. Non è ancora un modello addestrato.
- **Fold:** un gruppo di pazienti che, a turno, viene lasciato fuori dal fit.
- **Refit:** addestrare di nuovo da zero la pipeline con le impostazioni scelte,
  usando tutti i dati disponibili **in quella fase**.

Ci sono due livelli:

- **Inner CV:** prova le configurazioni e sceglie quale usare.
- **Outer CV:** valuta la configurazione scelta su altri pazienti.

La parola *nested*, cioè annidata, significa che la ricerca inner viene eseguita
**dentro ogni training outer**, non una volta sola prima di tutti i test.

## Passo 0 — Decidiamo che cosa provare

Per questo esempio usiamo:

```yaml
outer_cv:
  n_splits: 4
  n_repeats: 5

inner_cv:
  n_splits: 3

scoring: [f1, accuracy]
refit: f1
```

Significa: quattro test esterni per ripetizione, cinque ripetizioni complete,
tre validazioni interne per candidato. Calcoliamo F1 e accuracy, ma scegliamo
la configurazione in base a **F1 medio inner**, perché `refit: f1`.

Supponiamo di avere soltanto questi tre candidati RF:

| Candidato | Profondità massima | Numero di feature selezionate |
|---|---:|---:|
| R1 | 2 | 5 |
| R2 | 5 | 10 |
| R3 | Nessun limite | 10 |

Gli altri parametri restano fissi nell'esempio. Nella ricerca reale le combinazioni
derivano dalle liste dello YAML: `grid` le prova tutte, `randomized` ne campiona
al massimo `n_iter`.

**Fin qui abbiamo deciso le impostazioni da confrontare: non abbiamo ancora
addestrato né valutato un modello.**

## Passo 1 — Inizia la prima ripetizione: formiamo i gruppi esterni

Dividiamo i 48 pazienti in quattro gruppi da circa 12: **A, B, C e D**.

Per il primo fold esterno:

- **A: 12 pazienti di outer test**, che mettiamo da parte.
- **B+C+D: 36 pazienti di outer training**, che possiamo usare per scegliere
  e addestrare il modello.

I pazienti di A non partecipano al preprocessing, alla scelta delle feature
o degli iperparametri di questo fold. Verranno usati soltanto al passo 6.

## Passo 2 — Dividiamo soltanto i 36 pazienti di training

Per scegliere tra R1, R2 e R3, suddividiamo quei 36 pazienti in tre gruppi
interni da circa 12: **I, II e III**.

Non sono necessariamente i vecchi gruppi B, C e D: è una nuova suddivisione,
costruita esclusivamente con i pazienti del training esterno.

| Fold interno | Pazienti per addestrare | Pazienti per validare |
|---|---|---|
| 1 | II + III: 24 pazienti | I: 12 pazienti |
| 2 | I + III: 24 pazienti | II: 12 pazienti |
| 3 | I + II: 24 pazienti | III: 12 pazienti |

**A è ancora fuori da tutto questo.** Anche se ogni validazione interna usa 12
pazienti, non sono i 12 del test esterno.

## Passo 3 — Proviamo il primo candidato, R1

Per il primo fold interno:

1. Creiamo una nuova pipeline con le impostazioni di R1.
2. Sui 24 pazienti II+III apprendiamo il preprocessing: imputazione, filtri
   e selezione delle feature. Per RF non applichiamo scaling.
3. Sempre su quei 24 addestriamo la foresta.
4. Trasformiamo i 12 pazienti di I con le regole appena apprese.
5. Prediciamo il loro esito e calcoliamo F1 e accuracy.

Poi ricominciamo da zero per il secondo e il terzo fold interno, cambiando
quali pazienti servono al training e quali alla validazione.

Alla fine abbiamo **tre modelli temporanei e tre punteggi per R1**. Facciamo
la media dei punteggi: non prendiamo il migliore dei tre.

Anche quando il numero di feature resta uguale, i loro nomi possono cambiare:
la selezione viene appresa ogni volta sul training di quel fold.

## Passo 4 — Proviamo anche R2 e R3 e scegliamo le impostazioni

Ripetiamo il passo 3 per R2 e R3 usando **gli stessi split interni** per rendere
confrontabili i candidati.

Supponiamo di ottenere questi F1, inventati solo per illustrare il calcolo:

| Candidato | F1 interno 1 | F1 interno 2 | F1 interno 3 | F1 medio |
|---|---:|---:|---:|---:|
| R1 | 0,70 | 0,65 | 0,75 | 0,70 |
| R2 | 0,80 | 0,75 | 0,85 | **0,80** |
| R3 | 0,70 | 0,80 | 0,75 | 0,75 |

Vince **R2**, perché ha la media F1 più alta.

L'accuracy viene calcolata e conservata, ma in questo esempio non decide il
vincitore. Se R3 avesse accuracy maggiore, sceglieremmo comunque R2.

**Abbiamo scelto una configurazione, non uno dei tre modelli temporanei di R2.**
I pazienti di A non hanno ancora influenzato nulla.

## Passo 5 — Riaddestriamo R2 sui 36 pazienti

Ora la scelta è fatta. Creiamo una nuova pipeline con le impostazioni R2 e
riapprendiamo preprocessing e foresta su **tutti i 36 pazienti B+C+D**.

Non conserviamo il modello interno che aveva ottenuto F1 0,85: era stato
addestrato soltanto su 24 pazienti.

Questo è già un **refit**, ma riguarda il training di un singolo fold esterno.
Non è ancora il refit finale sui 48 pazienti.

## Passo 6 — Usiamo finalmente i 12 pazienti del test esterno

Applichiamo la pipeline del passo 5 ai pazienti di A e ne prediciamo l'esito.
Durante questa operazione non apprendiamo nulla da A e non modifichiamo la
configurazione per migliorare il risultato.

Calcoliamo le metriche outer: F1, accuracy, balanced accuracy, sensibilità,
specificità e ROC-AUC, quando definita.

Per esempio, l'F1 outer potrebbe essere **0,67**, diverso dallo 0,80 inner:
non è una contraddizione, perché stiamo valutando su pazienti diversi.

**Questa è la prima valutazione esterna completata.** Conserviamo punteggi e
parametri scelti per questo fold.

## Passo 7 — Completiamo gli altri tre fold esterni

Manteniamo la divisione A/B/C/D, ma cambiamo il gruppo lasciato fuori:

| Fold esterno | Dati sui quali rifare tutta la ricerca inner e il refit | Test esterno |
|---|---|---|
| 1, appena completato | B+C+D | A |
| 2 | A+C+D | B |
| 3 | A+B+D | C |
| 4 | A+B+C | D |

Per ogni riga **ripetiamo i passi 2–6**: riproviamo tutti i candidati sui dati
disponibili in quel training. Non riutilizziamo automaticamente R2.

Nel secondo fold potrebbe vincere R1, nel terzo R3: è normale.

A può ora essere usato come training nei fold 2–4. Questo è corretto: quando
abbiamo valutato su A nel fold 1, la relativa pipeline non lo aveva usato.

Alla fine della prima ripetizione abbiamo quattro valutazioni outer.
Ogni paziente è stato nel test esattamente una volta.

## Passo 8 — Eseguiamo le altre ripetizioni: questo è `n_repeats`

Con `n_repeats: 5` rifacciamo l'intero procedimento cinque volte in totale.

Nella seconda ripetizione rimescoliamo i pazienti e formiamo quattro gruppi
con una nuova composizione. Per ciascun nuovo fold rifacciamo ricerca interna,
refit sul training esterno e test. Poi procediamo allo stesso modo per le
ripetizioni 3, 4 e 5.

Quindi:

- 4 fold × 5 ripetizioni = **20 valutazioni outer per la Random Forest**.
- Ogni paziente viene usato come test cinque volte complessivamente.
- Non abbiamo 240 pazienti diversi: i pazienti sono sempre 48.

Serve a osservare quanto il risultato dipende dalla suddivisione dei pazienti.
Non è un modo per scegliere la ripetizione più fortunata e aumenta il costo
computazionale.

## Passo 9 — Riassumiamo le 20 valutazioni, senza scegliere il modello migliore

Per ogni metrica il codice calcola media e deviazione standard dei 20 risultati
outer. Questo è il riepilogo della procedura «scegli gli iperparametri RF tramite
inner CV, poi addestra RF».

**Non scegliamo il modello con F1 outer più alto.** Potrebbe aver avuto un test
più facile; selezionarlo in base al test renderebbe quel test parte della scelta.

Non facciamo nemmeno la media degli iperparametri e non prendiamo automaticamente
la configurazione che ha vinto più spesso.

La media delle metriche dei fold non coincide necessariamente con una metrica
calcolata aggregando tutte le predizioni: in particolare F1 e AUC non sono
metriche additive. Il report attuale usa le medie tra fold.

A questo punto abbiamo valutato la procedura. **Dobbiamo ancora costruire il
modello definitivo da salvare.**

## Passo 10 — Facciamo una nuova ricerca usando tutti i 48 pazienti

Questa fase è separata dalle 20 valutazioni precedenti.

Prendiamo tutti i 48 pazienti e facciamo una nuova inner CV a tre fold:

1. Per ogni candidato, tre addestramenti su circa 32 pazienti e tre validazioni
   sui rispettivi gruppi esclusi da circa 16.
2. Preprocessing e selezione delle feature vengono riappresi dentro ogni fold.
3. Calcoliamo il F1 medio di ogni candidato.
4. Scegliamo la configurazione con il F1 medio più alto.

La configurazione vincente può essere diversa da quelle scelte nei fold outer.
Non viene scelta usando il massimo dei 20 punteggi esterni.

I valori nella tabella **«REFIT FINALE SULL'INTERO DATASET»**, colonne
`Inner f1` e `Inner accuracy`, provengono da questa ricerca:
sono le medie della stessa configurazione vincente.

Non sono punteggi di training, ma non sono neppure un test indipendente:
abbiamo usato proprio questi risultati inner per scegliere la configurazione.

## Passo 11 — Addestriamo su tutti i 48 e salviamo

Con le impostazioni scelte al passo 10:

1. Creiamo una nuova pipeline.
2. Apprendiamo preprocessing e modello usando tutti i 48 pazienti.
3. Salviamo la pipeline in `models/random_forest.joblib`.
4. Registriamo gli iperparametri in `best_parameters.json`.

Questo è il **refit finale**. Il modello salvato non proviene da una particolare
ripetizione e non è uno dei modelli temporanei addestrati su 24, 32 o 36 pazienti.

Dopo questo fit non rimane un gruppo indipendente di test nei nostri 48
pazienti. Per valutare direttamente questo specifico modello finale servono
nuovi pazienti mai utilizzati nelle scelte precedenti.

**Le metriche outer valutano la procedura; il file `.joblib` contiene il modello
operativo costruito alla fine. Sono due risultati diversi dello stesso lavoro.**

## Che cosa cambia se abilitiamo anche SVM e regressione logistica?

Il codice attuale esegue i passi descritti **separatamente per ogni famiglia**,
usando gli stessi split outer per confrontarle. Completa la valutazione e il
refit finale di una famiglia, poi passa alla successiva.

Otteniamo quindi un riepilogo outer e un modello finale salvato per ogni
famiglia abilitata. **Non scegliamo attualmente la famiglia vincente nella
inner CV**, e non salviamo un unico vincitore globale.

Se scegliamo RF perché ha il miglior punteggio outer tra le famiglie,
il risultato del vincitore può essere ottimistico. Per valutare anche questa
scelta, l'inner CV dovrebbe confrontare insieme famiglie e iperparametri:
sarebbe un'estensione del codice attuale.

Lo stesso vale per voxel spacing, filtri o soglie decisionali scelti guardando
i risultati: vanno fissati prima o inclusi nella selezione interna. Attualmente
il tuning lavora su un CSV selezionato alla volta. Per confrontare più CSV
bisogna allineare i pazienti e mantenere gli stessi split.

## Perché una LOO fatta dopo la selezione non basta?

Supponi di scegliere gli iperparametri usando tutti i 48 pazienti, e solo dopo
fare LOO: ogni volta addestri su 47 e predici il paziente escluso.

Quel paziente non ha partecipato all'ultimo fit, ma **ha già contribuito alla
scelta degli iperparametri**. La sua esclusione arriva troppo tardi.

Una LOO esterna annidata farebbe invece questo:

1. Esclude un paziente prima della scelta.
2. Sui soli 47 rimasti esegue tutta la ricerca inner.
3. Riaddestra la configurazione scelta sui 47.
4. Predice il paziente escluso e ripete per tutti i pazienti.

Per calcolare l'AUC si aggregano le predizioni esterne, perché un test di un
solo paziente non contiene entrambe le classi. Ripetere la LOO standard non
crea split diversi; cambiare il seed del modello non cancella la selezione
precedente.

Il nostro tuning attuale usa outer CV a fold ripetuti, non outer LOO.

## Quali file guardare

| File | Che cosa contiene |
|---|---|
| `outer_fold_metrics.csv` | Metriche esterne e parametri scelti nei singoli fold, passi 6–8 |
| `summary.csv` | Riepilogo delle metriche esterne, passo 9 |
| `cv_results.csv` | Risultati delle ricerche interne; `phase=outer` per i fold esterni, `phase=final` per la ricerca del passo 10 |
| `best_parameters.json` | Parametri e punteggi inner della ricerca finale, passo 10 |
| `selected_features.csv` | Feature selezionate dalla pipeline finale su tutti i dati, passo 11 |
| `models/*.joblib` | Pipeline finali addestrate, passo 11 |
| `report.txt` / `report.html` | Le tabelle mostrate nel terminale |

## Limiti da ricordare

Con pochi pazienti le stime restano incerte. Ripetere la CV non crea nuovi
pazienti, e i fold e le ripetizioni non sono indipendenti.

L'IC 95% attualmente esportato usa
`1.96 * deviazione_standard / sqrt(numero_fold)`: è un'approssimazione che non
tiene conto di questa dipendenza, non una garanzia di copertura al 95%.

Cambiare continuamente la procedura dopo aver consultato l'outer score può
trasformare anche l'outer CV in uno strumento di selezione. Inoltre la nested
CV non garantisce prestazioni uguali su un altro centro o una popolazione diversa.

Per la parte operativa vedi [Tuning](tuning.md). Per approfondire il principio:
[CV annidata in scikit-learn](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)
e [separazione di training, preprocessing e valutazione](https://scikit-learn.org/stable/modules/cross_validation.html).
