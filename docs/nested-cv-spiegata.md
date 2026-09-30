# Nested cross-validation: spiegazione pratica

La nested cross-validation (CV annidata) separa due domande:

1. **Quale configurazione devo scegliere?** Risponde la CV interna, o *inner CV*.
2. **Quanto funziona la procedura su pazienti non usati per quella scelta?**
   Risponde la CV esterna, o *outer CV*.

Questa guida spiega il metodo e come viene applicato nel progetto. Per comandi,
configurazione e file prodotti vedi [Tuning](tuning.md).

## 1. Perché una sola CV non basta quando fai tuning

Supponiamo di provare molte configurazioni: diversi valori di `C` per una SVM,
diverse profondità per una Random Forest, diversi numeri di feature.

Ogni punteggio CV contiene una componente di variabilità: dipende anche dai
pazienti finiti nei fold. Scegliendo la configurazione col punteggio più alto,
possiamo scegliere anche quella che è stata particolarmente fortunata su quei
fold, non necessariamente quella che generalizza meglio.

Il punteggio usato per scegliere il vincitore può quindi essere ottimistico.
Non diventa una valutazione indipendente solo perché è stato calcolato in CV.
La nested CV riserva altri pazienti alla valutazione della scelta effettuata.
Questo è il motivo della distinzione tra CV annidata e non annidata descritta
nell'[esempio di scikit-learn](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).

## 2. Esempio con 48 pazienti

Consideriamo 4 fold esterni e 3 fold interni, con una riga per paziente.
I numeri seguenti sono illustrativi: con gruppi e classi sbilanciate le dimensioni
effettive dei fold possono variare.

```text
48 pazienti
│
├── 36 pazienti: outer training
│   │
│   ├── Inner CV a 3 fold, per ogni configurazione candidata:
│   │     24 pazienti per addestrare + 12 per validare
│   │     Ripeti 3 volte, cambiando il gruppo di validazione
│   │     Calcola il punteggio medio della configurazione
│   │
│   ├── Scegli gli iperparametri con la migliore metrica inner
│   └── Riaddestra la configurazione scelta su tutti i 36 pazienti
│
└── 12 pazienti: outer test
      Valuta la configurazione scelta senza modificarla in base al risultato
```

Ripeti l'intero procedimento per ciascuno dei 4 fold esterni: ogni paziente
entra nell'outer test una volta per ripetizione.

Con `n_repeats: 5` vengono generate cinque suddivisioni esterne e si ottengono
20 valutazioni outer per modello. Non sono 20 gruppi indipendenti di nuovi
pazienti: sono sempre gli stessi 48 pazienti, riutilizzati in suddivisioni diverse.

Gli iperparametri e le feature selezionate possono cambiare tra outer fold:
è normale, perché vengono scelti ogni volta usando un training diverso.

## 3. Dove va eseguito il preprocessing

All'interno di ogni inner fold si apprendono soltanto dal suo training:

- valori di imputazione dei dati mancanti;
- identificazione delle feature costanti e correlate;
- parametri dello scaler, quando previsto;
- selezione delle feature;
- parametri del classificatore.

Il validation fold viene trasformato usando le regole apprese sul training.
Dopo la selezione del candidato, l'intera pipeline viene riaddestrata sull'outer
training e applicata all'outer test.

Selezionare prima le feature usando tutti i pazienti e poi fare CV introdurrebbe
informazioni dei pazienti di test nella procedura. La pipeline serve a evitare
questa contaminazione, detta *data leakage*.
Vedi le [indicazioni di scikit-learn sul preprocessing in CV](https://scikit-learn.org/stable/modules/cross_validation.html).

Nel progetto lo scaling è escluso per i nuovi addestramenti Random Forest;
gli altri step restano attivi secondo configurazione. SVM e regressione
logistica rispettano lo scaling configurato.

## 4. Che cosa significa il refit finale sull'intero dataset

Dopo la valutazione outer, serve un modello operativo che sfrutti tutti i dati:

1. Si esegue una nuova ricerca con inner CV sui 48 pazienti. Con 3 fold,
   indicativamente ogni candidato viene addestrato su 32 e validato su 16.
2. Si sceglie la configurazione migliore secondo la metrica `refit`.
3. Si riaddestra l'intera pipeline con quella configurazione su tutti i 48.
4. Si salva la pipeline nel file `.joblib`.

Il modello salvato non viene poi testato su un insieme indipendente all'interno
di questa fase. I suoi punteggi inner sono quelli ottenuti nella ricerca del
punto 1, non metriche calcolate predicendo il training del punto 3.

La outer CV valuta la procedura di addestramento e selezione, non direttamente
lo specifico modello finale addestrato su tutti i dati. Per valutare quest'ultimo
servono pazienti indipendenti, non già usati per prendere decisioni.

## 5. Come leggere i punteggi del report

| Risultato | Che cosa rappresenta | Come usarlo |
|---|---|---|
| Metriche outer in `summary.csv` | Prestazioni sui fold esterni dopo il tuning interno | Valutazione principale della procedura per ciascun modello |
| Metriche inner nel refit finale | Prestazioni CV del candidato scelto nella ricerca su tutti i dati | Informazione sulla selezione, non test indipendente |
| Modello `.joblib` | Pipeline finale addestrata su tutti i pazienti disponibili | Ispezione o predizione di nuovi pazienti |

Con questa configurazione:

```yaml
scoring: [f1, accuracy]
refit: f1
```

viene scelta la configurazione con F1 medio inner più alto. L'accuracy mostrata
è quella della **stessa configurazione**, non il massimo separato di accuracy.

Il report outer contiene ROC-AUC, balanced accuracy, accuracy, sensibilità,
specificità e F1. Attualmente il progetto ne riporta media e deviazione standard
tra i fold esterni. Non è necessariamente lo stesso risultato che si otterrebbe
aggregando tutte le predizioni e calcolando una sola metrica: in particolare F1
e AUC non sono metriche additive.

## 6. Perché fare LOO dopo il tuning non risolve il problema

LOO significa *leave-one-out*: addestri lasciando fuori un paziente alla volta.
Il problema non è LOO in sé, ma l'ordine delle operazioni.

Procedura non indipendente dalla selezione:

1. Scegli configurazione e feature usando tutti i 48 pazienti.
2. Fissi la scelta.
3. Fai LOO addestrando su 47 e predicendo il paziente escluso.

Quel paziente è escluso dall'ultimo fit, ma ha già influenzato il punto 1.
Rifare soltanto il fit non cancella questa informazione.

Una **LOO esterna annidata** separerebbe correttamente le fasi:

1. Escludi un paziente prima di qualsiasi scelta.
2. Sui soli 47 rimasti esegui la ricerca interna, incluso il preprocessing.
3. Riaddestra la configurazione scelta sui 47 e predici il paziente escluso.
4. Ripeti per tutti i pazienti.

L'AUC non si può calcolare su un singolo paziente: in questo caso si aggregano
le predizioni esterne e poi si calcolano le metriche. Ripetere la LOO standard
non crea split diversi; cambiare il seed del modello può cambiare il fit, ma
non rende indipendente una selezione fatta prima su tutti i dati.

Il tuning attuale usa outer CV a fold ripetuti, non outer LOO.

## 7. Cosa fa oggi il nostro codice e quale limite resta

Attualmente il progetto:

- esegue una nested CV separata per ogni famiglia abilitata, per esempio RF,
  SVM e regressione logistica;
- usa gli stessi split outer per confrontarle;
- sceglie dentro l'inner CV gli iperparametri di quella famiglia e i parametri
  di preprocessing presenti nello spazio di ricerca;
- salva un modello finale per ogni famiglia;
- analizza un CSV radiomico selezionato alla volta.

**Non sceglie attualmente la famiglia vincente dentro l'inner CV.** Se guardiamo
i risultati outer, scegliamo la famiglia migliore e presentiamo solo il suo
punteggio, possiamo introdurre ulteriore ottimismo da selezione.

Per valutare l'intera scelta tra famiglie, l'inner CV dovrebbe confrontare
insieme famiglie e iperparametri. Ogni outer test valuterebbe soltanto il
vincitore scelto senza usare quel test. Questa è una possibile estensione,
non una funzionalità già implementata.

Lo stesso vale per la scelta di voxel spacing, filtri, soglie decisionali o
insiemi di variabili: se vengono scelti in base alle prestazioni, fanno parte
della selezione. Vanno fissati prima o inclusi nella procedura interna.
Per confrontare CSV di spacing diversi bisogna inoltre allineare i pazienti
e mantenere gli stessi split; ROI dello stesso paziente non devono finire
contemporaneamente in training e test.

## 8. Cosa non garantisce la nested CV

La nested CV riduce l'ottimismo dovuto alla selezione inclusa nel ciclo interno,
ma non crea nuovi dati e non elimina ogni incertezza:

- con pochi pazienti le metriche possono variare molto tra split;
- ripetizioni aggiuntive descrivono meglio questa variabilità, ma non aumentano
  il numero di pazienti indipendenti;
- cambiare ripetutamente la procedura dopo aver guardato l'outer score può
  trasformare anche la valutazione esterna in uno strumento di selezione;
- non garantisce trasferibilità a nuovi centri o popolazioni differenti.

Anche gli intervalli vanno letti con cautela: l'IC 95% attualmente esportato
usa `1.96 * deviazione_standard / sqrt(numero_fold)`. È un'approssimazione
che non tiene conto della dipendenza tra fold e ripetizioni; non va interpretata
come una garanzia di copertura statistica al 95%.

In sintesi: **inner per scegliere, outer per valutare la procedura, tutti i dati
per addestrare il modello finale, nuovi pazienti per una verifica indipendente.**
