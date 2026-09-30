# Documentazione di radiomic-extractor

Questa cartella descrive l'intero flusso del progetto: preparazione dei dati,
estrazione MIRP, analisi statistica, benchmark, tuning e tracciamento dei
risultati.

## Percorso consigliato

1. [Installazione e avvio](installazione-e-avvio.md)
2. [Configurazione ed esecuzione dell'estrazione](estrazione.md)
3. [Output e riproducibilità](output-e-riproducibilita.md)
4. [Analisi delle feature](analisi.md)
5. [Preparazione e analisi dei dati clinici](dati-clinici.md)
6. [Tuning con nested cross-validation](tuning.md)
7. [Ispezione e visualizzazione Random Forest](ispezione-random-forest.md)

## Riferimenti

- [Configurazione dell'estrazione a profili](configurazione-estrazione.md)
- [Riferimento completo delle configurazioni YAML](riferimento-configurazioni.md)
- [Comandi CLI e utility DICOM](cli-e-dicom.md)
- [Architettura del codice](architettura.md)
- [Risoluzione dei problemi](risoluzione-problemi.md)

## Flusso generale

```text
DICOM CT + RTStruct + pazienti.csv
              │
              ▼
      config/config_extractor.yaml
              │
              ▼
     Estrazione MIRP per spacing
              │
              ▼
radiomic-output/extraction-.../
  ├── config_resolved.yaml
  ├── feature_preview.csv
  └── features_voxel-spacing-*.csv
              │
              ▼
       Selezione interattiva
              │
       ┌──────┴────────┐
       ▼               ▼
 Analisi standard   Nested tuning
 config_analysis    config_tuning
                       │
                       ▼
              radiomic-output/tuning/tuning-.../
```

Le configurazioni YAML vengono validate prima dell'esecuzione. Errori di nome,
tipo o valore vengono quindi segnalati prima di iniziare elaborazioni lunghe.
