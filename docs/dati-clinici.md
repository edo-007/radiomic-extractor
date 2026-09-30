# Dati clinici

## Creazione esplicita del dataset

Il sorgente contiene 102 righe relative a 50 pazienti. Ogni paziente ha una
riga per ciascuna diagnosi potenziale. La pipeline seleziona per default la
diagnosi numero 1 e scarta le righe marcate `ASSENTE`: nel file attuale produce
45 pazienti con diagnosi primaria valida e 21 feature cliniche.

Il sorgente non viene modificato e il preprocessing non parte durante una
normale analisi. Il singolo dataset clinico viene creato soltanto eseguendo:

```powershell
.\.venv\Scripts\python.exe .\cli.py clinical --config config/config_clinical.yaml
```

Input e output sono definiti in `config/config_clinical.yaml`. Accanto al CSV vengono
salvati un report JSON, un dizionario delle feature e una copia della
configurazione. Sono documentazione del preprocessing, non altri dataset.

## Trasformazioni

- i token `missing` e simili diventano valori mancanti reali;
- `T`, `N`, `M`, `Grading`, residuo post-chirurgico e `Stadio` diventano
  variabili ordinali numeriche;
- le sottocategorie di N, M e stadio vengono accorpate al gruppo principale;
- `Sottosede` e `Istologia` vengono trasformate con one-hot encoding;
- linfonodi positivi ed esaminati diventano numerici;
- viene aggiunto il rapporto linfonodi positivi/esaminati quando valido;
- la data di diagnosi resta un metadato escluso dal modello standard.

La creazione del CSV non imputa e non scala. Imputazione, scaling, filtro di
correlazione e selezione vengono appresi nei training fold durante l'analisi.

## Modalita' di analisi

Il CSV radiomico scelto dal menu resta la fonte di ID e target. La modalita' si
imposta in `config/config_analysis.yaml`:

```yaml
dataset_mode: radiomic   # radiomic, clinical, combined
clinical_data:
  input_csv: "../radiomic-data/DATI_CLINICI/clinical_preprocessed.csv"
  separator: ";"
  id_column: nome_cognome
  restrict_radiomic_to_matched_patients: false
```

- `radiomic`: ignora il CSV clinico;
- `clinical`: analizza soltanto le feature `clinical_*`;
- `combined`: unisce feature radiomiche e cliniche.

Per `clinical` e `combined` viene usata l'intersezione degli ID. Nel dataset
attuale sono 41 pazienti tra i 48 radiomici; il terminale segnala gli esclusi.
La chiave tollera differenze di maiuscole, spazi, trattini, apostrofi e accenti.

Per un confronto sullo stesso campione, eseguire il baseline `radiomic` con
`restrict_radiomic_to_matched_patients: true`; quindi usare `clinical` e
`combined` mantenendo invariati comando e cross-validation.

## Nota metodologica

Nel file attuale la diagnosi primaria e' quasi sempre patologica. Per un modello
preoperatorio, grading, residuo chirurgico o stadiazione patologica potrebbero
non essere disponibili al momento della TC. In tal caso vanno esclusi tramite
`analysis.exclude_columns` o rimossi da `config/config_clinical.yaml`.
