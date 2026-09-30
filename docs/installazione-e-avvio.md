# Installazione e avvio

## Requisiti

- Windows con PowerShell o Prompt dei comandi;
- Python compatibile con le versioni presenti in `requirements.txt`;
- dataset DICOM contenente CT e RTStruct;
- ambiente virtuale `.venv` nella root del progetto.

## Preparazione dell'ambiente

Dalla root del progetto:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Per verificare la CLI:

```powershell
.\.venv\Scripts\python.exe .\cli.py --help
```

## File principali

| File | Funzione |
|---|---|
| `config/config_extractor.yaml` | Dataset, metadati DICOM e parametri MIRP |
| `config/config_analysis.yaml` | Preprocessing, modelli e valutazione standard |
| `config/config_tuning.yaml` | Nested CV e spazi degli iperparametri |
| `cli.py` | Punto di ingresso unificato |
| `run_extractor.bat` | Avvio immediato dell'estrattore |
| `deelay_extract.bat` | Avvio ritardato dell'estrattore |

## Comandi principali

```powershell
# Estrazione
.\.venv\Scripts\python.exe .\cli.py extract

# Analisi completa
.\.venv\Scripts\python.exe .\cli.py analysis all

# Tuning
.\.venv\Scripts\python.exe .\cli.py analysis tune
```

I file di configurazione predefiniti sono cercati nella directory corrente.
Per lavorare da un'altra directory conviene passare percorsi espliciti oppure
usare i file `.bat`, che si spostano automaticamente nella root del progetto.

## Avvio tramite batch

`run_extractor.bat` avvia subito `python -m extractor.main` e mantiene aperta
la finestra al termine.

`deelay_extract.bat` verifica l'esistenza di Python e della configurazione,
attende il numero di secondi specificato dal comando `timeout`, quindi esegue:

```bat
cli.py extract --config config/config_extractor.yaml
```

Nel file attuale `timeout /t 54000` corrisponde a 15 ore. Per esempio, per
attendere 10 minuti usare `timeout /t 600 /nobreak`.
