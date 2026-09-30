# Configurazione dell'estrazione a profili

Tutte le configurazioni operative sono in `config/`: `config_extractor.yaml`,
`config_analysis.yaml`, `config_tuning.yaml` e `config_clinical.yaml`, con i
profili in `config/extraction/`. I comandi senza `--config` usano questi file,
indipendentemente dalla directory del terminale. I percorsi negli YAML restano
relativi al file che li contiene. Gli snapshot dei run gia' eseguiti non vengono
spostati o convertiti.

Il comando non cambia:

```powershell
.\.venv\Scripts\python.exe cli.py extract --config config/config_extractor.yaml
```

## Dove modificare ogni impostazione

| File | Contenuto |
|---|---|
| `config/config_extractor.yaml` | Percorsi, metadati DICOM, ROI, spacing, 2D/3D, esecuzione |
| `config/extraction/preprocessing.yaml` | Interpolazione CT/maschera, antialiasing, resegmentazione, normalizzazione |
| `config/extraction/features.yaml` | Famiglie, discretizzazione, IVH, distanze e aggregazione texture |
| `config/extraction/filters.yaml` | Filtri attivi, parametri, response map e condizioni ai bordi |
| `config/extraction/advanced.yaml` | Crop, approssimazioni, gestione maschere, perturbazioni, rim/bulk |

I file contengono commenti con significato, valori ammessi e unita'. Non serve
toccare le opzioni avanzate per una normale estrazione. La suddivisione iniziale
mantiene le stesse impostazioni effettive della configurazione precedente:
spacing 1.25, 1 e 2 mm, elaborazione 3D, range HU [0, infinito), bin width 10,
filtri disattivati, spline CT 3 e maschera 1, nessun crop o perturbazione.

## Composizione senza override

Il file principale dichiara i profili, ciascuno opzionale:

```yaml
profiles:
  preprocessing: extraction/preprocessing.yaml
  features: extraction/features.yaml
  filters: extraction/filters.yaml
  advanced: extraction/advanced.yaml
```

Ogni profilo contiene solo una sezione `mirp`, con chiavi dello stesso schema
usato dal file principale. I nomi dei profili organizzano i file per argomento,
non impongono schemi distinti: un parametro puo' essere spostato, ma mai duplicato.

```yaml
# Esempio di contenuto di un profilo
mirp:
  roi_spline_order: 1
  anti_aliasing: true
```

- I percorsi dei profili sono relativi al file principale, non al terminale.
- Un parametro puo' comparire una sola volta: duplicati tra file, nello stesso
  file generano errori. Usare `num_processes` e `n_test`: gli alias storici
  `num_cpus` e `n-test` non sono supportati.
- Chiavi sconosciute, file mancanti, profili ripetuti e profili annidati sono rifiutati.
- Le combinazioni attive vengono validate con MIRP prima di leggere i DICOM.
- Se ometti un parametro si applica il suo default; il risultato viene registrato
  nello snapshot. Togliere un profilo NON significa necessariamente disabilitare
  tutte le sue operazioni: possono intervenire default diversi dai valori nel file.
- `null` ha un significato specifico per ciascuna opzione: per un metodo spaziale
  sceglie il default 2D/3D, per `filter_kernels` disattiva i filtri.
- Non sono previste conversioni automatiche dei vecchi YAML. Lo snapshot
  risolto usa lo schema corrente ed e' ricaricabile senza profili.

## Spacing isotropici

```yaml
mirp:
  voxel_spacing: [1.25, 1.0, 2.0]
```

La lista viene eseguita nell'ordine indicato, producendo un CSV per spacing.
Un numero singolo (`1.25`) o una lista con un elemento (`[1.25]`) indica un solo
spacing. Triplette dimensionali, liste annidate e valori duplicati, non finiti
o non positivi sono rifiutati. Le tre dimensioni uguali vengono generate
internamente quando i parametri sono passati a MIRP.

## Attivare nuove opzioni

Modificare la voce esistente nel profilo appropriato, senza aggiungerne una
seconda nel file principale. Per esempio:

- nearest neighbour sulla maschera: `roi_spline_order: 0` in preprocessing;
- crop con margine: `crop_around_roi: true` e `crop_distance: 150.0` in advanced;
- disabilitare le approssimazioni: `no_approximation: true` in advanced;
- discretizzazione a numero di bin fisso: `base_discretisation_method:
  fixed_bin_number` e `base_discretisation_n_bins: 32` in features;
- piu' bin width: `bin_width: [10.0, 20.0]`, con metodo `fixed_bin_size`.
  Questo produce piu' colonne nello stesso CSV, non ulteriori CSV per bin width.

`mask_select_largest_slice: true` richiede esplicitamente `by_slice: true`.
I kernel Laws e i set wavelet inattivi sono stati conservati nella forma 2D
preesistente. Prima di attivarli in 3D, scegliere tre componenti, ad esempio
`laws_kernel: [l5e5s5, w5r5l3]` e `separable_wavelet_set: [hhh, lll]`.

Le perturbazioni e la suddivisione delle maschere possono produrre piu' righe
per paziente. Queste righe non sono osservazioni indipendenti: nelle analisi
devono rimanere raggruppate per paziente. I parametri di perturbazione
rimuovono il comportamento standard solo quando esplicitamente attivati.

## Snapshot dell'esperimento

```text
extraction-.../
  config_resolved.yaml
  config_sources/
    00_config_extractor.yaml
    01_preprocessing.yaml
    02_features.yaml
    03_filters.yaml
    04_advanced.yaml
  features_voxel-spacing-1p25mm.csv
  ...
```

`config_resolved.yaml` e' autonomo, senza riferimenti ai profili originali:
contiene percorsi assoluti, opzioni unite, default dello schema e una sezione
`provenance` con versioni software, hash SHA-256 dei sorgenti e tutte le
impostazioni effettive di MIRP per ciascuno spacing (inclusi i default dipendenti
da 2D/3D). `provenance` e' informativa: i valori di esecuzione si modificano
nella sezione `mirp`, non nella copia delle impostazioni effettive.

Si puo' ricaricare lo snapshot con `--config percorso/config_resolved.yaml`,
purche' i dati di input siano ancora disponibili. Questo avvia un NUOVO
esperimento, non riprende quello precedente. Per riprodurre i risultati serve
anche conservare dati originali e ambiente software compatibile.

Analisi e tuning usano `config_resolved.yaml`; non viene creata una copia
`config.yaml`. Le copie in `config_sources/` conservano i byte originali,
commenti inclusi, acquisiti all'avvio: modificare un profilo durante la conferma
o durante l'estrazione non altera la provenienza del run. Sono copie di archivio;
per rieseguire usare lo snapshot risolto, non questi file rinominati.

Se il programma forza `export_features: true` per produrre il CSV, lo snapshot
riporta il valore effettivo mentre il sorgente mantiene il valore richiesto.

## Limiti intenzionali rispetto al template XML

L'estrattore rimane dedicato a CT + RTStruct con `ibsi_compliant=True`.
Le opzioni MRI (N4), PET (conversione SUV), Gaussian e Riesz non sono esposte:
non sono applicabili a questo flusso CT oppure non rientrano nel vincolo IBSI
attuale. Le chiavi XML possono avere nomi differenti dai parametri Python/YAML,
per esempio `incl_threshold` diventa `roi_interpolation_mask_inclusion_threshold`
e `noise_repetitions` diventa `perturbation_noise_repetitions`.
