# Estrazione radiomica

## Comando

```powershell
.\.venv\Scripts\python.exe .\cli.py extract --config .\config\config_extractor.yaml
```

## Flusso di esecuzione

1. Lo YAML viene caricato e validato con Pydantic.
2. Vengono cercate le coppie CT e RTStruct dei pazienti.
3. Se presente, `patients_csv` stabilisce quali pazienti includere e associa la
   classe clinica.
4. Vengono mostrati metadati e feature attese.
5. È possibile annullare entro 60 secondi.
6. Viene creata una cartella `extraction-YYYY-MM-DD_HH-MM-SS`.
7. Gli spacing vengono elaborati nell'ordine scritto nello YAML.
8. Il CSV corrente viene aggiornato dopo ogni paziente.

Alla richiesta di conferma:

- `N`, `Esc` o `Ctrl+C` annullano;
- `S` o `Invio` avviano immediatamente;
- senza risposta, l'estrazione parte automaticamente dopo 60 secondi.

La progress bar mostra paziente corrente, avanzamento, tempo trascorso e tempo
rimanente stimato. La stima diventa significativa dopo i primi pazienti.

## Dataset e output

Il file principale richiama quattro profili in `config/extraction/`.
Vedere [configurazione a profili](configurazione-estrazione.md) per sapere dove
modificare ciascun parametro e come vengono salvati gli snapshot.
Gli esempi `mirp:` qui sotto illustrano lo schema; non duplicare i parametri
gia' presenti nei profili.

```yaml
data:
  patients_folder: "C:/percorso/radiomic-data"
  patients_csv: "C:/percorso/radiomic-data/pazienti.csv"
  output_dir: "C:/percorso/radiomic-output"
```

I percorsi relativi sono risolti rispetto alla posizione del file YAML.
`patients_csv` è opzionale, ma senza di esso non è disponibile il target
clinico necessario alla classificazione.

Il CSV clinico deve identificare i pazienti tramite DICOM ID e nome. Il loader
controlla duplicati, corrispondenze e coerenza dei nomi; i pazienti DICOM non
presenti nel CSV clinico vengono esclusi.

## Voxel spacing

Gli spacing devono essere isotropici: z, y e x devono coincidere.

```yaml
mirp:
  voxel_spacing: [1.25, 1.0, 2.0]
```

Forme equivalenti supportate:

```yaml
# Un solo spacing
voxel_spacing: 1.25
voxel_spacing: [1.25]

# Più spacing isotropici in forma compatta
voxel_spacing: [1.25, 2.0]
```

Ogni valore rappresenta uno spacing isotropico completo. Triplette dimensionali,
liste annidate e valori non positivi, non finiti o duplicati vengono rifiutati.
Con `by_slice: true` il codice conserva questo formato uniforme nello YAML e
passa automaticamente a MIRP le sole componenti nel piano `[y, x]`.

## ROI, discretizzazione e feature base

```yaml
mirp:
  bin_width: 10.0
  by_slice: false
  roi_names: ["GTV"]
  resegmentation_intensity_range: [0, .nan]
  feature_families:
    - statistics
    - intensity_histogram
    - morph
    - glcm
    - glrlm
    - glszm
```

- `bin_width`: ampiezza del bin per la discretizzazione fixed-bin-size.
- `by_slice: false`: calcolo tridimensionale; `true` abilita il calcolo 2D.
- `roi_names`: ROI RTStruct da includere.
- `resegmentation_intensity_range`: intervallo HU; `.nan` indica un limite
  aperto.
- `feature_families`: famiglie calcolate sull'immagine originale.

Il codice forza `ibsi_compliant=true`.

## Filtri e response map

Filtri supportati:

- `mean`;
- `laplacian_of_gaussian` o alias `log`;
- `laws`;
- `gabor`;
- `separable_wavelet`;
- `nonseparable_wavelet`.

Esempio:

```yaml
mirp:
  filter_kernels:
    - laplacian_of_gaussian
    - gabor

  response_map_feature_families:
    - statistics
    - intensity_histogram
    - glcm

  response_map_discretisation_n_bins: 16
  boundary_condition: mirror
  laplacian_of_gaussian_sigma: [1.0, 2.0, 3.0]
  gabor_sigma: [1.0, 2.0]
  gabor_lambda: [1.0, 2.0, 4.0]
  gabor_theta: 0.0
  gabor_theta_step: 45.0
  gabor_rotation_invariance: true
  gabor_pooling_method: max
```

Le response map usano una discretizzazione fixed-bin-number, perché dopo il
filtraggio le intensità non conservano necessariamente un'unità fisica diretta.
I parametri specifici di ciascun filtro sono descritti e commentati direttamente
nel file `config/extraction/filters.yaml`.

Per Gabor, `gabor_theta_step` genera le orientazioni e ne abilita il pooling;
`gabor_rotation_invariance: true`, con estrazione 3D, estende il pooling ai tre
piani ortogonali. Con `max` viene conservata voxel per voxel la risposta più
forte, indipendentemente dalla direzione.

Con `by_slice: true`, i kernel Laws e i set wavelet devono avere due componenti
(`l5e5`, `hh`); in 3D ne richiedono tre (`l5e5s5`, `hhh`).

## Metadati DICOM

```yaml
dicom_metadata:
  enabled: true
  tags:
    - StudyDate
    - Manufacturer
    - PixelSpacing
    - tag: StructureSetLabel
      source: rt
```

I tag semplici vengono letti dalla CT. Con `source: rt` il tag viene letto dal
RTStruct. Le colonne aggiunte al CSV hanno prefisso `metadata_`, per esempio
`StudyDate` diventa `metadata_study_date`.

## Parametri di esecuzione

```yaml
mirp:
  num_processes: 1
  write_features: false
  export_features: true
```

L'estrazione batch attiva è sequenziale. `export_features` viene forzato a
`true` quando serve a costruire il CSV aggregato. Se `write_features` è attivo,
gli output nativi MIRP sono separati in sottocartelle per spacing.
