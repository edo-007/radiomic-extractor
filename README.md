# radiomic-extractor
orchestration and extraction of radiomic features via mirp

## Struttura

- `extractor/`: moduli Python per scansione pazienti ed estrazione feature.
- `config_extractor.yaml`: configurazione dell'estrattore, mantenuta nella root del progetto.

Avvio:

```bash
python -m extractor.main
```

Prima dell'estrazione il comando stampa un recap con numero e nomi delle feature
radiomiche previste, salva la lista completa in `results/feature_preview.csv` e
chiede conferma con `Continuare con l'estrazione radiomica? [s/N]`.

Se `patients_csv` e' configurato, i pazienti non presenti nel CSV clinico
vengono ignorati.

Info DICOM:

```bash
python cli.py --infodicom path/to/file.dcm
python cli.py --infodicom path/to/file.dcm --tag PatientID --tag "(0008,0020)"
```

Uso da codice:

```python
from extractor import DicomMetadataService

service = DicomMetadataService()
record = service.extract_from_file(
    "path/to/file.dcm",
    ["PatientID", "PatientName", "(0008,0020)"],
)
metadata = record.as_dict()
```

Metadati DICOM nel CSV delle feature:

```yaml
dicom_metadata:
  enabled: true
  tags:
    # Data dello studio CT nel formato DICOM YYYYMMDD.
    - StudyDate
```

La colonna generata nel CSV sara' `metadata_study_date`.

Filtri MIRP IBSI-compliant:

```yaml
mirp:
  # Il codice forza sempre ibsi_compliant=true.
  filter_kernels:
    - laplacian_of_gaussian
    - gabor
  response_map_feature_families:
    - statistics
    # - intensity_histogram
    # - glcm
  response_map_discretisation_n_bins: 16
  laplacian_of_gaussian_sigma: [5.0, 10.0]
  gabor_sigma: [2.0, 4.0]
  gabor_lambda: [1.0, 2.0]
```

Sono esposti solo i filtri IBSI-compliant: `mean`,
`laplacian_of_gaussian`/`log`, `laws`, `gabor`, `separable_wavelet` e
`nonseparable_wavelet`. Gaussian, Laplace semplice (`laplace`/`laplacian`),
Sobel, Prewitt, LBP e le varianti Riesz non sono configurabili perche'
richiedono `ibsi_compliant=false` o non hanno reference values IBSI.
