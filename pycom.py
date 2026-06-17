from mirp import extract_features, extract_image_parameters

import pandas as pd

# feature_data = extract_features(
#     mask="./data/20260218-155745_m.dcm",
#     image="./data/CAILOTTO^CLAUDIO_255599_CT_2026-02-12_133102_RT^Torace.(Adult)_Torace..3.0..B30s_n1",
#     base_discretisation_method="fixed_bin_number",
#     base_discretisation_n_bins=16,
#     num_cpus=8,
#     parallel_backend='joblib'
# )

# features_df = feature_data[0]
# features_df.transpose().to_csv('features_out.csv')

image_params_df = extract_image_parameters(
    image="./data/CAILOTTO^CLAUDIO_255599_CT_2026-02-12_133102_RT^Torace.(Adult)_Torace..3.0..B30s_n1",
)
if image_params_df is not None:
    image_params_df.transpose().to_csv('params_out.csv')
