from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analysis.clinical import prepare_clinical_dataset
from analysis.clinical_config import ClinicalConfig, LymphNodeRatioConfig
from analysis.config import AnalysisConfig, ClinicalDataConfig
from analysis.data import load_analysis_dataset


class ClinicalDataTest(unittest.TestCase):
    def _clinical_config(self, input_csv: Path, output_csv: Path) -> ClinicalConfig:
        return ClinicalConfig(
            input_csv=input_csv,
            output_csv=output_csv,
            numeric_columns=["Linfonodi positivi", "Linfonodi esaminati"],
            ordinal_columns={
                "T": {"T1": 1, "T2": 2, "T3": 3, "T4": 4},
                "Stadio": {"Stadio I": 1, "Stadio II": 2, "Stadio III": 3},
            },
            categorical_columns={"Istologia": ["Adenocarcinoma", "Carcinoma"]},
            lymph_node_ratio=LymphNodeRatioConfig(),
        )

    def test_preparation_creates_one_clinical_dataset_on_request(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "clinical_source.csv"
            output = root / "clinical_preprocessed.csv"
            pd.DataFrame(
                {
                    "Paziente": ["Mario Rossi", "Mario Rossi", "Anna Verdi"],
                    "Diagnosi": ["1", "2", "1 (ASSENTE)"],
                    "Data diagnosi": ["01/02/2024", "03/04/2025", "missing"],
                    "T": ["T3", "T4", "missing"],
                    "Stadio": ["Stadio II", "Stadio III", "missing"],
                    "Istologia": ["Adenocarcinoma", "Carcinoma", "missing"],
                    "Linfonodi positivi": ["2", "4", "missing"],
                    "Linfonodi esaminati": ["10", "12", "missing"],
                }
            ).to_csv(source, sep=";", index=False)

            result = prepare_clinical_dataset(
                self._clinical_config(source, output)
            )
            prepared = pd.read_csv(output, sep=";")

            self.assertEqual(result.patients, 1)
            self.assertEqual(result.excluded_absent_diagnoses, 1)
            self.assertEqual(prepared.loc[0, "nome_cognome"], "Mario Rossi")
            self.assertEqual(prepared.loc[0, "clinical_t"], 3.0)
            self.assertEqual(prepared.loc[0, "clinical_stadio"], 2.0)
            self.assertAlmostEqual(
                prepared.loc[0, "clinical_lymph_node_ratio"], 0.2
            )
            self.assertTrue(output.with_suffix(".report.json").is_file())
            self.assertTrue(output.with_suffix(".dictionary.csv").is_file())

    def test_analysis_can_use_clinical_or_combined_features(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            radiomic_csv = root / "radiomic.csv"
            clinical_csv = root / "clinical.csv"
            pd.DataFrame(
                {
                    "nome_cognome": ["Mario Rossi", "Anna Verdi", "Solo Radiomico"],
                    "stato_microsatellitare": ["STABILE", "INSTABILE", "STABILE"],
                    "rad_feature": [1.0, 2.0, 3.0],
                }
            ).to_csv(radiomic_csv, sep=";", index=False)
            pd.DataFrame(
                {
                    "nome_cognome": ["MARIO  ROSSI", "Anna-Verdi", "Solo Clinico"],
                    "clinical_stage": [2.0, 3.0, 4.0],
                }
            ).to_csv(clinical_csv, sep=";", index=False)

            clinical_settings = ClinicalDataConfig(input_csv=clinical_csv)
            clinical_config = AnalysisConfig(
                dataset_mode="clinical",
                clinical_data=clinical_settings,
            )
            clinical_dataset = load_analysis_dataset(
                clinical_config, radiomic_csv
            )
            self.assertEqual(clinical_dataset.patient_count, 2)
            self.assertEqual(
                list(clinical_dataset.X.columns), ["clinical_stage"]
            )
            self.assertEqual(clinical_dataset.excluded_unmatched_patients, 1)
            self.assertEqual(clinical_dataset.unused_clinical_patients, 1)

            combined_config = clinical_config.model_copy(
                update={"dataset_mode": "combined"}
            )
            combined_dataset = load_analysis_dataset(combined_config, radiomic_csv)
            self.assertEqual(
                set(combined_dataset.X.columns),
                {"rad_feature", "clinical_stage"},
            )

            matched_radiomic_config = AnalysisConfig(
                dataset_mode="radiomic",
                clinical_data=ClinicalDataConfig(
                    input_csv=clinical_csv,
                    restrict_radiomic_to_matched_patients=True,
                ),
            )
            matched_radiomic = load_analysis_dataset(
                matched_radiomic_config, radiomic_csv
            )
            self.assertEqual(matched_radiomic.patient_count, 2)
            self.assertEqual(list(matched_radiomic.X.columns), ["rad_feature"])


if __name__ == "__main__":
    unittest.main()
