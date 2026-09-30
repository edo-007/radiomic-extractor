from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.benchmark import iter_cv_splits, run_benchmark
from analysis.config import (
    AnalysisConfig,
    CrossValidationConfig,
    ImportanceConfig,
    ModelConfig,
    ModelsConfig,
    PreprocessingConfig,
    ReportingConfig,
)
from analysis.data import load_analysis_dataset
from analysis.importance import run_cross_validated_importance
from analysis.models import build_estimators
from analysis.univariate import run_univariate_analysis


class AnalysisPipelineTest(unittest.TestCase):
    def test_complete_binary_analysis(self) -> None:
        rng = np.random.default_rng(42)
        target = np.array([0] * 8 + [1] * 8)
        signal = target + rng.normal(0.0, 0.15, size=len(target))
        frame = pd.DataFrame(
            {
                "nome_cognome": [f"paziente_{index:02d}" for index in range(16)],
                "stato_microsatellitare": [
                    "INSTABILE" if value else "STABILE" for value in target
                ],
                "metadata_study_date": ["20260101"] * 16,
                "feature_signal": signal,
                "feature_correlated": signal * 2.0,
                "feature_noise": rng.normal(size=16),
                "feature_constant": [1.0] * 16,
            }
        )
        frame.loc[3, "feature_noise"] = np.nan
        # Simula due ROI dello stesso paziente: devono restare nello stesso fold.
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

        with tempfile.TemporaryDirectory() as temporary_directory:
            csv_path = Path(temporary_directory) / "features_test.csv"
            frame.to_csv(csv_path, sep=";", index=False)
            config = AnalysisConfig(
                preprocessing=PreprocessingConfig(correlation_threshold=0.9),
                cross_validation=CrossValidationConfig(
                    method="stratified_kfold",
                    n_splits=4,
                    n_repeats=1,
                    random_state=42,
                ),
                models=ModelsConfig(
                    logistic_l1=ModelConfig(enabled=False),
                    logistic_l2=ModelConfig(
                        params={
                            "C": 1.0,
                            "solver": "liblinear",
                            "class_weight": "balanced",
                            "max_iter": 1000,
                        }
                    ),
                    random_forest=ModelConfig(enabled=False),
                    svm_linear=ModelConfig(enabled=False),
                    svm_rbf=ModelConfig(enabled=False),
                    svm_poly=ModelConfig(enabled=False),
                    svm_sigmoid=ModelConfig(enabled=False),
                ),
                importance=ImportanceConfig(permutation_repeats=2),
                reporting=ReportingConfig(top_n_features=5),
            )

            dataset = load_analysis_dataset(config, csv_path)
            self.assertNotIn("metadata_study_date", dataset.X.columns)
            self.assertEqual(dataset.patient_count, 16)
            self.assertEqual(dataset.duplicated_patient_rows, 2)

            univariate = run_univariate_analysis(dataset, config.univariate)
            self.assertEqual(len(univariate), 4)
            self.assertEqual(univariate.iloc[0]["feature"], "feature_signal")

            estimators = build_estimators(
                config.models,
                random_state=config.cross_validation.random_state,
            )
            benchmark = run_benchmark(dataset, config, estimators)
            self.assertEqual(len(benchmark.folds), 4)
            self.assertGreater(benchmark.summary.iloc[0]["roc_auc_mean"], 0.9)

            importance = run_cross_validated_importance(
                dataset,
                config,
                estimators,
            )
            self.assertEqual(set(importance["feature"]), set(dataset.X.columns))
            self.assertEqual(importance.iloc[0]["feature"], "feature_signal")
            self.assertIn("selection_fraction", importance.columns)
            self.assertLessEqual(
                int(importance["n_measurements"].max()),
                4 * config.importance.permutation_repeats,
            )

            loo_config = config.model_copy(
                update={
                    "cross_validation": CrossValidationConfig(
                        method="loo",
                        random_state=42,
                    )
                }
            )
            self.assertEqual(
                loo_config.cross_validation.method,
                "leave_one_out",
            )
            loo_splits = iter_cv_splits(dataset, loo_config.cross_validation)
            self.assertEqual(len(loo_splits), dataset.patient_count)
            for split in loo_splits:
                train_groups = set(dataset.groups.iloc[split.train_indices])
                test_groups = set(dataset.groups.iloc[split.test_indices])
                self.assertEqual(len(test_groups), 1)
                self.assertTrue(train_groups.isdisjoint(test_groups))

            loo_benchmark = run_benchmark(dataset, loo_config, estimators)
            self.assertEqual(
                int(loo_benchmark.summary.iloc[0]["n_folds"]),
                dataset.patient_count,
            )
            self.assertEqual(
                int(loo_benchmark.folds.iloc[0]["test_rows"]),
                dataset.patient_count,
            )
            self.assertGreater(
                loo_benchmark.summary.iloc[0]["roc_auc_mean"],
                0.9,
            )


if __name__ == "__main__":
    unittest.main()
