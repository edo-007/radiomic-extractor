from __future__ import annotations

import tempfile
import unittest
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pydantic import ValidationError
from rich.console import Console

from analysis.artifacts import prepare_tuning_artifacts, save_tuning_artifacts
from analysis.benchmark import METRICS
from analysis.config import AnalysisConfig, PreprocessingConfig, load_analysis_configuration
from analysis.data import AnalysisDataset
from analysis.reporting import TerminalReporter
from analysis.tuning import run_nested_tuning, _build_search
from analysis.tuning_config import (
    InnerCVConfig,
    OuterCVConfig,
    TuningConfig,
    TuningModelConfig,
    load_tuning_configuration,
)


class NestedTuningTest(unittest.TestCase):
    def test_small_nested_tuning_produces_unbiased_folds_and_final_model(self) -> None:
        rng = np.random.default_rng(7)
        y = np.array([0] * 8 + [1] * 8)
        signal = y + rng.normal(0.0, 0.2, len(y))
        X = pd.DataFrame(
            {
                "signal": signal,
                "correlated": signal * 2.0,
                "noise_1": rng.normal(size=len(y)),
                "noise_2": rng.normal(size=len(y)),
            }
        )
        dataset = AnalysisDataset(
            input_csv=Path("synthetic.csv"),
            X=X,
            y=pd.Series(y),
            groups=pd.Series([f"patient-{index}" for index in range(len(y))]),
            positive_label="positive",
            negative_label="negative",
            ignored_non_numeric_columns=(),
            coerced_missing_values={},
        )
        analysis_config = AnalysisConfig(
            preprocessing=PreprocessingConfig(correlation_threshold=0.9),
        )
        tuning_config = TuningConfig(
            strategy="grid",
            scoring=["f1", "accuracy", "roc_auc"],
            refit="f1",
            n_jobs=1,
            outer_cv=OuterCVConfig(n_splits=2, n_repeats=1),
            inner_cv=InnerCVConfig(n_splits=2),
            preprocessing_parameters={"feature_selection__k": [1, 2]},
            models={
                "logistic_l2": TuningModelConfig(
                    parameters={"model__C": [0.1, 1.0]}
                )
            },
        )

        result = run_nested_tuning(dataset, analysis_config, tuning_config)

        self.assertEqual(len(result.outer_folds), 2)
        self.assertEqual(result.summary.iloc[0]["model"], "logistic_l2")
        for metric in METRICS:
            self.assertIn(f"{metric}_mean", result.summary.columns)
            self.assertIn(f"{metric}_std", result.summary.columns)
            self.assertIn(f"{metric}_ci95", result.summary.columns)
        self.assertIn("logistic_l2", result.final_parameters)
        self.assertIn("logistic_l2", result.final_models)
        self.assertFalse(result.search_results.empty)
        self.assertFalse(result.selected_features.empty)
        final_rows = result.search_results.query("phase == 'final'")
        best_row = final_rows.loc[final_rows["rank_test_f1"].idxmin()]
        details = result.final_parameters["logistic_l2"]
        self.assertEqual(details["selection_metric"], "f1")
        for metric in tuning_config.score_names():
            self.assertEqual(details["inner_cv_metrics"][metric], best_row[f"mean_test_{metric}"])
        self.assertEqual(details["inner_cv_score"], details["inner_cv_metrics"]["f1"])

        # Anche lo scoring singolo usa il refit nominato e la ricerca randomized.
        single = TuningConfig(
            strategy="randomized", scoring="accuracy", n_iter=1,
            inner_cv=InnerCVConfig(n_splits=2), models=tuning_config.models,
        )
        search = _build_search(
            model_name="logistic_l2", parameter_space={"model__C": [0.1, 1.0]},
            analysis_config=analysis_config, tuning_config=single,
            random_state=42, memory=None,
        )
        search.fit(dataset.X, dataset.y, groups=dataset.groups)
        self.assertEqual(search.refit, "accuracy")
        self.assertEqual(search.best_score_, search.cv_results_["mean_test_accuracy"][search.best_index_])

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_csv = root / "features.csv"
            analysis_yaml = root / "config_analysis.yaml"
            tuning_yaml = root / "config_tuning.yaml"
            input_csv.write_text("feature\n1\n", encoding="utf-8")
            analysis_bytes = b"# Original analysis\r\nanalysis: {}\r\n"
            analysis_yaml.write_bytes(analysis_bytes)
            tuning_yaml.write_text(yaml.safe_dump({"tuning": {
                **tuning_config.model_dump(mode="json"), "output_dir": "runs",
            }}), encoding="utf-8")
            tuning_bytes = tuning_yaml.read_bytes()
            loaded_analysis = load_analysis_configuration(analysis_yaml)
            loaded_tuning = load_tuning_configuration(tuning_yaml)
            # Modifiche dopo il caricamento non devono cambiare gli snapshot.
            analysis_yaml.write_text("analysis: {dataset_mode: combined}\n", encoding="utf-8")
            tuning_yaml.write_text("tuning: {}\n", encoding="utf-8")
            extraction_yaml = root / "config_resolved.yaml"
            extraction_yaml.write_text("mirp: {voxel_spacing: [1.25]}\n", encoding="utf-8")
            extraction_bytes = extraction_yaml.read_bytes()
            sources = root / "config_sources"
            sources.mkdir()
            source_bytes = b"# Original extraction comment\r\nmirp: {}\r\n"
            (sources / "00_config_extractor.yaml").write_bytes(source_bytes)
            clinical_csv = root / "clinical.csv"
            clinical_csv.write_text("feature\n2\n", encoding="utf-8")
            clinical_yaml = clinical_csv.with_suffix(".config.yaml")
            clinical_yaml.write_bytes(b"# clinical config\n")

            run_dir = prepare_tuning_artifacts(
                analysis_config=loaded_analysis,
                tuning_config=loaded_tuning,
                input_csv=input_csv,
                clinical_input_csv=clinical_csv,
            )
            # Cambiamenti durante il tuning non devono alterare la provenienza.
            extraction_yaml.write_text("changed", encoding="utf-8")
            clinical_yaml.write_text("changed", encoding="utf-8")
            save_tuning_artifacts(result, run_dir=run_dir)

            self.assertTrue((run_dir / "summary.csv").is_file())
            self.assertTrue((run_dir / "best_parameters.json").is_file())
            self.assertTrue((run_dir / "source.json").is_file())
            self.assertEqual((run_dir / "config_analysis.yaml").read_bytes(), analysis_bytes)
            self.assertEqual((run_dir / "config_tuning.yaml").read_bytes(), tuning_bytes)
            self.assertEqual((run_dir / "config_extraction.yaml").read_bytes(), extraction_bytes)
            self.assertEqual((run_dir / "extraction_config/config_resolved.yaml").read_bytes(), extraction_bytes)
            self.assertEqual((run_dir / "extraction_config/config_sources/00_config_extractor.yaml").read_bytes(), source_bytes)
            self.assertEqual((run_dir / "config_clinical.yaml").read_bytes(), b"# clinical config\n")
            resolved = yaml.safe_load((run_dir / "config_analysis_resolved.yaml").read_text(encoding="utf-8"))
            self.assertEqual(resolved["analysis"]["dataset_mode"], "radiomic")
            self.assertIn("preprocessing", resolved["analysis"])
            self.assertEqual(load_tuning_configuration(run_dir / "config_tuning_resolved.yaml").output_dir, run_dir.parent)
            metadata = json.loads((run_dir / "source.json").read_text(encoding="utf-8"))
            self.assertTrue(metadata["extraction_config_available"])
            self.assertIn("extraction_config/config_sources/00_config_extractor.yaml", metadata["configuration_sha256"])
            self.assertTrue((run_dir / "models" / "logistic_l2.joblib").is_file())

            reporter = TerminalReporter(analysis_config)
            terminal = io.StringIO()
            reporter.console = Console(file=terminal, width=180, record=True)
            reporter.print_dataset(dataset)
            reporter.print_tuning_plan(tuning_config)
            reporter.print_tuning(result, tuning_config, run_dir)
            reporter.save_report(run_dir)
            report_text = (run_dir / "report.txt").read_text(encoding="utf-8")
            self.assertEqual(report_text, terminal.getvalue())
            self.assertIn("Inner accuracy", report_text)
            self.assertIn("Inner f1", report_text)
            self.assertIn("OUTER NESTED CV", report_text)
            self.assertIn("DATI CARICATI", report_text)
            self.assertTrue((run_dir / "report.html").is_file())

    def test_scoring_validation(self):
        models = {"logistic_l2": TuningModelConfig()}
        self.assertEqual(TuningConfig(models=models, scoring=["f1", "accuracy"]).refit, "f1")
        for options in (
            {"scoring": []}, {"scoring": ["f1", "f1"]},
            {"scoring": ["f1"], "refit": "accuracy"}, {"scoring": ["unknown"]},
        ):
            with self.subTest(options=options), self.assertRaises(ValidationError):
                TuningConfig(models=models, **options)

    def test_missing_extraction_config_is_recorded_and_run_names_are_unique(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_csv = root / "features.csv"
            input_csv.write_text("a\n1\n", encoding="utf-8")
            config = TuningConfig(output_dir=root / "runs", models={"logistic_l2": TuningModelConfig()})
            paths = [prepare_tuning_artifacts(
                analysis_config=AnalysisConfig(), tuning_config=config, input_csv=input_csv,
            ) for _ in range(2)]
            self.assertNotEqual(*paths)
            self.assertFalse(json.loads((paths[0] / "source.json").read_text())["extraction_config_available"])


if __name__ == "__main__":
    unittest.main()
