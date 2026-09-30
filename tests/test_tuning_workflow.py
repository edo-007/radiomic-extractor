from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

from analysis.main import run_analysis
from analysis.tuning import run_nested_tuning


class TuningWorkflowTest(unittest.TestCase):
    def test_snapshots_precede_fits_and_reports_survive_success_or_failure(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                extraction = root / "extraction-test"
                extraction.mkdir()
                csv = extraction / "features.csv"
                pd.DataFrame({
                    "nome_cognome": [f"p{i}" for i in range(16)],
                    "stato_microsatellitare": [0, 1] * 8,
                    "signal": [float(i % 2) + 0.01 * i for i in range(16)],
                    "noise": list(range(16)),
                }).to_csv(csv, sep=";", index=False)
                (extraction / "config_resolved.yaml").write_bytes(b"# extraction snapshot\nmirp: {}\n")
                analysis_yaml = root / "analysis.yaml"
                tuning_yaml = root / "tuning.yaml"
                analysis_yaml.write_text(yaml.safe_dump({"analysis": {
                    "experiments_dir": str(root), "positive_class": 1,
                }}), encoding="utf-8")
                tuning_yaml.write_text(yaml.safe_dump({"tuning": {
                    "output_dir": "runs", "strategy": "grid", "n_jobs": 1,
                    "scoring": ["f1", "accuracy"], "refit": "f1",
                    "outer_cv": {"n_splits": 2, "n_repeats": 1},
                    "inner_cv": {"n_splits": 2},
                    "models": {"logistic_l2": {"parameters": {"model__C": [1.0]}}},
                }}), encoding="utf-8")
                original_analysis = analysis_yaml.read_bytes()
                original_tuning = tuning_yaml.read_bytes()

                def execute(dataset, config, tuning_config, reporter):
                    run = next((root / "runs").iterdir())
                    self.assertEqual((run / "config_analysis.yaml").read_bytes(), original_analysis)
                    self.assertEqual((run / "config_tuning.yaml").read_bytes(), original_tuning)
                    self.assertTrue((run / "extraction_config/config_resolved.yaml").is_file())
                    # Simula l'utente che modifica i file mentre il tuning gira.
                    analysis_yaml.write_text("changed during fitting", encoding="utf-8")
                    tuning_yaml.write_text("changed during fitting", encoding="utf-8")
                    if fail:
                        raise RuntimeError("synthetic failure")
                    return run_nested_tuning(dataset, config, tuning_config)

                with (
                    patch("analysis.main.select_analysis_input", return_value=csv),
                    patch("analysis.main._run_tuning_with_progress", side_effect=execute),
                    redirect_stdout(io.StringIO()),
                ):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                            run_analysis("tune", analysis_yaml, tuning_yaml)
                    else:
                        self.assertEqual(run_analysis("tune", analysis_yaml, tuning_yaml), 0)

                run = next((root / "runs").iterdir())
                self.assertEqual((run / "config_analysis.yaml").read_bytes(), original_analysis)
                self.assertEqual((run / "config_tuning.yaml").read_bytes(), original_tuning)
                self.assertTrue((run / "report.html").is_file())
                report = (run / "report.txt").read_text(encoding="utf-8")
                self.assertIn("DATI CARICATI", report)
                self.assertIn("PIANO DEL TUNING", report)
                self.assertIn("synthetic failure" if fail else "Inner accuracy", report)
                self.assertEqual((run / "summary.csv").exists(), not fail)


if __name__ == "__main__":
    unittest.main()
