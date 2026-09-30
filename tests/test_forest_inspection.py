from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from analysis.config import AnalysisConfig, FeatureSelectionConfig, PreprocessingConfig
from analysis.forest_inspection import discover_forest_runs, feature_coordinates, inspect_forest
from analysis.preprocessing import build_analysis_pipeline
from analysis.tuning import _build_search
from analysis.tuning_config import TuningConfig, TuningModelConfig
from cli import build_parser


class ForestInspectionTest(unittest.TestCase):
    def test_new_forests_never_scale_other_estimators_still_do(self):
        for scaling in ("standard", "robust", "none"):
            config = PreprocessingConfig(scaling=scaling)
            for estimator in (RandomForestClassifier(), LogisticRegression(), SVC()):
                with self.subTest(scaling=scaling, estimator=type(estimator).__name__):
                    pipeline = build_analysis_pipeline(estimator, config, random_state=42)
                    expected = scaling != "none" and not isinstance(estimator, RandomForestClassifier)
                    self.assertEqual("scaler" in pipeline.named_steps, expected)
                    for name in ("imputer", "constant_filter", "correlation_filter", "feature_selection"):
                        self.assertIn(name, pipeline.named_steps)
        search = _build_search(
            model_name="random_forest", parameter_space={"model__n_estimators": [2]},
            analysis_config=AnalysisConfig(),
            tuning_config=TuningConfig(models={"random_forest": TuningModelConfig()}),
            random_state=42, memory=None,
        )
        self.assertNotIn("scaler", search.estimator.named_steps)

    def test_inspection_of_new_and_legacy_forests_preserves_model_and_thresholds(self):
        rng = np.random.default_rng(42)
        signal = rng.normal(50, 7, size=40)
        X = pd.DataFrame({
            "constant": np.ones(40), "signal <original>": signal,
            "correlated": signal * 2, "noise": rng.normal(100, 20, size=40),
        })
        y = (signal > 50).astype(int)
        for scaling in ("none", "standard", "robust"):
            with self.subTest(scaling=scaling), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                model_dir = root / "models"
                model_dir.mkdir()
                config = PreprocessingConfig(scaling=scaling, feature_selection=FeatureSelectionConfig(k=1))
                # Simula i vecchi modelli: scaler presente prima della foresta.
                pipeline = build_analysis_pipeline(LogisticRegression(), config, random_state=42)
                pipeline.set_params(model=RandomForestClassifier(n_estimators=3, max_depth=4, random_state=42))
                pipeline.fit(X, y)
                names, scale, offset = feature_coordinates(pipeline)
                self.assertEqual(names, ["signal <original>"])
                np.testing.assert_allclose((X[names].to_numpy() - offset) / scale, pipeline[:-1].transform(X))
                model_path = model_dir / "random_forest.joblib"
                joblib.dump(pipeline, model_path)
                original_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
                report = inspect_forest(root, max_depth=0)
                self.assertEqual(original_hash, hashlib.sha256(model_path.read_bytes()).hexdigest())
                summary = json.loads((report / "summary.json").read_text())
                forest = pipeline.named_steps["model"]
                self.assertEqual(summary["trees"], 3)
                self.assertEqual(summary["depth_max"], max(t.get_depth() for t in forest.estimators_))
                stats = pd.read_csv(report / "trees.csv", sep=";")
                self.assertEqual(stats.depth.tolist(), [t.get_depth() for t in forest.estimators_])
                nodes = pd.read_csv(report / "nodes.csv", sep=";")
                self.assertEqual(len(nodes), sum(t.tree_.node_count for t in forest.estimators_))
                splits = nodes.loc[nodes.feature.notna()]
                np.testing.assert_allclose(splits.threshold_original, splits.threshold_model * scale[0] + offset[0])
                svg = (report / "trees/tree-0001.svg").read_text(encoding="utf-8")
                self.assertIn("sottoalbero non visualizzato", svg)
                self.assertIn("signal &lt;original&gt;", svg)
                self.assertNotIn("signal <original>", svg)
                page = (report / "index.html").read_text(encoding="utf-8")
                self.assertIn("tree-0003.svg", page)
                self.assertIn("Scegli albero", page)
                self.assertNotEqual(report, inspect_forest(root, max_depth=2))

    def test_discovery_order_and_invalid_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = [root / "tuning-old", root / "tuning-new"]
            for i, run in enumerate(runs, start=1):
                (run / "models").mkdir(parents=True)
                model = run / "models/random_forest.joblib"
                joblib.dump(LogisticRegression(), model)
                os.utime(model, (i * 1000, i * 1000))
            self.assertEqual(discover_forest_runs(root), list(reversed(runs)))
            with self.assertRaises(ValueError):
                inspect_forest(runs[0])
            with self.assertRaises(ValueError):
                inspect_forest(runs[0], max_depth=-1)
            with self.assertRaises(FileNotFoundError):
                inspect_forest(root)
        args = build_parser().parse_args(["analysis", "inspect-rf", "--run-dir", "example", "--max-depth", "6"])
        self.assertEqual(args.run_dir, Path("example"))
        self.assertEqual(args.max_depth, 6)


if __name__ == "__main__":
    unittest.main()
