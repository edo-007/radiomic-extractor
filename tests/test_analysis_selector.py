from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from analysis.selector import discover_experiments, list_feature_csvs


class AnalysisInputSelectorTest(unittest.TestCase):
    def test_experiments_are_sorted_newest_first_and_preview_is_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            older = root / "experiment-older"
            newer = root / "extraction-newer"
            empty = root / "experiment-empty"
            older.mkdir()
            newer.mkdir()
            empty.mkdir()
            tuning = root / "tuning-old"
            tuning.mkdir()
            (tuning / "summary.csv").write_text("model\na\n", encoding="utf-8")

            (older / "features_voxel-spacing-1mm.csv").write_text(
                "feature\n1\n", encoding="utf-8"
            )
            (newer / "features_voxel-spacing-2mm.csv").write_text(
                "feature\n2\n", encoding="utf-8"
            )
            (newer / "feature_preview.csv").write_text(
                "nome_colonna\nfeature\n", encoding="utf-8"
            )
            os.utime(older, (1000, 1000))
            os.utime(newer, (2000, 2000))

            experiments = discover_experiments(root)

            self.assertEqual(
                [experiment.path.name for experiment in experiments],
                ["extraction-newer", "experiment-older"],
            )
            self.assertEqual(
                [path.name for path in list_feature_csvs(newer)],
                ["features_voxel-spacing-2mm.csv"],
            )

    def test_missing_or_empty_experiments_directory_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with self.assertRaises(FileNotFoundError):
                discover_experiments(root / "missing")
            with self.assertRaises(FileNotFoundError):
                discover_experiments(root)


if __name__ == "__main__":
    unittest.main()
