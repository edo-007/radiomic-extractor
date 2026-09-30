from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

from extractor.models import DataConfig, MirpConfig


class VoxelSpacingConfigTest(unittest.TestCase):
    def test_single_spacing_list(self) -> None:
        config = MirpConfig(voxel_spacing=[1.25])

        self.assertEqual(config.voxel_spacings(), [[1.25, 1.25, 1.25]])
        self.assertEqual(config.to_mirp_kwargs()["new_spacing"], [1.25, 1.25, 1.25])

    def test_multiple_and_compact_isotropic_spacings(self) -> None:
        compact = MirpConfig(voxel_spacing=[1.25, 2.0])

        expected = [[1.25, 1.25, 1.25], [2.0, 2.0, 2.0]]
        self.assertEqual(compact.voxel_spacings(), expected)

    def test_three_distinct_scalar_spacings(self) -> None:
        config = MirpConfig(voxel_spacing=[1.25, 1.0, 2.0])
        self.assertEqual(config.voxel_spacings(), [[1.25] * 3, [1.0] * 3, [2.0] * 3])

    def test_scalar_spacing_is_expanded_to_three_dimensions(self) -> None:
        config = MirpConfig(voxel_spacing=2.0)

        self.assertEqual(config.voxel_spacings(), [[2.0, 2.0, 2.0]])

    def test_by_slice_passes_only_in_plane_spacing_to_mirp(self) -> None:
        config = MirpConfig(
            voxel_spacing=[1.25],
            by_slice=True,
        )

        self.assertEqual(config.to_mirp_kwargs()["new_spacing"], [1.25, 1.25])

    def test_anisotropic_or_duplicate_spacings_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            MirpConfig(voxel_spacing=[[1.0, 1.0, 2.0]])
        with self.assertRaises(ValidationError):
            MirpConfig(voxel_spacing=[1.0, 1.0])
        with self.assertRaises(ValidationError):
            MirpConfig(voxel_spacing=[1.25, 1.25, 1.25])


class ExperimentOutputTest(unittest.TestCase):
    def test_experiment_folder_and_spacing_csv_names_are_unique(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = DataConfig(
                patients_folder=root,
                output_dir=root / "output",
            )
            fixed_time = datetime(2026, 9, 2, 14, 5, 6)

            first = config.create_experiment_dir(now=fixed_time)
            second = config.create_experiment_dir(now=fixed_time)
            csv_path = config.ensure_features_csv_path(
                output_dir=first,
                voxel_spacing=[1.25, 1.25, 1.25],
            )

            self.assertEqual(first.name, "extraction-2026-09-02_14-05-06")
            self.assertEqual(second.name, "extraction-2026-09-02_14-05-06_2")
            self.assertEqual(csv_path.parent, first)
            self.assertEqual(csv_path.name, "features_voxel-spacing-1p25mm.csv")


if __name__ == "__main__":
    unittest.main()
