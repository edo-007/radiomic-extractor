from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml
from mirp.settings.generic import SettingsClass

from extractor.config_profiles import save_configuration_snapshot, yaml_value
from extractor.feature_preview_service import RadiomicFeaturePreviewService
from extractor.models import MirpConfig, load_configuration


class ExtractionProfilesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.main = self.root / "config.yaml"
        self.profile = self.root / "profiles" / "preprocessing.yaml"
        self.profile.parent.mkdir()

    def write(self, mirp=None, profile=None, **top):
        config = {"data": {"patients_folder": ".", "output_dir": "output"}, "mirp": mirp or {}, **top}
        if profile is not None:
            self.profile.write_text(yaml.safe_dump({"mirp": profile}), encoding="utf-8")
            config["profiles"] = {"preprocessing": "profiles/preprocessing.yaml"}
        self.main.write_text(yaml.safe_dump(config), encoding="utf-8")

    def test_relative_profiles_paths_and_standalone_config(self):
        self.write(mirp={"voxel_spacing": [1.25, 1.0, 2.0]}, profile={"roi_spline_order": 0})
        config = load_configuration(self.main)
        self.assertEqual(config.data.patients_folder, self.root)
        self.assertEqual(config.data.output_dir, self.root / "output")
        self.assertEqual(config.mirp.roi_spline_order, 0)
        self.assertEqual(len(config._source_files), 2)
        self.write(mirp={"bin_width": 10.0})
        self.assertEqual(load_configuration(self.main).mirp.bin_width, 10.0)

    def test_duplicates_are_rejected(self):
        for left, right in [({"bin_width": 10}, {"bin_width": 20})]:
            with self.subTest(left=left):
                self.write(mirp=left, profile=right)
                with self.assertRaisesRegex(ValueError, "duplicato"):
                    load_configuration(self.main)
        self.write(profile={})
        self.profile.write_text("mirp:\n  crop_distance: 1\n  crop_distance: 2\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicata"):
            load_configuration(self.main)

    def test_unknown_keys_are_rejected_at_all_levels(self):
        self.write(profile={"roi_spline_oder": 0})
        with self.assertRaisesRegex(ValueError, "roi_spline_oder"):
            load_configuration(self.main)
        for extra in [{"unknown": True}, {"data": {"patients_folder": ".", "typo": 1}},
                      {"dicom_metadata": {"typo": 1}}]:
            with self.subTest(extra=extra):
                self.write(**extra)
                with self.assertRaises(ValueError):
                    load_configuration(self.main)

    def test_missing_invalid_nested_and_repeated_profiles(self):
        self.write(profile={})
        self.profile.unlink()
        with self.assertRaises(FileNotFoundError):
            load_configuration(self.main)
        for content in ["[]", "null", "profiles: {}", "mirp: []", "data: {}", "mirp: {}\nprofiles: {}"]:
            with self.subTest(content=content):
                self.write(profile={})
                self.profile.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_configuration(self.main)
        self.write(profiles={"preprocesing": "missing.yaml"})
        with self.assertRaises(ValueError):
            load_configuration(self.main)
        self.write(profiles={"features": "config.yaml"})
        with self.assertRaisesRegex(ValueError, "circolare"):
            load_configuration(self.main)
        self.write(profile={})
        raw = yaml.safe_load(self.main.read_text(encoding="utf-8"))
        raw["profiles"]["features"] = raw["profiles"]["preprocessing"]
        self.main.write_text(yaml.safe_dump(raw), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "volte"):
            load_configuration(self.main)

    def test_new_settings_reach_mirp(self):
        self.write(mirp={
            "roi_spline_order": 0, "crop_around_roi": True, "crop_distance": 12.0,
            "no_approximation": True, "resegmentation_sigma": 3.0,
            "base_discretisation_method": "fixed_bin_number", "base_discretisation_n_bins": 8,
            "feature_families": ["glcm"], "glcm_distance": [1, 2],
            "glcm_spatial_method": "3d_average",
            "filter_kernels": ["mean"], "mean_filter_kernel_size": 3,
            "mean_filter_boundary_condition": "nearest",
        })
        settings = SettingsClass(**load_configuration(self.main).mirp.to_mirp_kwargs())
        self.assertEqual(settings.roi_interpolate.spline_order, 0)
        self.assertTrue(settings.perturbation.crop_around_roi)
        self.assertEqual(settings.perturbation.crop_distance, 12)
        self.assertTrue(settings.general.no_approximation)
        self.assertEqual(settings.roi_resegment.sigma, 3)
        self.assertEqual(settings.feature_extr.discretisation_n_bins, [8])
        self.assertEqual(settings.feature_extr.glcm_distance, [1, 2])
        self.assertEqual(settings.img_transform.mean_filter_boundary_condition, "nearest")

    def test_incompatible_active_settings_fail_before_extraction(self):
        cases = [
            {"by_slice": True, "feature_families": ["glcm"], "glcm_spatial_method": "3d_average"},
            {"mask_select_largest_slice": True},
            {"filter_kernels": ["laws"], "laws_kernel": ["l5e5"]},
            {"filter_kernels": ["separable_wavelet"], "separable_wavelet_families": ["coif4"], "separable_wavelet_set": ["hh"]},
            {"roi_spline_order": 7}, {"bin_width": []},
            {"base_discretisation_method": "fixed_bin_number"},
        ]
        for case in cases:
            with self.subTest(case=case):
                self.write(mirp=case)
                with self.assertRaises(ValueError):
                    load_configuration(self.main)

    def test_default_effective_settings_match_previous_wrapper(self):
        config = MirpConfig(bin_width=10, resegmentation_intensity_range=[0, float("nan")])
        before = SettingsClass(
            ibsi_compliant=True, by_slice=False, new_spacing=[1.0] * 3,
            base_feature_families=config.feature_families,
            base_discretisation_method="fixed_bin_size", base_discretisation_bin_width=10.0,
            texture_feature_pooling_method="average", resegmentation_intensity_range=[0.0, float("nan")],
        )
        after = SettingsClass(**config.to_mirp_kwargs())
        self.assertEqual(yaml.safe_dump(yaml_value(before)), yaml.safe_dump(yaml_value(after)))

    def test_snapshot_is_standalone_keeps_nan_and_original_bytes(self):
        self.write(mirp={"voxel_spacing": [1.25, 1, 2], "export_features": False},
                   profile={"resegmentation_intensity_range": [0, float("nan")]})
        config = load_configuration(self.main)
        original = self.profile.read_bytes()
        self.profile.write_text("changed after loading", encoding="utf-8")
        output = self.root / "snapshot"
        output.mkdir()
        active = config.mirp.model_copy(update={"export_features": True})
        path = save_configuration_snapshot(config, output, mirp_config=active)
        self.assertFalse((output / "config.yaml").exists())
        self.assertIn(".nan", path.read_text(encoding="utf-8"))
        self.assertEqual((output / "config_sources" / "01_preprocessing.yaml").read_bytes(), original)
        loaded = load_configuration(path)
        self.assertTrue(loaded.mirp.export_features)
        self.assertEqual(loaded.mirp.voxel_spacings(), config.mirp.voxel_spacings())
        self.assertEqual(loaded.data.output_dir, config.data.output_dir)
        self.assertEqual(len(loaded.provenance["effective_mirp_settings"]), 3)
        self.assertEqual(len(loaded.provenance["sources"]), 2)
        self.assertNotIn("profiles", yaml.safe_load(path.read_text(encoding="utf-8")))

    def test_alternative_discretisation_preview(self):
        config = MirpConfig(feature_families=["intensity_histogram"],
                            base_discretisation_method="fixed_bin_number", base_discretisation_n_bins=8)
        preview = RadiomicFeaturePreviewService(config).build()
        self.assertGreater(preview.radiomic_feature_count, 0)
        self.assertIn("fixed_bin_number", preview.groups[0].parameters[0])

    def test_removed_legacy_fields_are_rejected(self):
        for extra in [{"mirp": {"num_cpus": 1}}, {"n-test": 1}, {"patients_folder": "."}]:
            with self.subTest(extra=extra):
                self.write(**extra)
                with self.assertRaises(ValueError):
                    load_configuration(self.main)


if __name__ == "__main__":
    unittest.main()
