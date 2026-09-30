from __future__ import annotations

import inspect
import os
import tempfile
import unittest
from pathlib import Path

import yaml

from cli import build_parser
from analysis.config import load_analysis_configuration
from analysis.clinical_config import load_clinical_configuration
from analysis.tuning_config import load_tuning_configuration
from extractor.models import CONFIG_FILE


class ConfigLayoutTest(unittest.TestCase):
    def test_all_operational_configs_are_under_config(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("extractor", "analysis", "clinical", "tuning"):
            self.assertTrue((root / "config" / f"config_{name}.yaml").is_file())
            self.assertFalse((root / f"config_{name}.yaml").exists())
        path = root / "config/config_extractor.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        for relative in config["profiles"].values():
            profile = (path.parent / relative).resolve()
            self.assertTrue(profile.is_relative_to(root / "config/extraction"))
            self.assertTrue(profile.is_file())

    def test_defaults_do_not_depend_on_working_directory(self):
        config_dir = Path(__file__).resolve().parents[1] / "config"
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                parser = build_parser()
                for command, name in [(["extract"], "extractor"), (["clinical"], "clinical"),
                                      (["analysis", "tune"], "analysis")]:
                    self.assertEqual(parser.parse_args(command).config, config_dir / f"config_{name}.yaml")
                self.assertEqual(parser.parse_args(["analysis", "tune"]).tuning_config,
                                 config_dir / "config_tuning.yaml")
                self.assertEqual(CONFIG_FILE, config_dir / "config_extractor.yaml")
                tuning = load_tuning_configuration()
                self.assertEqual(
                    tuning.output_dir,
                    config_dir.parent.parent / "radiomic-output" / "tuning",
                )
                for loader, name in [(load_analysis_configuration, "analysis"),
                                     (load_clinical_configuration, "clinical"),
                                     (load_tuning_configuration, "tuning")]:
                    self.assertEqual(inspect.signature(loader).parameters["config_path"].default,
                                     config_dir / f"config_{name}.yaml")
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
