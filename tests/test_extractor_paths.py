from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import call, patch

from extractor.dicom_metadata_service import filesystem_path
from extractor.models import MirpConfig, MirpExtractor, Paziente


class MirpPathTest(unittest.TestCase):
    def test_all_paths_are_converted_before_calling_mirp(self):
        patient = Paziente.model_construct(
            nome="SYNTHETIC", path_ct=Path("input/ct"), path_rt=Path("input/rt"),
        )
        output = Path("output")
        converted = [Path("converted/ct"), Path("converted/rt"), Path("converted/output")]
        config = MirpConfig(voxel_spacing=[1.25], roi_names=["GTV"], write_features=True)
        with patch("extractor.models._filesystem_path", side_effect=converted) as convert:
            with patch("mirp.extract_features", return_value=[]) as extract:
                result = MirpExtractor(config=config).extract(patient, output_dir=output)

        self.assertEqual(convert.call_args_list, [call(patient.path_ct), call(patient.path_rt), call(output)])
        kwargs = extract.call_args.kwargs
        self.assertEqual(kwargs["image"], str(converted[0]))
        self.assertEqual(kwargs["mask"], str(converted[1]))
        self.assertEqual(kwargs["write_dir"], str(converted[2]))
        self.assertEqual(kwargs["new_spacing"], [1.25] * 3)
        self.assertEqual(kwargs["roi_name"], ["GTV"])
        self.assertTrue(kwargs["write_features"])
        self.assertTrue(kwargs["export_features"])
        self.assertEqual(result, [])

    def test_no_output_directory_stays_none(self):
        patient = Paziente.model_construct(nome="SYNTHETIC", path_ct=Path("ct"), path_rt=Path("rt"))
        with patch("mirp.extract_features", return_value=[]) as extract:
            MirpExtractor().extract(patient)
        self.assertIsNone(extract.call_args.kwargs["write_dir"])
        self.assertFalse(extract.call_args.kwargs["write_features"])

    @unittest.skipUnless(os.name == "nt", "Percorsi estesi specifici di Windows")
    def test_long_windows_paths_reach_mirp_with_extended_prefix(self):
        root = Path("C:/synthetic") / ("a" * 120) / ("b" * 120)
        patient = Paziente.model_construct(nome="SYNTHETIC", path_ct=root / "ct", path_rt=root / "rt")
        self.assertGreaterEqual(len(str(patient.path_ct / "slice.dcm")), 260)
        with patch("mirp.extract_features", return_value=[]) as extract:
            MirpExtractor().extract(patient, output_dir=root / "output")
        for key in ("image", "mask", "write_dir"):
            path = extract.call_args.kwargs[key]
            self.assertTrue(path.startswith("\\\\?\\"), key)
            self.assertEqual(str(filesystem_path(Path(path))), path)


if __name__ == "__main__":
    unittest.main()
