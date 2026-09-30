import importlib.util
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analysis_pipeline import analyse
from provenance import input_provenance, parse_sigmf_metadata


class AnalysisPipelineTests(unittest.TestCase):
    def test_pipeline_keeps_raw_and_derived_branches(self):
        samples = [complex(0.3 + math.cos(2 * math.pi * index / 16), -0.2 + math.sin(2 * math.pi * index / 16)) for index in range(512)]
        report = analyse(samples, 1_000_000, {"dc_offset": {"enabled": True}})
        self.assertTrue(report["denoising"]["raw_preserved"])
        self.assertIn("manual_parameters", report["raw_branch"])
        if report["raw_branch"]["visualization"] is not None:
            self.assertIn("constellation", report["raw_branch"]["visualization"])
        self.assertTrue(report["denoising"]["actions"][0]["applied"])

    def test_sigmf_and_input_provenance_keep_sources_explicit(self):
        sigmf = parse_sigmf_metadata('{"global":{"core:sample_rate":2400000,"core:datatype":"ci16_le"},"captures":[{"core:frequency":145000000}]}')
        provenance = input_provenance(b"abcd", "s16le", 2_400_000, 145_000_000, metadata_source="sigmf_metadata")
        self.assertEqual(sigmf["sample_rate_hz"], 2_400_000)
        self.assertEqual(provenance["capture"]["sample_rate_source"], "sigmf_metadata")
