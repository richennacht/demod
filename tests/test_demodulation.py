import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from demodulation import demodulate
from gnu_radio_adapter import flowgraph


class DemodulationTests(unittest.TestCase):
    def test_bpsk_hard_decisions_for_manual_sps(self):
        samples = [value for bit in (0, 1, 0, 1) for value in ([complex(-1 if bit == 0 else 1, 0)] * 4)]
        report = demodulate(samples, 1_000_000, "bpsk", 4)
        self.assertEqual(report["status"], "hard_decisions_available")
        self.assertEqual(report["bits_preview"], "0101")
        self.assertEqual(report["symbol_count"], 4)

    def test_qpsk_and_fsk_are_bounded_supported_paths(self):
        qpsk = [value for symbol in (complex(-1, -1), complex(-1, 1), complex(1, -1), complex(1, 1)) for value in [symbol] * 2]
        report = demodulate(qpsk, 1_000_000, "qpsk", 2)
        self.assertEqual(report["bit_count"], 8)
        graph = flowgraph("2fsk", 1_000_000, 4)
        self.assertEqual(graph["blocks"][-1]["block"], "Binary Slicer")
