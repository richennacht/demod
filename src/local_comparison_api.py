#!/usr/bin/env python3
"""Local, non-persistent API for comparing DSP and learned IQ estimates.

This deliberately binds to localhost by default.  It is a test harness for
authorised recordings, not a public RF-upload service.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from analyze_signal import decode_raw_iq
from dual_parameter_estimator import DualParameterEstimator
from feature_analysis import analyse_iq
from generate_synthetic import load_recipes

try:
    from spectral_analysis import analyse_spectrum
except ModuleNotFoundError:
    analyse_spectrum = None


MAX_INPUT_BYTES = 16 * 1024 * 1024
SUPPORTED_FORMATS = ("s16le", "s16be", "s8", "cu8", "f32le", "f32be")


class ComparisonService:
    """One in-memory learned baseline paired with deterministic DSP evidence."""

    def __init__(self, recipes_path: Path, examples: int = 96) -> None:
        self.recipes_path = recipes_path
        self.examples = examples
        self.estimator = DualParameterEstimator.train_from_recipes(load_recipes(recipes_path), examples=examples)

    def analyse_bytes(self, raw: bytes, iq_format: str, sample_rate_hz: float) -> dict[str, Any]:
        if iq_format not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported iq format: {iq_format}")
        if sample_rate_hz <= 0:
            raise ValueError("X-DEmod-Sample-Rate must be a positive number.")
        if len(raw) > MAX_INPUT_BYTES:
            raise ValueError(f"Test-harness input limit is {MAX_INPUT_BYTES} bytes; chunk larger captures in a controlled pipeline.")
        samples = decode_raw_iq(raw, iq_format)
        if len(samples) < 8:
            raise ValueError("At least eight complete complex I/Q samples are required for the DSP-versus-model comparison.")
        manual = analyse_iq(samples, sample_rate_hz)
        report: dict[str, Any] = {
            "run_id": str(uuid.uuid4()),
            "input": {
                "byte_count": len(raw), "iq_format": iq_format,
                "sample_rate_hz": sample_rate_hz, "sample_rate_source": "analyst_supplied_hypothesis",
            },
            "manual_dsp": manual,
            "automated_parameter_comparison": self.estimator.compare(samples, sample_rate_hz),
            "provenance": {
                "raw_data_persisted": False,
                "denoising_applied": False,
                "manual_branch": "feature_analysis.analyse_iq deterministic statistics and DFT preview",
                "automated_branch": "7-to-12-to-3 TinyMLP; trained in memory from the checked-in synthetic recipe set",
                "automated_targets": ["dc_i", "dc_q", "carrier_offset_hz"],
                "not_supported_by_model": ["sample_rate", "centre_frequency", "symbol_timing", "modulation", "FEC", "interleaver"],
                "model_training_recipes": str(self.recipes_path).replace("\\\\", "/"),
                "training_examples": self.examples,
                "scope": "MVP comparison baseline; not a calibrated production model or blind decoder.",
            },
        }
        report["visualization"] = analyse_spectrum(samples, sample_rate_hz) if analyse_spectrum else {
            "available": False, "reason": "Install requirements-dsp.txt for NumPy FFT/STFT visualisation."
        }
        return report


def make_handler(service: ComparisonService):
    class Handler(BaseHTTPRequestHandler):
        server_version = "DEmodComparison/0.1"

        def _send(self, status: int, payload: dict[str, Any]) -> None:
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, X-DEmod-IQ-Format, X-DEmod-Sample-Rate")
            self.end_headers()
            self.wfile.write(encoded)

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, X-DEmod-IQ-Format, X-DEmod-Sample-Rate")
            self.end_headers()

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(200, {"status": "ok", "data_persistence": "none", "max_input_bytes": MAX_INPUT_BYTES})
            else:
                self._send(404, {"error": "Use GET /health or POST /analyse."})

        def do_POST(self) -> None:
            if self.path != "/analyse":
                self._send(404, {"error": "Use POST /analyse."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_INPUT_BYTES:
                    raise ValueError(f"Content-Length must be between 1 and {MAX_INPUT_BYTES} bytes.")
                raw = self.rfile.read(length)
                report = service.analyse_bytes(raw, self.headers.get("X-DEmod-IQ-Format", "s16le"), float(self.headers.get("X-DEmod-Sample-Rate", "0")))
                self._send(200, report)
            except (ValueError, OverflowError) as error:
                self._send(400, {"error": str(error)})

        def log_message(self, format: str, *args: Any) -> None:
            print("[comparison-api] " + format % args)
    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DEmod's local DSP-versus-ML comparison API.")
    parser.add_argument("--host", default="127.0.0.1", help="Keep the default loopback host for authorised local testing.")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--recipes", type=Path, default=Path("data/recipes/mvp-recipes.json"))
    parser.add_argument("--examples", type=int, default=96)
    args = parser.parse_args()
    service = ComparisonService(args.recipes, args.examples)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(service))
    print(f"DEmod local comparison API listening at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
