#!/usr/bin/env python3
"""Local, non-persistent API for comparing DSP and learned IQ estimates.

This deliberately binds to localhost by default.  It is a test harness for
authorised recordings, not a public RF-upload service.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent
WEB_ROOT = ROOT.parent / "web"
sys.path.insert(0, str(ROOT))

from analysis_pipeline import analyse as analyse_pipeline
from analyze_signal import decode_raw_iq
from demodulation import SUPPORTED as SUPPORTED_DEMODULATIONS, demodulate
from dual_parameter_estimator import DualParameterEstimator
from generate_synthetic import load_recipes
from gnu_radio_adapter import flowgraph
from modulation_classifier import CentroidAMC
from provenance import input_provenance


MAX_INPUT_BYTES = 16 * 1024 * 1024
SUPPORTED_FORMATS = ("s16le", "s16be", "s8", "cu8", "f32le", "f32be")


class ComparisonService:
    """One in-memory learned baseline paired with deterministic DSP evidence."""

    def __init__(self, recipes_path: Path, examples: int = 96) -> None:
        self.recipes_path = recipes_path
        self.examples = examples
        recipes = load_recipes(recipes_path)
        self.estimator = DualParameterEstimator.train_from_recipes(recipes, examples=examples)
        self.classifier = CentroidAMC.train(recipes, examples=examples)

    def analyse_bytes(self, raw: bytes, iq_format: str, sample_rate_hz: float, denoise_profile: str = "raw", centre_frequency_hz: float | None = None, gain_db: float | None = None, metadata_source: str = "analyst_hypothesis") -> dict[str, Any]:
        if iq_format not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported iq format: {iq_format}")
        if sample_rate_hz <= 0:
            raise ValueError("X-DEmod-Sample-Rate must be a positive number.")
        if len(raw) > MAX_INPUT_BYTES:
            raise ValueError(f"Test-harness input limit is {MAX_INPUT_BYTES} bytes; chunk larger captures in a controlled pipeline.")
        samples = decode_raw_iq(raw, iq_format)
        if len(samples) < 8:
            raise ValueError("At least eight complete complex I/Q samples are required for the DSP-versus-model comparison.")
        profiles: dict[str, dict[str, dict[str, Any]]] = {
            "raw": {}, "dc_only": {"dc_offset": {"enabled": True}},
            "dc_and_impulse": {"dc_offset": {"enabled": True}, "impulse_blanking": {"enabled": True}},
        }
        if denoise_profile not in profiles:
            raise ValueError("X-DEmod-Denoise-Profile must be raw, dc_only, or dc_and_impulse.")
        pipeline = analyse_pipeline(samples, sample_rate_hz, profiles[denoise_profile])
        # The learned baseline is deliberately compared on the untouched input.
        # Derived denoising results are shown separately in the analysis graph.
        comparison_samples = samples
        report: dict[str, Any] = {
            "run_id": str(uuid.uuid4()),
            "input": input_provenance(raw, iq_format, sample_rate_hz, centre_frequency_hz, gain_db, metadata_source),
            "analysis": pipeline,
            "manual_dsp": pipeline["raw_branch"]["features"],
            "manual_parameter_estimation": pipeline["raw_branch"]["manual_parameters"],
            "automated_parameter_comparison": self.estimator.compare(comparison_samples, sample_rate_hz),
            "modulation_classification": self.classifier.predict(comparison_samples),
            "provenance": {
                "raw_data_persisted": False,
                "denoising_profile": denoise_profile,
                "manual_branch": "feature analysis, FFT/STFT, energy segmentation and named manual estimators",
                "automated_branch": "7-to-12-to-3 TinyMLP; trained in memory from the checked-in synthetic recipe set",
                "automated_targets": ["dc_i", "dc_q", "carrier_offset_hz", "modulation_classification"],
                "not_supported_by_model": ["sample_rate", "centre_frequency", "symbol_timing", "FEC", "interleaver"],
                "model_training_recipes": str(self.recipes_path).replace("\\\\", "/"),
                "training_examples": self.examples,
                "scope": "MVP comparison baseline; not a calibrated production model or blind decoder.",
            },
        }
        return report

    def demodulate_bytes(self, raw: bytes, iq_format: str, sample_rate_hz: float, modulation: str, samples_per_symbol: int, timing_offset: int = 0, carrier_offset_hz: float = 0.0, parameter_source: str = "analyst_override") -> dict[str, Any]:
        if modulation == "auto":
            classification = self.analyse_bytes(raw, iq_format, sample_rate_hz)["modulation_classification"]
            if classification["abstained"] or classification["predicted_modulation"] not in SUPPORTED_DEMODULATIONS:
                return {"status": "abstained", "reason": "Automatic classifier did not select a supported receiver; provide X-DEmod-Modulation manually.", "classification": classification}
            modulation = classification["predicted_modulation"]
            parameter_source = "automatic_classifier"
        samples = decode_raw_iq(raw, iq_format)
        result = demodulate(samples, sample_rate_hz, modulation, samples_per_symbol, timing_offset, carrier_offset_hz, parameter_source)
        result["gnu_radio_graph"] = flowgraph(modulation, sample_rate_hz, samples_per_symbol, carrier_offset_hz)
        return result


def make_handler(service: ComparisonService):
    class Handler(BaseHTTPRequestHandler):
        server_version = "DEmodComparison/0.1"

        def _send(self, status: int, payload: dict[str, Any]) -> None:
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, X-DEmod-IQ-Format, X-DEmod-Sample-Rate, X-DEmod-Denoise-Profile, X-DEmod-Centre-Frequency, X-DEmod-Gain, X-DEmod-Metadata-Source, X-DEmod-Modulation, X-DEmod-Samples-Per-Symbol, X-DEmod-Timing-Offset, X-DEmod-Carrier-Offset")
            self.end_headers()
            self.wfile.write(encoded)

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            # Lets Chromium's private/local-network preflight reach a loopback API from the hosted UI.
            self.send_header("Access-Control-Allow-Private-Network", "true")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, X-DEmod-IQ-Format, X-DEmod-Sample-Rate, X-DEmod-Denoise-Profile, X-DEmod-Centre-Frequency, X-DEmod-Gain, X-DEmod-Metadata-Source, X-DEmod-Modulation, X-DEmod-Samples-Per-Symbol, X-DEmod-Timing-Offset, X-DEmod-Carrier-Offset")
            self.end_headers()

        def _send_static(self, relative: str) -> None:
            target = (WEB_ROOT / (relative or "index.html")).resolve()
            if WEB_ROOT.resolve() not in target.parents or not target.is_file():
                self._send(404, {"error": "UI file not found."})
                return
            body = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/health":
                self._send(200, {"status": "ok", "data_persistence": "none", "max_input_bytes": MAX_INPUT_BYTES, "supported_formats": list(SUPPORTED_FORMATS), "supported_demodulations": list(SUPPORTED_DEMODULATIONS)})
            elif path in ("/", "/ui"):
                self.send_response(302)
                self.send_header("Location", "/ui/")
                self.end_headers()
            elif path.startswith("/ui/"):
                self._send_static(path[len("/ui/"):])
            else:
                self._send(404, {"error": "Use GET /health, GET /ui/, POST /analyse or POST /demodulate."})

        def do_POST(self) -> None:
            if self.path not in ("/analyse", "/demodulate"):
                self._send(404, {"error": "Use POST /analyse or POST /demodulate."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_INPUT_BYTES:
                    raise ValueError(f"Content-Length must be between 1 and {MAX_INPUT_BYTES} bytes.")
                raw = self.rfile.read(length)
                optional_float = lambda name: float(self.headers[name]) if self.headers.get(name) else None
                if self.path == "/analyse":
                    report = service.analyse_bytes(raw, self.headers.get("X-DEmod-IQ-Format", "s16le"), float(self.headers.get("X-DEmod-Sample-Rate", "0")), self.headers.get("X-DEmod-Denoise-Profile", "raw"), optional_float("X-DEmod-Centre-Frequency"), optional_float("X-DEmod-Gain"), self.headers.get("X-DEmod-Metadata-Source", "analyst_hypothesis"))
                else:
                    if not self.headers.get("X-DEmod-Samples-Per-Symbol"):
                        raise ValueError("X-DEmod-Samples-Per-Symbol is required for MVP demodulation.")
                    report = service.demodulate_bytes(raw, self.headers.get("X-DEmod-IQ-Format", "s16le"), float(self.headers.get("X-DEmod-Sample-Rate", "0")), self.headers.get("X-DEmod-Modulation", "auto"), int(self.headers["X-DEmod-Samples-Per-Symbol"]), int(self.headers.get("X-DEmod-Timing-Offset", "0")), optional_float("X-DEmod-Carrier-Offset") or 0.0)
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
    print(f"Analyst UI served at http://{args.host}:{args.port}/ui/")
    server.serve_forever()


if __name__ == "__main__":
    main()
