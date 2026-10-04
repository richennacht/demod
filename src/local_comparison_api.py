#!/usr/bin/env python3
"""Local, non-persistent API for comparing DSP and learned IQ estimates.

This deliberately binds to localhost by default.  It is a test harness for
authorised recordings, not a public RF-upload service.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import math
import wave
import sys
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent
WEB_ROOT = ROOT.parent / "web"
sys.path.insert(0, str(ROOT))

from analysis_pipeline import analyse as analyse_pipeline
from analyze_signal import decode_raw_iq, summarise_real
from capture_input import decode_capture, capture_provenance
from spectral_analysis import analyse_spectrum
from demodulation import SUPPORTED as SUPPORTED_DEMODULATIONS, demodulate
from dual_parameter_estimator import DualParameterEstimator
from generate_synthetic import load_recipes
from gnu_radio_adapter import flowgraph
from learned_analysis import analysis_region, carrier_offset, classify
from modulation_classifier import CentroidAMC
from provenance import input_provenance
from rate_estimation import estimate as estimate_rates, wav_metadata


MAX_INPUT_BYTES = 16 * 1024 * 1024
SUPPORTED_FORMATS = ("s16le", "s16be", "s8", "cu8", "f32le", "f32be")


class ComparisonService:
    """One in-memory learned baseline paired with deterministic DSP evidence."""

    def __init__(self, recipes_path: Path, examples: int = 4) -> None:
        self.recipes_path = recipes_path
        self.examples = examples
        recipes = load_recipes(recipes_path)
        self.estimator = DualParameterEstimator.train_from_recipes(recipes, examples=examples)
        self.classifier = CentroidAMC.train(recipes, examples=examples)

    def analyse_bytes(self, raw: bytes, iq_format: str, sample_rate_hz: float, denoise_profile: str = "raw", centre_frequency_hz: float | None = None, gain_db: float | None = None, metadata_source: str = "analyst_hypothesis", wav_role: str = 'unspecified', if_centre_hz: float = 0.) -> dict[str, Any]:
        if iq_format not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported iq format: {iq_format}")
        if len(raw) > MAX_INPUT_BYTES:
            raise ValueError(f"Test-harness input limit is {MAX_INPUT_BYTES} bytes; chunk larger captures in a controlled pipeline.")
        samples, sample_rate_hz, wav_info = decode_capture(raw, iq_format, sample_rate_hz, wav_role, if_centre_hz)
        if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
            raise ValueError('Sample rate must be finite and positive, or supplied by a WAV header.')
        if len(samples) < 8:
            raise ValueError("At least eight complete complex I/Q samples are required for the DSP-versus-model comparison.")
        input_record = capture_provenance(raw, iq_format, sample_rate_hz, centre_frequency_hz, gain_db, metadata_source, wav_info)
        if wav_info and wav_role == 'audio':
            summary = summarise_real([x.real for x in samples], int(sample_rate_hz))
            vis = analyse_spectrum(samples, sample_rate_hz)
            return {'kind': 'audio', 'run_id': str(uuid.uuid4()), 'input': input_record,
                    'audio_summary': summary, 'manual_dsp': summary,
                    'analysis': {'raw_branch': {'visualization': vis, 'features': summary}},
                    'modulation_classification': {'abstained': True, 'predicted_modulation': None, 'confidence': 0., 'reason': 'Already-demodulated audio; RF modulation is not recoverable from an audio container.'},
                    'provenance': {'raw_data_persisted': False, 'denoising_profile': 'raw', 'manual_branch': 'PCM audio levels, spectrum and time-frequency overview', 'automated_branch': 'RF models not applied to declared audio', 'training_examples': 0, 'model_training_recipes': 'none', 'scope': 'Audio overview only; no original RF or payload reconstruction.'}}
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
        import numpy as np
        segments = ((pipeline.get("raw_branch") or {}).get("visualization") or {}).get("segmentation", {}).get("segments")
        region, region_info = analysis_region(np.asarray(samples, dtype=np.complex128), segments)
        comparison = self.estimator.compare(comparison_samples, sample_rate_hz)
        learned_cfo = carrier_offset(region, sample_rate_hz)
        if learned_cfo is not None:
            legacy = comparison["carrier_offset_hz"]["learned"]
            dsp = comparison["carrier_offset_hz"]["dsp"]
            value = learned_cfo["carrier_offset_hz"]
            tolerance = max(250.0, 0.001 * sample_rate_hz)
            comparison["carrier_offset_hz"] = {"dsp": dsp, "learned": round(value, 6), "absolute_disagreement": round(abs(dsp - value), 6), "agreement": float(abs(dsp - value) <= tolerance), "tolerance_hz": tolerance, "learned_detail": learned_cfo, "legacy_tinymlp": round(legacy, 6)}
        legacy_classification = self.classifier.predict(comparison_samples)
        classification = classify(region, learned_cfo["normalised_cycles_per_sample"] if learned_cfo else None)
        if classification is not None:
            classification["legacy_centroid"] = {k: legacy_classification[k] for k in ("predicted_modulation", "confidence", "abstained")}
        report: dict[str, Any] = {
            "run_id": str(uuid.uuid4()),
            "input": input_record,
            "analysis": pipeline,
            "manual_dsp": pipeline["raw_branch"]["features"],
            "manual_parameter_estimation": pipeline["raw_branch"]["manual_parameters"],
            "automated_parameter_comparison": comparison,
            "modulation_classification": classification or legacy_classification,
            "learned_region": region_info,
            "rate_estimation": estimate_rates(region, sample_rate_hz) if len(region) >= 128 else None,
            "provenance": {
                "raw_data_persisted": False,
                "denoising_profile": denoise_profile,
                "manual_branch": "feature analysis, FFT/STFT, energy segmentation and named manual estimators",
                "automated_branch": ("SpecCFO carrier offset and DemodAMC classifier (data/models, evaluated in research/README.md); TinyMLP kept for DC only" if classification is not None else "7-to-12-to-3 TinyMLP and feature centroids; trained in memory from the checked-in synthetic recipe set"),
                "automated_targets": ["dc_i", "dc_q", "carrier_offset_hz", "modulation_classification"],
                "not_supported_by_model": ["sample_rate", "centre_frequency", "symbol_timing", "FEC", "interleaver"],
                "model_training_recipes": str(self.recipes_path).replace("\\\\", "/"),
                "training_examples": self.examples,
                "scope": "MVP comparison baseline; not a calibrated production model or blind decoder.",
            },
        }
        if wav_info and report['rate_estimation']:
            report['rate_estimation']['sample_rate_source'] = 'wav_header'
        return report

    def demodulate_bytes(self, raw: bytes, iq_format: str, sample_rate_hz: float, modulation: str, samples_per_symbol: int | None = None, timing_offset: int | None = 0, carrier_offset_hz: float | None = 0.0, parameter_source: str = "analyst_override", wav_role: str = 'unspecified', if_centre_hz: float = 0., fec_frame_offset_bits: int = 0, backend: str = 'auto', pulse: str = 'rect', rrc_rolloff: float = .35, phase_radians: float | None = 0., fsk_tones_hz=None) -> dict[str, Any]:
        samples, sample_rate_hz, wav_info = decode_capture(raw, iq_format, sample_rate_hz, wav_role, if_centre_hz)
        if wav_info and wav_role == 'audio':
            return {'status': 'abstained', 'reason': 'Declared audio is already demodulated. Use audio overview/playback; original RF modulation cannot be reconstructed.'}
        if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
            raise ValueError('Sample rate must be finite and positive.')
        report = None
        sources = {'modulation': 'analyst_override', 'samples_per_symbol': 'analyst_override', 'carrier_offset_hz': 'analyst_override', 'timing_offset_samples': 'analyst_fixed_offset'}
        if modulation == "auto" or samples_per_symbol is None or carrier_offset_hz is None:
            report = self.analyse_bytes(raw, iq_format, sample_rate_hz, wav_role=wav_role, if_centre_hz=if_centre_hz)
        if modulation == "auto":
            classification = report["modulation_classification"]
            if classification["abstained"] or classification["predicted_modulation"] not in SUPPORTED_DEMODULATIONS:
                return {"status": "abstained", "reason": "Automatic classifier did not select a supported receiver; provide X-DEmod-Modulation manually.", "classification": classification}
            modulation = classification["predicted_modulation"]
            parameter_source = "automatic_classifier"
            sources['modulation'] = 'DemodAMC'
        if samples_per_symbol is None:
            rate = (report.get('rate_estimation') or {}).get('learned')
            if modulation not in ('bpsk', 'qpsk','8psk','16qam','64qam') or not rate or rate['abstained'] or rate['samples_per_symbol'] is None:
                return {'status': 'abstained', 'reason': 'No supported confident SPS estimate; set samples per symbol manually (automatic SPS supports linear PSK/QAM only).', 'rate_estimation': report.get('rate_estimation')}
            samples_per_symbol = int(rate['samples_per_symbol'])
            sources['samples_per_symbol'] = 'symbol_rate.npz'
        if carrier_offset_hz is None:
            cfo = report['automated_parameter_comparison']['carrier_offset_hz'].get('learned_detail')
            if not cfo or cfo['confidence'] < .7 or abs(cfo['carrier_offset_hz']) > cfo['trained_range_hz'][1]:
                return {'status': 'abstained', 'reason': 'Carrier estimate unavailable or low confidence; set carrier offset manually.'}
            carrier_offset_hz = cfo['carrier_offset_hz']
            sources['carrier_offset_hz'] = cfo['model_file']
        region_info = {'source': 'whole_capture', 'sample_start': 0, 'sample_count': len(samples)}
        if report is not None and sources['samples_per_symbol'] != 'analyst_override':
            region_info = report['learned_region']
            start, count = region_info['sample_start'], region_info['sample_count']
            samples = samples[start:start + count]
        result = demodulate(samples, sample_rate_hz, modulation, samples_per_symbol, timing_offset, carrier_offset_hz, parameter_source, fec_frame_offset_bits, backend,pulse,rrc_rolloff,phase_radians,fsk_tones_hz)
        if report is not None:
            result['configuration']['parameter_source'] = 'model_guided_with_static_timing'
        sources['timing_offset_samples']=result['configuration'].get('timing_source','analyst_fixed_offset')
        sources['phase_radians']=result['configuration'].get('phase_source','legacy_no_phase_correction')
        sources['pulse']='analyst_hypothesis'
        result['configuration']['parameter_sources'] = sources
        result['input_region'] = region_info
        result['input'] = capture_provenance(raw, iq_format, sample_rate_hz, None, None, 'analyst_hypothesis', wav_info)
        if report is not None:
            result['input'] = report['input']
            result['automatic_evidence'] = {'classification': report['modulation_classification'], 'rates': report.get('rate_estimation'), 'carrier_offset': report['automated_parameter_comparison']['carrier_offset_hz']}
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
            self.send_header("Access-Control-Allow-Headers", "Content-Type, X-DEmod-IQ-Format, X-DEmod-Sample-Rate, X-DEmod-Denoise-Profile, X-DEmod-Centre-Frequency, X-DEmod-Gain, X-DEmod-Metadata-Source, X-DEmod-Modulation, X-DEmod-Samples-Per-Symbol, X-DEmod-Timing-Offset, X-DEmod-Carrier-Offset, X-DEmod-Known-Symbol-Rate, X-DEmod-Recording-Duration, X-DEmod-WAV-Role, X-DEmod-IF-Centre, X-DEmod-FEC-Frame-Offset, X-DEmod-Pulse, X-DEmod-RRC-Rolloff, X-DEmod-Phase-Radians, X-DEmod-FSK-Tones-Hz, X-DEmod-Receiver-Backend")
            self.end_headers()
            self.wfile.write(encoded)

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            # Lets Chromium's private/local-network preflight reach a loopback API from the hosted UI.
            self.send_header("Access-Control-Allow-Private-Network", "true")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, X-DEmod-IQ-Format, X-DEmod-Sample-Rate, X-DEmod-Denoise-Profile, X-DEmod-Centre-Frequency, X-DEmod-Gain, X-DEmod-Metadata-Source, X-DEmod-Modulation, X-DEmod-Samples-Per-Symbol, X-DEmod-Timing-Offset, X-DEmod-Carrier-Offset, X-DEmod-Known-Symbol-Rate, X-DEmod-Recording-Duration, X-DEmod-WAV-Role, X-DEmod-IF-Centre, X-DEmod-FEC-Frame-Offset, X-DEmod-Pulse, X-DEmod-RRC-Rolloff, X-DEmod-Phase-Radians, X-DEmod-FSK-Tones-Hz, X-DEmod-Receiver-Backend")
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
            if self.path not in ("/analyse", "/demodulate", "/rates", "/fec"):
                self._send(404, {"error": "Use POST /analyse or POST /demodulate."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_INPUT_BYTES:
                    raise ValueError(f"Content-Length must be between 1 and {MAX_INPUT_BYTES} bytes.")
                raw = self.rfile.read(length)
                optional_float = lambda name: float(self.headers[name]) if self.headers.get(name) else None
                if self.path == "/fec":
                    from fec_identification import analyse_bits
                    payload = json.loads(raw)
                    if not isinstance(payload, dict) or not isinstance(payload.get('bits'), str):
                        raise ValueError('FEC input must be a JSON object with a binary string named bits.')
                    report = analyse_bits(payload['bits'], payload.get('frame_offset_bits', 0))
                elif self.path == "/rates":
                    if raw[:4] == b'RIFF' and raw[8:12] == b'WAVE':
                        meta, values = wav_metadata(raw)
                        role = self.headers.get('X-DEmod-WAV-Role', 'unspecified')
                        report = {"metadata": meta, "rates": None, "note": "WAV header gives Fs. Stereo channel interpretation requires an analyst declaration."}
                        if role not in ('unspecified', 'audio'):
                            iq, fs, info = decode_capture(raw, 's16le', optional_float('X-DEmod-Sample-Rate') or 0., role, optional_float('X-DEmod-IF-Centre') or 0.)
                            report['rates'] = estimate_rates(iq, fs) if len(iq) >= 128 else None
                            report['metadata'] = info
                    else:
                        values = decode_raw_iq(raw, self.headers.get("X-DEmod-IQ-Format", "s16le"))
                        report = {"metadata": None, "rates": estimate_rates(values, optional_float("X-DEmod-Sample-Rate"), optional_float("X-DEmod-Known-Symbol-Rate"), optional_float("X-DEmod-Recording-Duration"))}
                elif self.path == "/analyse":
                    report = service.analyse_bytes(raw, self.headers.get("X-DEmod-IQ-Format", "s16le"), float(self.headers.get("X-DEmod-Sample-Rate", "0")), self.headers.get("X-DEmod-Denoise-Profile", "raw"), optional_float("X-DEmod-Centre-Frequency"), optional_float("X-DEmod-Gain"), self.headers.get("X-DEmod-Metadata-Source", "analyst_hypothesis"), self.headers.get('X-DEmod-WAV-Role', 'unspecified'), optional_float('X-DEmod-IF-Centre') or 0.)
                    duration = optional_float("X-DEmod-Recording-Duration")
                    if duration is not None:
                        if not math.isfinite(duration) or duration <= 0:
                            raise ValueError('Recording duration must be finite and positive.')
                        count = report['analysis']['raw_branch']['features']['sample_count']
                        fs = report['input']['capture']['sample_rate_hz']
                        if abs(count / duration - fs) > max(1e-6, fs * 1e-6):
                            raise ValueError('Supplied Fs conflicts with sample count / recording duration. Clear duration to use a manual override.')
                        report['input']['capture']['sample_rate_source'] = 'sample_count / analyst_recording_duration'
                        report['input']['capture']['recording_duration_seconds'] = duration
                        if report.get('rate_estimation'):
                            report['rate_estimation']['sample_rate_source'] = 'sample_count / analyst_recording_duration'
                else:
                    sps = int(self.headers['X-DEmod-Samples-Per-Symbol']) if self.headers.get('X-DEmod-Samples-Per-Symbol') else None
                    timing=self.headers.get('X-DEmod-Timing-Offset','0')
                    phase=self.headers.get('X-DEmod-Phase-Radians','0')
                    tones=self.headers.get('X-DEmod-FSK-Tones-Hz')
                    report = service.demodulate_bytes(raw, self.headers.get("X-DEmod-IQ-Format", "s16le"), float(self.headers.get("X-DEmod-Sample-Rate", "0")), self.headers.get("X-DEmod-Modulation", "auto"), sps, None if timing=='auto' else int(timing), optional_float("X-DEmod-Carrier-Offset"), wav_role=self.headers.get('X-DEmod-WAV-Role', 'unspecified'), if_centre_hz=optional_float('X-DEmod-IF-Centre') or 0., fec_frame_offset_bits=int(self.headers.get('X-DEmod-FEC-Frame-Offset', '0')),backend=self.headers.get('X-DEmod-Receiver-Backend','auto'),pulse=self.headers.get('X-DEmod-Pulse','rect'),rrc_rolloff=float(self.headers.get('X-DEmod-RRC-Rolloff','.35')),phase_radians=None if phase=='auto' else float(phase),fsk_tones_hz=[float(v) for v in tones.split(',')] if tones else None)
                self._send(200, report)
            except (ValueError, OverflowError, wave.Error, EOFError) as error:
                self._send(400, {"error": str(error)})

        def log_message(self, format: str, *args: Any) -> None:
            print("[comparison-api] " + format % args)
    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DEmod's local DSP-versus-ML comparison API.")
    parser.add_argument("--host", default="127.0.0.1", help="Keep the default loopback host for authorised local testing.")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--recipes", type=Path, default=Path("data/recipes/mvp-recipes.json"))
    parser.add_argument("--examples", type=int, default=4, help="Legacy TinyMLP/centroid fallback examples; learned NumPy models are loaded from data/models.")
    args = parser.parse_args()
    service = ComparisonService(args.recipes, args.examples)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(service))
    print(f"DEmod local comparison API listening at http://{args.host}:{args.port}")
    print(f"Analyst UI served at http://{args.host}:{args.port}/ui/")
    server.serve_forever()


if __name__ == "__main__":
    main()
