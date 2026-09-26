#!/usr/bin/env python3
"""Transparent baseline analysis for SIH26147 WAV and raw IQ recordings."""

from __future__ import annotations

import argparse
import json
import math
import struct
import wave
from pathlib import Path
from typing import Iterable


def rms(samples: Iterable[float]) -> float:
    values = list(samples)
    return math.sqrt(sum(value * value for value in values) / len(values)) if values else 0.0


def summarise_real(samples: list[float], sample_rate: int) -> dict[str, float | int]:
    if not samples:
        raise ValueError("The recording contains no samples.")
    average = sum(samples) / len(samples)
    centered = [sample - average for sample in samples]
    level = rms(centered)
    peak = max(abs(value) for value in centered)
    zero_crossings = sum(
        1 for left, right in zip(centered, centered[1:]) if (left < 0 <= right) or (right < 0 <= left)
    )
    dominant_frequency = (zero_crossings * sample_rate) / (2 * max(1, len(centered) - 1))
    return {
        "sample_count": len(samples),
        "duration_seconds": round(len(samples) / sample_rate, 6),
        "sample_rate_hz": sample_rate,
        "dc_offset": round(average, 8),
        "rms_level": round(level, 8),
        "peak_level": round(peak, 8),
        "zero_crossing_frequency_hz": round(dominant_frequency, 3),
    }


def read_wav(path: Path) -> tuple[list[float], int, dict[str, int]]:
    with wave.open(str(path), "rb") as source:
        channels, width, sample_rate, frames = source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getnframes()
        if width not in (1, 2, 4):
            raise ValueError(f"Unsupported WAV sample width: {width} bytes.")
        raw = source.readframes(frames)
    formats = {1: "B", 2: "h", 4: "i"}
    unpacked = struct.unpack("<" + formats[width] * (len(raw) // width), raw)
    scale = {1: 128.0, 2: 32768.0, 4: 2147483648.0}[width]
    values = [((value - 128) if width == 1 else value) / scale for value in unpacked]
    mono = [sum(values[index:index + channels]) / channels for index in range(0, len(values), channels)]
    return mono, sample_rate, {"channels": channels, "sample_width_bytes": width}


def read_s16le_iq(path: Path) -> list[complex]:
    raw = path.read_bytes()
    if len(raw) % 4:
        raise ValueError("s16le IQ files must contain complete interleaved I/Q pairs.")
    integers = struct.unpack("<" + "h" * (len(raw) // 2), raw)
    return [complex(integers[index] / 32768.0, integers[index + 1] / 32768.0) for index in range(0, len(integers), 2)]


def analyse(path: Path, sample_rate: int | None, iq_format: str) -> dict:
    suffix = path.suffix.lower()
    if suffix == ".wav":
        samples, resolved_rate, source = read_wav(path)
        return {"input": {"path": str(path), "format": "wav", **source}, "measurements": summarise_real(samples, resolved_rate)}
    if suffix == ".iq":
        if sample_rate is None:
            raise ValueError("--sample-rate is required for raw IQ input.")
        if iq_format != "s16le":
            raise ValueError("Only interleaved signed 16-bit little-endian IQ is supported in this MVP.")
        samples = read_s16le_iq(path)
        magnitude = [abs(value) for value in samples]
        phase = [math.atan2(value.imag, value.real) for value in samples]
        return {
            "input": {"path": str(path), "format": "iq", "iq_format": iq_format},
            "measurements": {
                **summarise_real(magnitude, sample_rate),
                "mean_phase_radians": round(sum(phase) / len(phase), 8),
            },
        }
    raise ValueError("Supported input extensions are .wav and .iq.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a transparent baseline analysis report for WAV or raw IQ data.")
    parser.add_argument("input", type=Path, help="Path to a .wav or .iq recording")
    parser.add_argument("--sample-rate", type=int, help="Required for raw IQ input")
    parser.add_argument("--iq-format", default="s16le", choices=["s16le"], help="Raw IQ sample format")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    arguments = parser.parse_args()
    report = analyse(arguments.input, arguments.sample_rate, arguments.iq_format)
    report["analysis_version"] = "0.1.0"
    encoded = json.dumps(report, indent=2)
    if arguments.output:
        arguments.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
