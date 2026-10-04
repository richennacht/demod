"""Bounded, analyst-configured MVP receivers for baseband BPSK/QPSK/2-FSK.

These receivers deliberately require a declared samples-per-symbol value. They
provide inspectable hard decisions for controlled test signals, not blind or
protocol-level decoding.
"""

from __future__ import annotations

import math
from typing import Any


SUPPORTED = ("bpsk", "qpsk", "2fsk")


def _correct(samples: list[complex], carrier_offset_hz: float, sample_rate_hz: float) -> list[complex]:
    dc = sum(samples) / len(samples)
    return [(sample - dc) * complex(math.cos(-2 * math.pi * carrier_offset_hz * index / sample_rate_hz), math.sin(-2 * math.pi * carrier_offset_hz * index / sample_rate_hz)) for index, sample in enumerate(samples)]


def _integrate_and_dump(samples: list[complex], samples_per_symbol: int, timing_offset: int) -> list[complex]:
    if samples_per_symbol < 1:
        raise ValueError("samples_per_symbol must be at least 1")
    if timing_offset < 0 or timing_offset >= samples_per_symbol:
        raise ValueError("timing_offset must be in [0, samples_per_symbol)")
    result = []
    for start in range(timing_offset, len(samples) - samples_per_symbol + 1, samples_per_symbol):
        block = samples[start:start + samples_per_symbol]
        result.append(sum(block) / len(block))
    return result


def _pack(bits: list[int]) -> list[int]:
    return [sum(bits[index + shift] << (7 - shift) for shift in range(min(8, len(bits) - index))) for index in range(0, len(bits), 8)]


def _evm(symbols: list[complex], points: list[complex]) -> float:
    if not symbols:
        return float("inf")
    scale = math.sqrt(sum(abs(value) ** 2 for value in symbols) / len(symbols)) or 1.0
    normalized = [value / scale for value in symbols]
    return math.sqrt(sum(min(abs(value - point) ** 2 for point in points) for value in normalized) / len(normalized))


def demodulate(samples: list[complex], sample_rate_hz: float, modulation: str, samples_per_symbol: int, timing_offset: int = 0, carrier_offset_hz: float = 0.0, source: str = "analyst_override", fec_frame_offset_bits: int = 0) -> dict[str, Any]:
    if modulation not in SUPPORTED:
        raise ValueError(f"Supported MVP modulations are {', '.join(SUPPORTED)}.")
    if len(samples) < samples_per_symbol * 2:
        raise ValueError("Insufficient samples for two symbols at the supplied samples_per_symbol.")
    corrected = _correct(samples, carrier_offset_hz, sample_rate_hz)
    symbol_values = _integrate_and_dump(corrected, samples_per_symbol, timing_offset)
    if modulation == "bpsk":
        bits = [int(value.real >= 0) for value in symbol_values]
        evm = _evm(symbol_values, [-1 + 0j, 1 + 0j])
        decisions = [{"i": round(value.real, 6), "q": round(value.imag, 6), "symbol": bit} for value, bit in zip(symbol_values[:256], bits[:256])]
    elif modulation == "qpsk":
        # Fixed quadrant mapping: 00=(-,-), 01=(-,+), 10=(+,-), 11=(+,+).
        symbols = [(int(value.real >= 0) << 1) | int(value.imag >= 0) for value in symbol_values]
        bits = [bit for symbol in symbols for bit in ((symbol >> 1) & 1, symbol & 1)]
        root_half = 1 / math.sqrt(2)
        evm = _evm(symbol_values, [complex(i * root_half, q * root_half) for i in (-1, 1) for q in (-1, 1)])
        decisions = [{"i": round(value.real, 6), "q": round(value.imag, 6), "symbol": symbol} for value, symbol in zip(symbol_values[:256], symbols[:256])]
    else:
        phases = [math.atan2(value.imag, value.real) for value in corrected]
        discriminator = [((right - left + math.pi) % (2 * math.pi)) - math.pi for left, right in zip(phases, phases[1:])]
        symbol_values = [complex(sum(discriminator[start:start + samples_per_symbol]) / samples_per_symbol, 0) for start in range(timing_offset, len(discriminator) - samples_per_symbol + 1, samples_per_symbol)]
        bits = [int(value.real >= 0) for value in symbol_values]
        evm = 0.0  # FSK uses discriminator separation, not constellation EVM.
        decisions = [{"frequency_step_rad": round(value.real, 6), "symbol": bit} for value, bit in zip(symbol_values[:256], bits[:256])]
    # Use the full bounded hard-bit region, never the 512-bit presentation preview.
    if type(fec_frame_offset_bits) is not int or fec_frame_offset_bits < 0 or fec_frame_offset_bits >= len(bits):
        raise ValueError('FEC frame offset must be an integer within receiver bits.')
    if len(bits) >= 1680:
        from fec_identification import analyse_bits
        fec = analyse_bits(bits, fec_frame_offset_bits)
    else:
        fec = {"status": "abstained", "reason": "Need at least 1680 bits for experimental FEC identification.",
               "bit_count_supplied": len(bits), "scope": "No code or interleaver inferred from this short preview."}
    return {
        "status": "hard_decisions_available", "modulation": modulation,
        "configuration": {"samples_per_symbol": samples_per_symbol, "timing_offset_samples": timing_offset, "carrier_offset_hz": carrier_offset_hz, "parameter_source": source},
        "symbol_count": len(symbol_values), "bit_count": len(bits), "bits_preview": "".join(map(str, bits[:512]),), "packed_bytes_preview": _pack(bits[:512]),
        "quality": {"evm_rms": None if modulation == "2fsk" else round(evm, 6), "meaning": "constellation EVM after fixed integrate-and-dump; it is not BER without known reference bits."},
        "decisions_preview": decisions,
        "fec_identification": fec,
        "limitations": ["No framing, CRC, FEC decoding, decryption, differential decoding, equalisation, or protocol recognition is performed.", "Timing is a fixed integrate-and-dump override, not a recovered clock.", "Hard bits are candidate decisions, not validated payload data."],
    }
