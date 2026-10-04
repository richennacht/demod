"""Optional GNU Radio integration descriptor for interactive DEmod receivers."""

from __future__ import annotations

import importlib.util
from typing import Any


def availability() -> dict[str, Any]:
    try:
        from gnuradio import gr, digital
        installed = hasattr(gr,'top_block') and hasattr(digital,'generic_demod')
    except ImportError:
        installed = False
    return {"installed": installed, "backend": "gnuradio" if installed else "python_fallback", "install_note": None if installed else "GNU Radio runtime is absent. The PSK/QAM runner in gnuradio/run_receiver.py requires a separate GNU Radio 3.10 installation; descriptors do not execute."}


def flowgraph(modulation: str, sample_rate_hz: float, samples_per_symbol: int, carrier_offset_hz: float = 0.0) -> dict[str, Any]:
    common = [
        {"block": "File Source", "role": "read complex IQ"},
        {"block": "DC Blocker", "role": "optional reversible derived branch"},
        {"block": "Frequency Xlating FIR Filter", "role": "channel selection and carrier translation", "frequency_offset_hz": carrier_offset_hz},
    ]
    if modulation in ("bpsk", "qpsk", "8psk", "16qam", "64qam"):
        blocks = common + [
            {"block": "FLL Band-Edge", "role": "coarse carrier correction"},
            {"block": "PFB Clock Sync", "role": "matched filtering and timing recovery", "samples_per_symbol": samples_per_symbol},
            {"block": "Constellation Receiver", "role": "Costas-loop phase tracking and hard decisions", "constellation": modulation.upper()},
            {"block": "Unpack K Bits", "role": "bit stream"},
        ]
    elif modulation in ("2fsk","4fsk"):
        blocks = common + [
            {"block": "Quadrature Demod", "role": "phase-difference frequency discriminator"},
            {"block": "Symbol Sync", "role": "timing recovery", "samples_per_symbol": samples_per_symbol},
            {"block": "Binary Slicer" if modulation=='2fsk' else "Tone constellation decisions", "role": "hard FSK bit decisions"},
        ]
    else:
        raise ValueError("GNU Radio MVP flowgraphs support bpsk, qpsk, and 2fsk.")
    return {"engine": availability(), "sample_rate_hz": sample_rate_hz, "blocks": blocks, "note": "A graph descriptor only; execution requires installed GNU Radio and reviewed receiver parameters."}
