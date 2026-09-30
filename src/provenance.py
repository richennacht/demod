"""Capture metadata and immutable input provenance for DEmod analysis runs."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def input_provenance(raw: bytes, iq_format: str, sample_rate_hz: float, centre_frequency_hz: float | None = None, gain_db: float | None = None, metadata_source: str = "analyst_hypothesis") -> dict[str, Any]:
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    return {
        "sha256": hashlib.sha256(raw).hexdigest(), "byte_count": len(raw),
        "representation": {"iq_format": iq_format, "iq_format_source": metadata_source},
        "capture": {
            "sample_rate_hz": sample_rate_hz, "sample_rate_source": metadata_source,
            "centre_frequency_hz": centre_frequency_hz,
            "centre_frequency_source": metadata_source if centre_frequency_hz is not None else "unavailable",
            "gain_db": gain_db, "gain_source": metadata_source if gain_db is not None else "unavailable",
        },
        "raw_data_persisted": False,
    }


def parse_sigmf_metadata(text: str) -> dict[str, Any]:
    """Read the portable capture fields used by SigMF sidecars.

    Unknown fields are retained only as key names: DEmod does not treat arbitrary
    metadata as verified truth.
    """
    document = json.loads(text)
    global_meta = document.get("global", {})
    captures = document.get("captures", [])
    first_capture = captures[0] if captures else {}
    return {
        "source": "sigmf_metadata", "sample_rate_hz": global_meta.get("core:sample_rate"),
        "datatype": global_meta.get("core:datatype"),
        "centre_frequency_hz": first_capture.get("core:frequency"),
        "datetime": first_capture.get("core:datetime"),
        "known_fields": sorted(set(global_meta) | set(first_capture)),
        "capture_count": len(captures),
    }
