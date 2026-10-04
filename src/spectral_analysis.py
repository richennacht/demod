"""NumPy-backed, provenance-ready spectrum, waterfall and burst analysis."""

from __future__ import annotations

from typing import Any

import numpy as np


def _db(power: np.ndarray) -> np.ndarray:
    floor = np.finfo(float).tiny
    return 10.0 * np.log10(np.maximum(power, floor))


def _compact(values: np.ndarray, limit: int) -> list[float]:
    if len(values) <= limit:
        return [round(float(value), 5) for value in values]
    indices = np.linspace(0, len(values) - 1, limit, dtype=int)
    return [round(float(values[index]), 5) for index in indices]


def segment_energy(samples: np.ndarray, sample_rate_hz: float, window: int = 256, min_duration_ms: float = 1.0) -> dict[str, Any]:
    """Robust energy segmentation; labels are candidates, not ground truth."""
    power = np.abs(samples) ** 2
    kernel = np.ones(min(window, len(samples))) / min(window, len(samples))
    smoothed = np.convolve(power, kernel, mode="same")
    # A median fails when a long burst occupies about half a preview.  The lower
    # fifth remains a conservative background candidate; this is still only a
    # segment proposal, not a signal/noise decision.
    baseline = float(np.quantile(smoothed, 0.2))
    lower_background = smoothed[smoothed <= baseline]
    mad = float(np.median(np.abs(lower_background - baseline))) if len(lower_background) else 0.0
    threshold = max(baseline + 6 * 1.4826 * mad, baseline * 3)
    active = smoothed > threshold
    minimum = max(1, int(sample_rate_hz * min_duration_ms / 1000))
    segments: list[dict[str, Any]] = []
    start: int | None = None
    for index, flag in enumerate(active):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            if index - start >= minimum:
                segments.append({"sample_start": start, "sample_count": index - start, "duration_seconds": round((index - start) / sample_rate_hz, 7)})
            start = None
    if start is not None and len(samples) - start >= minimum:
        segments.append({"sample_start": start, "sample_count": len(samples) - start, "duration_seconds": round((len(samples) - start) / sample_rate_hz, 7)})
    # Max-pooled envelope so short bursts survive decimation in the time plot.
    points = min(1024, len(smoothed))
    edges = np.linspace(0, len(smoothed), points + 1, dtype=int)
    pooled = np.array([float(np.max(smoothed[a:b])) if b > a else float(smoothed[a]) for a, b in zip(edges[:-1], edges[1:])])
    envelope = {
        "time_seconds": [round(float((a + b) / 2 / sample_rate_hz), 7) for a, b in zip(edges[:-1], edges[1:])],
        "power_db": [round(float(value), 3) for value in _db(pooled)],
        "threshold_db": round(float(_db(np.array([threshold]))[0]), 3),
        "baseline_db": round(float(_db(np.array([baseline]))[0]), 3),
        "pooling": "max over equal blocks of the smoothed power",
    }
    return {"method": "robust_smoothed_energy", "window_samples": min(window, len(samples)), "baseline_power": baseline, "mad_power": mad, "threshold_power": threshold, "segments": segments, "envelope": envelope, "limitation": "An energy segment is not proof of a protocol burst, radar pulse, or interference."}


def analyse_spectrum(samples: list[complex], sample_rate_hz: float, fft_size: int = 1024, hop_size: int = 256) -> dict[str, Any]:
    """Compute true FFT/PSD, STFT waterfall, constellation points and energy segments.

    Results are compact JSON previews; callers retain the raw data and analysis
    settings to reproduce full-resolution plots.
    """
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    signal = np.asarray(samples, dtype=np.complex128)
    if len(signal) < 8:
        raise ValueError("At least eight complex samples are required.")
    size = min(int(fft_size), len(signal))
    window = np.hanning(size)
    centered = signal - np.mean(signal)
    fft = np.fft.fftshift(np.fft.fft(centered[:size] * window, n=size))
    frequencies = np.fft.fftshift(np.fft.fftfreq(size, d=1.0 / sample_rate_hz))
    psd = _db(np.abs(fft) ** 2 / max(np.sum(window**2), 1.0))
    frames = 1 + max(0, (len(centered) - size) // max(1, hop_size))
    frames = min(frames, 128)  # preview cap; streaming worker will use all frames.
    stft_rows = []
    stft_times = []
    absolute_rows: list[np.ndarray] = []
    for frame in range(frames):
        start = frame * hop_size
        block = centered[start:start + size]
        if len(block) < size:
            break
        row = _db(np.abs(np.fft.fftshift(np.fft.fft(block * window))) ** 2 / max(np.sum(window**2), 1.0))
        absolute_rows.append(row.copy())
        row -= np.max(row)
        stft_rows.append(_compact(np.clip(row, -90, 0), 256))
        stft_times.append(round((start + size / 2) / sample_rate_hz, 7))
    # Welch average and max hold over the whole capture (50% overlap), so a
    # quiet first block cannot hide later bursts. Same window and bin grid.
    welch_hop = max(1, size // 2)
    starts = np.arange(0, len(centered) - size + 1, welch_hop)
    mean_power = np.zeros(size)
    max_power = np.zeros(size)
    for chunk in range(0, len(starts), 512):
        block_starts = starts[chunk:chunk + 512]
        frames_block = np.stack([centered[start:start + size] for start in block_starts]) * window
        power = np.abs(np.fft.fftshift(np.fft.fft(frames_block, axis=1), axes=1)) ** 2 / max(np.sum(window**2), 1.0)
        mean_power += power.sum(axis=0)
        max_power = np.maximum(max_power, power.max(axis=0))
    mean_power /= max(1, len(starts))
    welch_db = _db(mean_power)
    welch_reference = float(np.max(welch_db))
    peak_index = int(np.argmax(psd))
    power_linear = 10 ** (psd / 10)
    order = np.argsort(power_linear)[::-1]
    chosen: list[int] = []
    cumulative = 0.0
    for index in order:
        chosen.append(int(index)); cumulative += float(power_linear[index])
        if cumulative >= 0.99 * float(np.sum(power_linear)):
            break
    bandwidth = float(frequencies[max(chosen)] - frequencies[min(chosen)]) if chosen else 0.0
    constellation = centered[np.linspace(0, len(centered) - 1, min(512, len(centered)), dtype=int)]
    if absolute_rows:
        stacked = np.vstack(absolute_rows)
        global_rows = [[round(value, 1) for value in _compact(np.clip(row - float(np.max(stacked)), -120, 0), 256)] for row in stacked]
    else:
        global_rows = []
    segmentation = segment_energy(centered, sample_rate_hz)
    first = segmentation["segments"][0]["sample_start"] if segmentation["segments"] else 0
    preview_start = max(0, min(first - 128, len(centered) - 1024))
    preview = centered[preview_start:preview_start + 1024]
    time_preview = {"start_sample": int(preview_start), "sample_count": int(len(preview)), "i": [round(float(v), 6) for v in preview.real], "q": [round(float(v), 6) for v in preview.imag], "dc_removed": True, "placement": "128 samples before the first energy segment, else the capture start"}
    return {
        "analysis_type": "numpy_fft_stft_v1",
        "input": {"sample_count": int(len(signal)), "sample_rate_hz": sample_rate_hz, "dc_removed_for_spectral_plots": True, "raw_samples_modified": False},
        "spectrum": {"window": "hann", "fft_size": size, "frequency_hz": _compact(frequencies, 512), "power_db": _compact(psd - np.max(psd), 512), "power_db_scope": "single FFT of the first fft_size samples", "peak_frequency_hz": round(float(frequencies[peak_index]), 5), "occupied_bandwidth_99pct_hz": round(abs(bandwidth), 5),
                     "welch_power_db": _compact(welch_db - welch_reference, 512), "max_hold_db": _compact(_db(max_power) - welch_reference, 512), "welch_frames": int(len(starts)), "welch_overlap": 0.5, "welch_reference": "dB relative to the peak of the Welch average"},
        "waterfall": {"fft_size": size, "hop_size": hop_size, "time_seconds": stft_times, "frequency_hz": _compact(frequencies, 256), "power_db_relative": stft_rows, "power_db_global": global_rows, "global_reference": "dB relative to the strongest bin across all shown frames; power_db_relative normalises each frame to its own peak", "frame_cap": 128},
        "time_preview": time_preview,
        "constellation": {"sampled_points": [{"i": round(float(value.real), 6), "q": round(float(value.imag), 6)} for value in constellation], "note": "DC-centred display points; this is not timing/carrier-corrected."},
        "segmentation": segmentation,
        "provenance": {"library": "numpy", "denoising_applied": False, "plot_values": "computed from interpreted complex IQ"},
    }
