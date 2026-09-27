"""On-demand, provenance-preserving windows from the POWDER PCP dataset.

Only the requested byte ranges are held in memory. Full captures are never
copied into the repository or generated training corpus.
"""

from __future__ import annotations

import concurrent.futures
import random
import struct
import urllib.request


BASE_URL = "https://huggingface.co/datasets/T-Arshad/POWDER_CoChannel_Protocol_Dataset/resolve/main"


def capture_urls() -> list[str]:
    """Return the 768 documented Round/Gain/state capture URLs."""
    urls = []
    for round_id in range(1, 4):
        for gain in (60, 70, 80, 90):
            for state in range(64):
                bits = format(state, "06b")
                urls.append(f"{BASE_URL}/Round{round_id}_Gain{gain}/otax3102node_tx{bits}_id{state}.bin")
    return urls


def read_window(url: str, offset_samples: int, length: int) -> list[complex]:
    request = urllib.request.Request(url, headers={"Range": f"bytes={offset_samples * 8}-{(offset_samples + length) * 8 - 1}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read()
    if len(raw) != length * 8:
        raise ValueError(f"Expected {length * 8} bytes from {url}, received {len(raw)}")
    values = struct.unpack("<" + "f" * (len(raw) // 4), raw)
    return [complex(values[index], values[index + 1]) for index in range(0, len(values), 2)]


def sampled_windows(count: int, seed: int = 26147, length: int = 4096, offset_samples: int = 0, workers: int = 12) -> list[list[complex]]:
    """Fetch a seeded, provenance-distinct subset concurrently, then release it."""
    if count < 1 or count > 768:
        raise ValueError("count must be between 1 and 768")
    selected = random.Random(seed).sample(capture_urls(), count)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(lambda url: read_window(url, offset_samples, length), selected))
