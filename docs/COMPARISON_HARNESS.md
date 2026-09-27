# DSP-versus-automated extraction harness

`src/local_comparison_api.py` is the first directly testable comparison surface. It accepts an authorised raw IQ byte stream in memory, computes the existing deterministic DSP features, and presents their DC/CFO values beside the current small learned baseline. It deliberately runs on `127.0.0.1` by default and never writes the submitted capture, a derived IQ file, or a report to disk.

## Start it

Install the optional NumPy visualisation dependency, then run the API from the repository root:

```powershell
python -m pip install -r requirements-dsp.txt
python src/local_comparison_api.py
```

The service has `GET /health` and `POST /analyse`. The post body is raw bytes, not a JSON/base64 copy. Supply two explicit analyst hypotheses because a headerless IQ stream cannot reveal them reliably:

```powershell
Invoke-WebRequest http://127.0.0.1:8787/analyse -Method Post -InFile .\capture.iq `
  -ContentType application/octet-stream `
  -Headers @{ "X-DEmod-IQ-Format" = "s16le"; "X-DEmod-Sample-Rate" = "2400000" }
```

The response contains `manual_dsp`, `automated_parameter_comparison`, FFT/STFT plot data when NumPy is installed, and a `provenance` block. The test-harness input cap is 16 MiB. That is intentional: it makes a bounded comparison experiment, not an architecture that uploads or holds terabyte-scale RF archives. A ministry deployment should run the same worker inside the authorised network and use chunked/object-store ingestion controlled by the data owner.

## What is compared today

| Parameter | Deterministic branch | Automated branch | Current decision rule |
| --- | --- | --- | --- |
| DC I/Q | Complex sample mean | 7→12→3 MLP estimate | Report absolute disagreement; do not silently choose a value. |
| Coarse carrier offset | Fourth-power phase-increment estimator | Same MLP target | Report absolute disagreement; it is intentionally fallible outside the recipe distribution. |
| Spectrum, occupied bandwidth, burst candidates | DFT/NumPy FFT, STFT, energy segmentation | None | Manual evidence only. |
| Sample rate, centre frequency, timing, modulation, FEC/interleaving | Analyst metadata or future supported estimators | None | Explicitly unsupported by this trained baseline. |

The learned baseline is trained afresh, in memory, from `data/recipes/mvp-recipes.json`; it sees generated samples but not their recipes during inference. It is an auditable calibration comparison, not a claim of blind operational extraction. Its manual CFO baseline uses the fourth-power technique appropriate mainly to PSK-like content; the report therefore preserves disagreement rather than masking it.

## Research basis and next acceptance test

The feature branch follows the FFT/STFT workflow in [PySDR's frequency-domain guide](https://pysdr.org/content/frequency_domain) and [NumPy FFT documentation](https://numpy.org/doc/stable/reference/routines.fft.html). The carrier/modulation evidence is framed by [Dobre et al. (2007)](https://doi.org/10.1049/iet-com:20050176) and the hybrid-model design is constrained by [Thakur & Imtiaz (2026)](https://www.mdpi.com/2079-9292/15/10/2163).

The next legitimate model comparison is symbol timing: add fractional timing-offset/pulse-shape labels to the recipe generator, then compare a learned rate/phase predictor against Oerder–Meyr acquisition and Gardner/Mueller–Müller tracking. Until that labelled curriculum exists, DEmod must not report a learned timing result.
