# Dataset and Label Schema

## Decision

DEmod will use two deliberately different data roles:

| Role | Initial source | What it proves | What it does not prove |
| --- | --- | --- | --- |
| Real-capture evaluation | Badger and Kim's SDR-captured SigMF RFML corpus | That ingestion, visualization, segmentation, and classification behave on recorded RF | Exact transmitter-side bits, FEC, interleaver, or all physical impairments |
| Synthetic training and regression | DEmod-controlled waveform generator | Exact labels for format, modulation, impairments, and coding pipeline | Generalization to every receiver, channel, and protocol |
| Baseline-only benchmark | RadioML 2016.10A | Comparable automatic-modulation-recognition baseline | Real-world performance; it is generated, not over-the-air data |

The selected real corpus is the **Open-Sourced Time-Frequency Domain RF Classification Framework** dataset (Zenodo DOI `10.5281/zenodo.4603987`). It contains SDR-captured, labelled IQ in paired `.sigmf-data` and `.sigmf-meta` files. Its complete archive is 183.5 GB, so the hackathon repository must record a pinned, licensed subset rather than commit or download the entire archive. Start with its `testing_data_1msps` material and a small, documented selection from each permitted class.

The corpus has labels for operational class/device families (for example LoRa variants, analogue/digital PTT, doorbells and keyfobs). These are valuable for signal presence, coarse family classification, visualization, and metadata ingestion, but are **not** ground truth for modulation order, symbol rate, FEC, or interleaver. Those properties must remain `unknown` unless a source explicitly supplies them.

The accompanying class map uses a source-specific integer `core:class` (for example LoRa 125 = `0`, GD55 = `1`, NFM = `2`, white noise = `8`, and centre-FFT artefact = `9`). Preserve that original integer under `source_class`, but map it into DEmod's human-readable `signal_family` rather than treating the integer as a modulation label. The repository code is MIT-licensed; confirm the data redistribution terms from the dataset owner before committing any capture bytes.

## Why SigMF plus a manifest

SigMF is the capture interchange format. Its `global` section records the sample datatype and sample rate; `captures` record sample position and centre frequency; `annotations` locate a signal in time/frequency and may carry a label. It is ideal for raw IQ provenance and segment bounds, but it does not prescribe all the experiment labels DEmod needs.

Each file should therefore retain its source `.sigmf-meta` unchanged and receive one row in `data/manifests/records.jsonl`, validated against `data/schema/demod-record.schema.json`. DEmod-specific keys use the `demod:` namespace so they cannot be confused with source-provided SigMF values.

## Label contract

### Values that must never be guessed into ground truth

For real captures, use `null` and a clear `label_source` whenever modulation detail, baud rate, FEC, interleaver, gain, transmitter settings, or payload bits are not supplied. A model prediction belongs in `predictions`, never in `truth`.

### Provenance and split

Every record has a stable ID, SHA-256, license/provenance, source type, and a split. Split **by capture session / transmitter / receiver / location**, not by random adjacent windows. Otherwise nearly identical windows leak into train and test sets and inflate accuracy.

### One record, many annotations

A record represents a capture file. Each annotation represents an occupied or noise segment using absolute `sample_start` and `sample_count`, plus RF band edges when known. A wideband file may therefore have several annotations.

### Hierarchical targets

1. `signal_present`: `present`, `noise`, or `artifact`.
2. `signal_family`: coarse source/protocol family; can be known for real data.
3. `modulation_family` and `modulation`: only when independently known.
4. `parameters`: numeric truth only when measured, instrumented, or generated.
5. `coding`: FEC/interleaver/scrambler truth; generally synthetic-only.

`label_source` is one of `generated`, `instrumented_tx`, `protocol_decoder`, `source_metadata`, `manual_review`, or `unknown`; `label_confidence` is `high`, `medium`, `low`, or `unknown`.

## Synthetic-generation requirements

Synthetic data must use the identical envelope but fully populate `truth`. For each generated segment persist: raw sample format and byte order; sample rate; centre frequency; modulation and order; pulse shape and roll-off; symbols per sample; channel/noise model and SNR; carrier and sampling-frequency offsets; phase noise; DC offset; I/Q gain and phase imbalance; clipping/interferers; and FEC, interleaver, and scrambler configuration. Store a payload or bitstream **hash**, not plaintext payload, unless the test specifically requires reproducibility of the bits.

Each generated record should store a generator version, deterministic seed, and parameter-spec hash. This makes every waveform reproducible and prevents labels drifting after a generator change.

## Initial data plan

1. Add 20–50 legally redistributable or source-referenced SigMF real captures as an **evaluation-only** set, retaining original metadata and checksums.
2. Generate 2,000–10,000 short synthetic captures across supported formats/modulations and impairment ranges. This is sufficient for a credible demonstrator; it is not a production-scale training claim.
3. Keep the final blind demonstration capture/session entirely separate from tuning and training.
4. Report performance separately for synthetic exact-truth tasks and real-capture coarse-label tasks.

## Sources

- Zenodo dataset: https://doi.org/10.5281/zenodo.4603987
- SigMF specification: https://github.com/sigmf/SigMF/blob/main/sigmf-schema.json
- Elmaghbub and Hamdaoui, real LoRa/SigMF capture design: https://arxiv.org/abs/2201.02213
- RadioML baseline description: https://github.com/sofwerx/deepsig_datasets
