# Storage Architecture

## MVP: filesystem + SigMF + SQLite

Raw IQ is large, sequentially read, and immutable. Store it as `.sigmf-data` files with neighbouring `.sigmf-meta` JSON, partitioned by `source_type/dataset_version/split/recording_id`. Store only the URI, checksum, SigMF-derived metadata, labels, and analysis-result URIs in SQLite.

```text
raw IQ / SigMF metadata  -> local disk for MVP; object storage in deployment
                                  |
DEmod manifest JSONL ------------> SQLite for MVP; PostgreSQL in deployment
                                  |
analysis JSON / plots -----------> file/object storage; URI recorded in catalog
```

Never place multi-GB raw IQ blobs in SQLite, PostgreSQL, or a browser database. Database backups become slow, range reads are poor, and metadata queries compete with binary transfers.

## Deployment evolution

| Scale | Raw data | Catalog | Why |
| --- | --- | --- | --- |
| Laptop/demo | filesystem + SigMF | SQLite | zero operations; deterministic offline demo |
| Team/ministry pilot | S3-compatible MinIO or cloud object storage | PostgreSQL | durable object storage, shared catalog, access control |
| Large corpus | object storage with Parquet/JSONL manifests | PostgreSQL + queue/workers | lifecycle rules, batch indexing, async analysis |

Use object keys such as `captures/synthetic/v0.1/train/syn-qpsk-00000.sigmf-data`. Store SHA-256, byte size, source license, and immutable dataset version in the catalog. Keep raw captures immutable; derived plots, FFTs, and model predictions are versioned artifacts linked through `analysis_runs`.

## Recipe-first synthetic data

Synthetic IQ is not stored as a dataset. Versioned JSON recipes specify modulation, sample rate, pulse/channel choices, and impairment distributions. The batch loader samples a recipe with a deterministic seed, creates IQ in RAM, returns only IQ and permitted task labels to the model, and releases the batch after training. Recipe ID, seed, and realised impairment values are hidden audit metadata.

The initial curriculum has two stages: clean linear modulation, then RF effects added one at a time—AWGN, CFO, DC offset, I/Q gain/phase imbalance, phase noise, impulsive noise, clipping, and multipath. Add a stage only when the real-vs-synthetic diagnostic identifies a remaining gap.

## First commands

```powershell
python src/generate_synthetic.py --recipes data/recipes/mvp-recipes.json --count 24
```

The generator writes no IQ files. Real raw captures remain outside Git; commit recipes, schema, source manifests, and code—not binary recordings.
