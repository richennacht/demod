# Data manifests

This repository stores metadata and small, permitted demonstration data only. Do not commit the 183.5 GB Zenodo archive or any capture whose licence/redistribution status has not been recorded.

`records.jsonl` will contain one object per capture conforming to `../schema/demod-record.schema.json`. Keep upstream `.sigmf-meta` files unmodified beside the data, and add DEmod's normalised record separately.

For real data, use `null` for unspecified physical/coding values. Synthetic records must fill all values the generator controls.
