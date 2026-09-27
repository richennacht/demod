#!/usr/bin/env python3
"""SQLite metadata catalog for DEmod raw-IQ files.

Only manifests and derived metadata go in the database. Raw IQ stays in file or
object storage and is referenced by URI plus SHA-256.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS recordings (
  recording_id TEXT PRIMARY KEY, split TEXT NOT NULL, source_type TEXT NOT NULL,
  data_uri TEXT NOT NULL, sigmf_meta_uri TEXT, datatype TEXT, sample_rate_hz REAL,
  center_frequency_hz REAL, sha256 TEXT, dataset_name TEXT, label_source TEXT NOT NULL,
  manifest_json TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS annotations (
  annotation_id TEXT PRIMARY KEY, recording_id TEXT NOT NULL REFERENCES recordings(recording_id),
  sample_start INTEGER NOT NULL, sample_count INTEGER NOT NULL, modulation TEXT,
  modulation_family TEXT, signal_family TEXT, signal_present TEXT NOT NULL,
  truth_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_recordings_split ON recordings(split, source_type);
CREATE INDEX IF NOT EXISTS idx_annotations_modulation ON annotations(modulation);
CREATE TABLE IF NOT EXISTS analysis_runs (
  run_id TEXT PRIMARY KEY, recording_id TEXT NOT NULL REFERENCES recordings(recording_id),
  pipeline_version TEXT NOT NULL, result_uri TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


def connect(database: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database)
    connection.executescript(SCHEMA)
    return connection


def ingest_manifest(connection: sqlite3.Connection, manifest: Path) -> int:
    inserted = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        provenance, data, capture = record["provenance"], record["data"], record["capture"]
        connection.execute(
            "INSERT OR REPLACE INTO recordings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT created_at FROM recordings WHERE recording_id = ?), CURRENT_TIMESTAMP))",
            (record["recording_id"], record["split"], record["source_type"], data["uri"], data.get("sigmf_meta_uri"), data.get("datatype"), capture.get("sample_rate_hz"), capture.get("center_frequency_hz"), provenance.get("sha256"), provenance["dataset_name"], provenance["label_source"], json.dumps(record, sort_keys=True), record["recording_id"]),
        )
        for annotation in record["annotations"]:
            truth = annotation["truth"]
            connection.execute(
                "INSERT OR REPLACE INTO annotations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (annotation["annotation_id"], record["recording_id"], annotation["sample_start"], annotation["sample_count"], truth.get("modulation"), truth.get("modulation_family"), truth.get("signal_family"), truth["signal_present"], json.dumps(truth, sort_keys=True)),
            )
        inserted += 1
    connection.commit()
    return inserted


def main() -> None:
    parser = argparse.ArgumentParser(description="Create and populate the DEmod metadata catalog.")
    parser.add_argument("database", type=Path)
    parser.add_argument("--ingest", type=Path, help="DEmod records.jsonl manifest")
    args = parser.parse_args()
    connection = connect(args.database)
    try:
        count = ingest_manifest(connection, args.ingest) if args.ingest else 0
        print(json.dumps({"database": str(args.database), "records_ingested": count}, indent=2))
    finally:
        connection.close()


if __name__ == "__main__":
    main()
