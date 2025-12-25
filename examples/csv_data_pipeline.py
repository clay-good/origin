#!/usr/bin/env python3
"""
CSV Data Pipeline Example - Origin Provenance Tracking

This example demonstrates how to track provenance for tabular data
loaded from CSV files. This is common in traditional ML pipelines
using pandas, scikit-learn, or similar tools.

No external dependencies required - uses only Python standard library.

Run this example:
    python examples/csv_data_pipeline.py
"""

import csv
import os
import tempfile
from typing import Any, Dict, Iterator, List, Tuple
from datetime import datetime

# Origin imports
from origin.hooks.base import BaseHook
from origin.storage.database import ProvenanceDatabase
from origin.core.record import SourceRecord
from origin.query.engine import QueryEngine
from origin.cards.generator import ProvenanceCardGenerator


class CSVDataHook(BaseHook):
    """
    Hook for CSV data pipelines.

    Tracks provenance for tabular data loaded from CSV files.
    Each row becomes a sample, fingerprinted based on its contents.
    """

    def __init__(
        self,
        db: ProvenanceDatabase,
        session_id: str,
        source_id: str = "csv_data",
        license_id: str = None,
        include_header: bool = True,
    ):
        """
        Initialize the CSV hook.

        Args:
            db: ProvenanceDatabase instance.
            session_id: Session ID for this run.
            source_id: Identifier for this data source.
            license_id: SPDX license ID for the data.
            include_header: Whether to include column names in sample data.
        """
        super().__init__(db, session_id, source_id, license_id)
        self.include_header = include_header
        self.columns = []

    def _extract_samples(self, batch: Any) -> List[Tuple[Any, Dict[str, Any]]]:
        """Extract samples from a batch of CSV rows."""
        samples = []

        if isinstance(batch, list):
            for i, row in enumerate(batch):
                if isinstance(row, dict):
                    # Row is already a dict (from DictReader)
                    sample_data = row
                    metadata = {"index": i, "columns": list(row.keys())}
                elif isinstance(row, (list, tuple)):
                    # Row is a list/tuple - convert to dict if we have columns
                    if self.columns:
                        sample_data = dict(zip(self.columns, row))
                    else:
                        sample_data = {"values": list(row)}
                    metadata = {"index": i}
                else:
                    sample_data = {"value": row}
                    metadata = {"index": i, "format": "unknown"}

                samples.append((sample_data, metadata))
        else:
            samples.append((batch, {"format": "single"}))

        return samples

    def load_csv(
        self,
        filepath: str,
        batch_size: int = 100,
        delimiter: str = ",",
        has_header: bool = True,
    ) -> Iterator[List[Dict]]:
        """
        Load a CSV file and yield batches of rows.

        Args:
            filepath: Path to the CSV file.
            batch_size: Number of rows per batch.
            delimiter: CSV delimiter character.
            has_header: Whether the first row is a header.

        Yields:
            Batches of row dictionaries.
        """
        batch = []

        with open(filepath, "r", newline="", encoding="utf-8") as f:
            if has_header:
                reader = csv.DictReader(f, delimiter=delimiter)
                self.columns = reader.fieldnames or []
            else:
                reader = csv.reader(f, delimiter=delimiter)
                self.columns = []

            for row in reader:
                if has_header:
                    batch.append(dict(row))
                else:
                    batch.append(list(row))

                if len(batch) >= batch_size:
                    self.observe(batch)
                    yield batch
                    batch = []

        if batch:
            self.observe(batch)
            yield batch


def create_sample_csv_files(temp_dir: str) -> Dict[str, str]:
    """Create sample CSV files for demonstration."""
    files = {}

    # Training data CSV (MIT license - open data)
    train_path = os.path.join(temp_dir, "train_data.csv")
    with open(train_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "feature_1", "feature_2", "feature_3", "label"])
        for i in range(50):
            writer.writerow([
                f"train_{i:04d}",
                round(i * 0.1, 2),
                round(i * 0.2, 2),
                round(i * 0.3, 2),
                i % 3
            ])
    files["train"] = train_path

    # Validation data CSV (Apache-2.0 license)
    val_path = os.path.join(temp_dir, "val_data.csv")
    with open(val_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "feature_1", "feature_2", "feature_3", "label"])
        for i in range(20):
            writer.writerow([
                f"val_{i:04d}",
                round(i * 0.15, 2),
                round(i * 0.25, 2),
                round(i * 0.35, 2),
                i % 3
            ])
    files["val"] = val_path

    # External data CSV (CC-BY-NC-4.0 - non-commercial restriction)
    external_path = os.path.join(temp_dir, "external_data.csv")
    with open(external_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "feature_1", "feature_2", "feature_3", "label"])
        for i in range(15):
            writer.writerow([
                f"ext_{i:04d}",
                round(i * 0.12, 2),
                round(i * 0.22, 2),
                round(i * 0.32, 2),
                i % 3
            ])
    files["external"] = external_path

    return files


def main():
    """Demonstrate CSV pipeline with provenance tracking."""

    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "csv_provenance.db")

        print("=" * 60)
        print("Origin Provenance Tracking - CSV Data Pipeline Example")
        print("=" * 60)

        # Create sample CSV files
        print("\n[Setup] Creating sample CSV files...")
        csv_files = create_sample_csv_files(temp_dir)
        for name, path in csv_files.items():
            print(f"    {name}: {path}")

        # Initialize database
        db = ProvenanceDatabase(db_path)
        print(f"\n[Setup] Database initialized: {db_path}")

        # ─────────────────────────────────────────────────────────────
        # Register data sources with licenses
        # ─────────────────────────────────────────────────────────────
        print("\n[1] Registering data sources with license information...")

        sources = [
            SourceRecord(
                source_id="training_data",
                source_type="csv",
                source_path=csv_files["train"],
                license_id="MIT",
                first_seen=datetime.now().isoformat()
            ),
            SourceRecord(
                source_id="validation_data",
                source_type="csv",
                source_path=csv_files["val"],
                license_id="Apache-2.0",
                first_seen=datetime.now().isoformat()
            ),
            SourceRecord(
                source_id="external_data",
                source_type="csv",
                source_path=csv_files["external"],
                license_id="CC-BY-NC-4.0",
                first_seen=datetime.now().isoformat()
            ),
        ]

        for source in sources:
            db.record_source(source)
            print(f"    Registered: {source.source_id} ({source.license_id})")

        # ─────────────────────────────────────────────────────────────
        # Start training session
        # ─────────────────────────────────────────────────────────────
        print("\n[2] Starting training session...")
        session = db.begin_session(config_hash="tabular_classifier_v1")
        print(f"    Session ID: {session.session_id}")

        # ─────────────────────────────────────────────────────────────
        # Process each data source
        # ─────────────────────────────────────────────────────────────
        print("\n[3] Processing data with provenance tracking...")

        data_configs = [
            ("training_data", csv_files["train"], "MIT"),
            ("validation_data", csv_files["val"], "Apache-2.0"),
            ("external_data", csv_files["external"], "CC-BY-NC-4.0"),
        ]

        all_data = []
        for source_id, filepath, license_id in data_configs:
            print(f"\n    Processing: {source_id}")

            # Create hook for this source
            hook = CSVDataHook(
                db=db,
                session_id=session.session_id,
                source_id=source_id,
                license_id=license_id
            )

            # Load and process data
            source_data = []
            for batch in hook.load_csv(filepath, batch_size=10):
                for row in batch:
                    source_data.append(row)
                    all_data.append((row, source_id, license_id))

            stats = hook.get_stats()
            print(f"      Rows loaded: {len(source_data)}")
            print(f"      Batches: {stats['batches_observed']}")
            print(f"      Unique samples: {stats['unique_samples']}")

        print(f"\n    Total data points: {len(all_data)}")

        # ─────────────────────────────────────────────────────────────
        # Simulate training (just demonstrating the data flow)
        # ─────────────────────────────────────────────────────────────
        print("\n[4] Simulating model training...")

        # In a real pipeline, you would:
        # - Split data into features and labels
        # - Train a model (sklearn, xgboost, etc.)
        # - Track the model artifacts

        print("    Training simulated (provenance already recorded)")

        # ─────────────────────────────────────────────────────────────
        # End session
        # ─────────────────────────────────────────────────────────────
        print("\n[5] Ending session...")
        db.end_session(session.session_id, status="completed")
        print("    Session completed")

        # ─────────────────────────────────────────────────────────────
        # Query provenance
        # ─────────────────────────────────────────────────────────────
        print("\n[6] Querying provenance data...")

        engine = QueryEngine(db)

        session_info = db.get_session(session.session_id)
        batches = db.list_batches(session.session_id)
        print(f"\n    Session Summary:")
        print(f"      Status: {session_info.status if session_info else 'unknown'}")
        print(f"      Total batches: {len(batches)}")

        license_breakdown = engine.get_license_breakdown(session.session_id)
        total_samples = sum(license_breakdown.values())
        print(f"      Total samples: {total_samples}")
        print(f"\n    License Breakdown:")
        for license_id, count in license_breakdown.items():
            print(f"      {license_id}: {count} samples")

        # Check for conflicts
        conflicts = engine.find_conflicts(session.session_id)
        print(f"\n    License Conflicts: {len(conflicts)}")
        if conflicts:
            for conflict in conflicts:
                print(f"      - {conflict.get('license_a')} + {conflict.get('license_b')}")
        else:
            print("      No conflicts (all licenses are compatible)")

        # ─────────────────────────────────────────────────────────────
        # Important: Note the CC-BY-NC restriction
        # ─────────────────────────────────────────────────────────────
        print("\n    Important Notes:")
        print("      - CC-BY-NC-4.0 data is present")
        print("      - This restricts the model to non-commercial use")
        print("      - Origin flags this for your awareness")

        # ─────────────────────────────────────────────────────────────
        # Generate provenance card
        # ─────────────────────────────────────────────────────────────
        print("\n[7] Generating provenance card...")

        generator = ProvenanceCardGenerator(db)
        card = generator.generate(session.session_id)

        # Save card
        card_path = os.path.join(temp_dir, "PROVENANCE_CARD.md")
        with open(card_path, "w") as f:
            f.write(card)
        print(f"    Card saved: {card_path}")

        # Preview
        card_lines = card.split("\n")
        print("\n    Provenance Card Preview:")
        print("    " + "-" * 50)
        for line in card_lines[:15]:
            print(f"    {line}")
        print(f"    ... ({len(card_lines) - 15} more lines)")

        db.close()

        print("\n" + "=" * 60)
        print("CSV pipeline example completed!")
        print("=" * 60)
        print("\nKey points for tabular data pipelines:")
        print("  - Each row becomes a fingerprinted sample")
        print("  - Register sources with license information upfront")
        print("  - Use separate hooks for different data sources")
        print("  - Origin tracks which licenses apply to your model")
        print("  - Non-commercial restrictions are flagged automatically")


if __name__ == "__main__":
    main()
