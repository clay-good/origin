#!/usr/bin/env python3
"""
Basic Usage Example - Origin Provenance Tracking

This example demonstrates the core functionality of Origin without requiring
any optional dependencies (PyTorch, HuggingFace). It shows how to:

1. Initialize a provenance database
2. Start a training session
3. Record samples and batches manually
4. Query provenance data
5. Generate a provenance card

Run this example:
    python examples/basic_usage.py
"""

import os
import tempfile
from datetime import datetime

# Origin imports (core library - no external dependencies)
from origin.storage.database import ProvenanceDatabase
from origin.core.fingerprint import fingerprint_sample, merkle_root as compute_merkle_root
from origin.core.record import ProvenanceRecord, BatchRecord, SourceRecord, LicenseRecord
from origin.query.engine import QueryEngine
from origin.cards.generator import ProvenanceCardGenerator


def main():
    """Demonstrate basic Origin usage."""

    # Create a temporary directory for this example
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "provenance.db")

        print("=" * 60)
        print("Origin Provenance Tracking - Basic Usage Example")
        print("=" * 60)

        # ─────────────────────────────────────────────────────────────
        # Step 1: Initialize the database
        # ─────────────────────────────────────────────────────────────
        print("\n[1] Initializing provenance database...")
        db = ProvenanceDatabase(db_path)
        print(f"    Database created at: {db_path}")

        # ─────────────────────────────────────────────────────────────
        # Step 2: Register data sources with license information
        # ─────────────────────────────────────────────────────────────
        print("\n[2] Registering data sources...")

        # Register a source with MIT license
        source_a = SourceRecord(
            source_id="synthetic_dataset_a",
            source_type="generated",
            source_path="/data/synthetic_a",
            license_id="MIT",
            first_seen=datetime.now().isoformat()
        )
        db.record_source(source_a)
        print(f"    Registered source: {source_a.source_id} (License: MIT)")

        # Register another source with Apache-2.0 license
        source_b = SourceRecord(
            source_id="synthetic_dataset_b",
            source_type="generated",
            source_path="/data/synthetic_b",
            license_id="Apache-2.0",
            first_seen=datetime.now().isoformat()
        )
        db.record_source(source_b)
        print(f"    Registered source: {source_b.source_id} (License: Apache-2.0)")

        # ─────────────────────────────────────────────────────────────
        # Step 3: Start a training session
        # ─────────────────────────────────────────────────────────────
        print("\n[3] Starting training session...")
        session = db.begin_session(config_hash="example_config_v1")
        print(f"    Session ID: {session.session_id}")
        print(f"    Started at: {session.created_at}")

        # ─────────────────────────────────────────────────────────────
        # Step 4: Simulate training with sample recording
        # ─────────────────────────────────────────────────────────────
        print("\n[4] Recording training samples...")

        # Simulate 3 batches of training data
        batches_data = [
            # Batch 1: Samples from source A
            {
                "source": "synthetic_dataset_a",
                "license": "MIT",
                "samples": [b"sample_a_1", b"sample_a_2", b"sample_a_3", b"sample_a_4"]
            },
            # Batch 2: Samples from source B
            {
                "source": "synthetic_dataset_b",
                "license": "Apache-2.0",
                "samples": [b"sample_b_1", b"sample_b_2", b"sample_b_3", b"sample_b_4"]
            },
            # Batch 3: Mixed samples
            {
                "source": "synthetic_dataset_a",
                "license": "MIT",
                "samples": [b"sample_a_5", b"sample_a_6", b"sample_b_5", b"sample_b_6"]
            },
        ]

        total_samples = 0
        for batch_idx, batch_info in enumerate(batches_data):
            print(f"\n    Batch {batch_idx + 1}:")

            # Fingerprint each sample in the batch
            sample_fingerprints = []
            for sample_data in batch_info["samples"]:
                # Compute SHA-256 fingerprint
                fingerprint = fingerprint_sample(sample_data)
                sample_fingerprints.append(fingerprint)

                # Create and record sample
                sample = ProvenanceRecord(
                    sample_id=fingerprint,
                    source_id=batch_info["source"],
                    content_type="bytes",
                    byte_size=len(sample_data),
                    timestamp=datetime.now().isoformat(),
                    metadata={"batch_index": batch_idx, "license_id": batch_info["license"]}
                )
                db.record_sample(sample)
                total_samples += 1

            # Compute Merkle root for batch integrity
            merkle_root = compute_merkle_root(sample_fingerprints)

            # Create and record batch
            batch = BatchRecord(
                batch_id=merkle_root,
                session_id=session.session_id,
                batch_index=batch_idx,
                sample_count=len(sample_fingerprints),
                created_at=datetime.now().isoformat(),
                sample_ids=tuple(sample_fingerprints)
            )
            db.record_batch(batch)

            print(f"      Source: {batch_info['source']}")
            print(f"      Samples: {len(batch_info['samples'])}")
            print(f"      Merkle root: {merkle_root[:16]}...")

        print(f"\n    Total samples recorded: {total_samples}")

        # ─────────────────────────────────────────────────────────────
        # Step 5: End the training session
        # ─────────────────────────────────────────────────────────────
        print("\n[5] Ending training session...")
        db.end_session(session.session_id, status="completed")
        print("    Session marked as completed")

        # ─────────────────────────────────────────────────────────────
        # Step 6: Query provenance data
        # ─────────────────────────────────────────────────────────────
        print("\n[6] Querying provenance data...")

        engine = QueryEngine(db)

        # Get license breakdown
        license_breakdown = engine.get_license_breakdown(session.session_id)
        print(f"\n    License Breakdown:")
        for license_id, count in license_breakdown.items():
            print(f"      {license_id}: {count} samples")

        # Get source breakdown
        source_breakdown = engine.get_source_breakdown(session.session_id)
        print(f"\n    Source Breakdown:")
        for source_id, count in source_breakdown.items():
            print(f"      {source_id}: {count} samples")

        # Check for license conflicts
        conflicts = engine.find_conflicts(session.session_id)
        print(f"\n    License Conflicts: {len(conflicts)}")
        if conflicts:
            for conflict in conflicts:
                print(f"      - {conflict['license_a']} + {conflict['license_b']}: {conflict['conflict_type']}")
        else:
            print("      No conflicts detected (MIT and Apache-2.0 are compatible)")

        # ─────────────────────────────────────────────────────────────
        # Step 7: Generate a provenance card
        # ─────────────────────────────────────────────────────────────
        print("\n[7] Generating provenance card...")

        generator = ProvenanceCardGenerator(db)
        card = generator.generate(session.session_id)

        # Print first 30 lines of the card
        card_lines = card.split("\n")
        print("\n    Provenance Card Preview (first 30 lines):")
        print("    " + "-" * 50)
        for line in card_lines[:30]:
            print(f"    {line}")
        if len(card_lines) > 30:
            print(f"    ... ({len(card_lines) - 30} more lines)")

        # ─────────────────────────────────────────────────────────────
        # Cleanup
        # ─────────────────────────────────────────────────────────────
        db.close()

        print("\n" + "=" * 60)
        print("Example completed successfully!")
        print("=" * 60)
        print("\nKey takeaways:")
        print("  - Origin tracks data provenance without modifying your data")
        print("  - Each sample gets a unique SHA-256 fingerprint")
        print("  - Batches use Merkle trees for integrity verification")
        print("  - License compatibility is checked automatically")
        print("  - Provenance cards provide audit-ready documentation")


if __name__ == "__main__":
    main()
