#!/usr/bin/env python3
"""
CLI Workflow Example - Origin Provenance Tracking

This example demonstrates how to use the Origin CLI tool for provenance
tracking workflows. It shows the full lifecycle from initialization to
generating reports, using shell commands that developers can run directly.

No external dependencies required - uses only Python standard library.

Run this example:
    python examples/cli_workflow.py
"""

import os
import subprocess
import tempfile
from datetime import datetime

# Origin imports for setting up test data
from origin.storage.database import ProvenanceDatabase
from origin.core.fingerprint import fingerprint_sample, merkle_root
from origin.core.record import ProvenanceRecord, BatchRecord, SourceRecord


def run_cli(cmd: str, cwd: str = None) -> tuple:
    """Run an Origin CLI command and return output."""
    full_cmd = f"PYTHONPATH=src python3 -m origin.cli.main {cmd}"
    result = subprocess.run(
        full_cmd,
        shell=True,
        capture_output=True,
        text=True,
        cwd=cwd or os.getcwd()
    )
    return result.stdout, result.stderr, result.returncode


def setup_demo_data(db_path: str) -> str:
    """Set up demo data for CLI examples."""
    db = ProvenanceDatabase(db_path)

    # Register sources
    sources = [
        ("training_images", "folder", "/data/images/train", "CC-BY-4.0"),
        ("validation_images", "folder", "/data/images/val", "CC-BY-4.0"),
        ("external_dataset", "download", "https://example.com/data", "MIT"),
    ]

    for source_id, source_type, source_path, license_id in sources:
        db.record_source(SourceRecord(
            source_id=source_id,
            source_type=source_type,
            source_path=source_path,
            license_id=license_id,
            first_seen=datetime.now().isoformat()
        ))

    # Create a training session with samples
    session = db.begin_session(config_hash="image_classifier_v1")

    # Add samples from each source
    sample_data = [
        ("training_images", [b"img1", b"img2", b"img3", b"img4", b"img5"]),
        ("validation_images", [b"val1", b"val2"]),
        ("external_dataset", [b"ext1", b"ext2", b"ext3"]),
    ]

    batch_idx = 0
    for source_id, samples in sample_data:
        fingerprints = []
        for sample in samples:
            fp = fingerprint_sample(sample)
            fingerprints.append(fp)

            db.record_sample(ProvenanceRecord(
                sample_id=fp,
                source_id=source_id,
                content_type="image",
                byte_size=len(sample),
                timestamp=datetime.now().isoformat(),
                metadata={}
            ))

        batch_fp = merkle_root(fingerprints)
        db.record_batch(BatchRecord(
            batch_id=batch_fp,
            session_id=session.session_id,
            batch_index=batch_idx,
            sample_count=len(fingerprints),
            created_at=datetime.now().isoformat(),
            sample_ids=tuple(fingerprints)
        ))
        batch_idx += 1

    db.end_session(session.session_id, status="completed")
    db.close()

    return session.session_id


def main():
    """Demonstrate CLI workflow with Origin."""

    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "provenance.db")

        print("=" * 70)
        print("Origin Provenance Tracking - CLI Workflow Example")
        print("=" * 70)

        # ─────────────────────────────────────────────────────────────
        # Step 1: Initialize a new database
        # ─────────────────────────────────────────────────────────────
        print("\n[1] Initialize a new provenance database")
        print("-" * 50)
        print(f"$ origin --path {db_path} init")

        stdout, stderr, code = run_cli(f"--path {db_path} init")
        print(stdout if stdout else "(Database initialized)")

        # Set up demo data
        print("\n    (Setting up demo data for examples...)")
        session_id = setup_demo_data(db_path)
        print(f"    Created session: {session_id[:32]}...")

        # ─────────────────────────────────────────────────────────────
        # Step 2: Check database status
        # ─────────────────────────────────────────────────────────────
        print("\n[2] Check database status")
        print("-" * 50)
        print(f"$ origin --path {db_path} status")

        stdout, stderr, code = run_cli(f"--path {db_path} status")
        print(stdout if stdout else stderr)

        # ─────────────────────────────────────────────────────────────
        # Step 3: List all sessions
        # ─────────────────────────────────────────────────────────────
        print("\n[3] List all training sessions")
        print("-" * 50)
        print(f"$ origin --path {db_path} sessions")

        stdout, stderr, code = run_cli(f"--path {db_path} sessions")
        print(stdout if stdout else stderr)

        # ─────────────────────────────────────────────────────────────
        # Step 4: Inspect a specific session
        # ─────────────────────────────────────────────────────────────
        print("\n[4] Inspect a specific session")
        print("-" * 50)
        print(f"$ origin --path {db_path} inspect {session_id[:16]}...")

        stdout, stderr, code = run_cli(f"--path {db_path} inspect {session_id}")
        print(stdout if stdout else stderr)

        # ─────────────────────────────────────────────────────────────
        # Step 5: Query for a specific license
        # ─────────────────────────────────────────────────────────────
        print("\n[5] Query for samples with a specific license")
        print("-" * 50)
        print(f"$ origin --path {db_path} query MIT")

        stdout, stderr, code = run_cli(f"--path {db_path} query MIT")
        print(stdout if stdout else stderr)

        # ─────────────────────────────────────────────────────────────
        # Step 6: Check for license conflicts
        # ─────────────────────────────────────────────────────────────
        print("\n[6] Check for license conflicts")
        print("-" * 50)
        print(f"$ origin --path {db_path} conflicts")

        stdout, stderr, code = run_cli(f"--path {db_path} conflicts")
        print(stdout if stdout else "(No conflicts detected)" if not stdout else stdout)

        # ─────────────────────────────────────────────────────────────
        # Step 7: Generate a provenance card
        # ─────────────────────────────────────────────────────────────
        print("\n[7] Generate a provenance card")
        print("-" * 50)
        card_path = os.path.join(temp_dir, "PROVENANCE_CARD.md")
        print(f"$ origin --path {db_path} card {session_id[:16]}... --output {card_path}")

        stdout, stderr, code = run_cli(f"--path {db_path} card {session_id} --output {card_path}")
        print(stdout if stdout else f"Card saved to: {card_path}")

        # Show card preview
        if os.path.exists(card_path):
            with open(card_path, "r") as f:
                card_content = f.read()
            print("\n    Card Preview:")
            for line in card_content.split("\n")[:15]:
                print(f"    {line}")
            print("    ...")

        # ─────────────────────────────────────────────────────────────
        # Step 8: Export session data
        # ─────────────────────────────────────────────────────────────
        print("\n[8] Export session data to JSON")
        print("-" * 50)
        export_path = os.path.join(temp_dir, "export.json")
        print(f"$ origin --path {db_path} export {session_id[:16]}... --format json --output {export_path}")

        stdout, stderr, code = run_cli(f"--path {db_path} export {session_id} --format json --output {export_path}")
        print(stdout if stdout else f"Exported to: {export_path}")

        # ─────────────────────────────────────────────────────────────
        # Step 9: Trace a sample
        # ─────────────────────────────────────────────────────────────
        print("\n[9] Trace a sample back to its source")
        print("-" * 50)

        # Get a sample ID to trace
        db = ProvenanceDatabase(db_path)
        batches = db.list_batches(session_id)
        if batches and batches[0].sample_ids:
            sample_id = batches[0].sample_ids[0]
            print(f"$ origin --path {db_path} trace {sample_id[:16]}...")

            stdout, stderr, code = run_cli(f"--path {db_path} trace {sample_id}")
            print(stdout if stdout else stderr)
        db.close()

        # ─────────────────────────────────────────────────────────────
        # Step 10: Show version
        # ─────────────────────────────────────────────────────────────
        print("\n[10] Show Origin version")
        print("-" * 50)
        print("$ origin version")

        stdout, stderr, code = run_cli("version")
        print(stdout if stdout else stderr)

        print("\n" + "=" * 70)
        print("CLI Workflow Example Complete!")
        print("=" * 70)
        print("""
CLI Commands Reference:
  origin init         Initialize a new provenance database
  origin status       Show database statistics
  origin sessions     List all training sessions
  origin inspect      Show details for a session
  origin query        Check if a license is present
  origin conflicts    List license conflicts
  origin card         Generate a provenance card
  origin export       Export session data (JSON/JSONL)
  origin trace        Trace a sample to its source
  origin version      Show version information

Global Options:
  --path PATH         Path to the provenance database (default: ./origin.db)
  --quiet, -q         Suppress non-essential output
  --json              Output in JSON format

For help on any command:
  origin <command> --help
        """)


if __name__ == "__main__":
    main()
