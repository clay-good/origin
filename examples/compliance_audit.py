#!/usr/bin/env python3
"""
Compliance Audit Example - Origin Provenance Tracking

This example demonstrates how to use Origin for compliance auditing.
It shows how to:

1. Query training data provenance after the fact
2. Generate audit-ready reports
3. Check license compatibility
4. Export provenance data for regulators
5. Trace individual samples back to their sources

This is particularly relevant for EU AI Act compliance and other
regulatory requirements around AI transparency.

No external dependencies required - uses only Python standard library.

Run this example:
    python examples/compliance_audit.py
"""

import json
import os
import tempfile
from datetime import datetime

# Origin imports
from origin.storage.database import ProvenanceDatabase
from origin.core.fingerprint import fingerprint_sample, merkle_root as compute_merkle_root
from origin.core.record import ProvenanceRecord, BatchRecord, SourceRecord, LicenseRecord
from origin.query.engine import QueryEngine
from origin.core.license import LicenseAnalyzer
from origin.cards.generator import ProvenanceCardGenerator
from origin.export.formats import ProvenanceExporter


def setup_demo_database(db: ProvenanceDatabase) -> str:
    """
    Set up a demo database with multiple training sessions.

    Returns the session_id of the most recent session for auditing.
    """
    # Register sources with different licenses
    sources = [
        SourceRecord(
            source_id="public_dataset",
            source_type="download",
            source_path="https://example.com/public-data",
            license_id="CC-BY-4.0",
            first_seen="2025-01-01T00:00:00Z"
        ),
        SourceRecord(
            source_id="licensed_dataset",
            source_type="purchase",
            source_path="vendor://licensed-data-v2",
            license_id="proprietary",
            first_seen="2025-01-05T00:00:00Z"
        ),
        SourceRecord(
            source_id="open_source_data",
            source_type="repository",
            source_path="https://github.com/example/data",
            license_id="MIT",
            first_seen="2025-01-10T00:00:00Z"
        ),
        SourceRecord(
            source_id="research_dataset",
            source_type="academic",
            source_path="https://research.edu/dataset",
            license_id="CC-BY-NC-4.0",
            first_seen="2025-01-15T00:00:00Z"
        ),
    ]

    for source in sources:
        db.record_source(source)

    # Create a training session
    session = db.begin_session(config_hash="production_model_v2.1")

    # Simulate training with samples from different sources
    sample_data = [
        ("public_dataset", "CC-BY-4.0", [b"pub_sample_1", b"pub_sample_2", b"pub_sample_3"]),
        ("licensed_dataset", "proprietary", [b"lic_sample_1", b"lic_sample_2"]),
        ("open_source_data", "MIT", [b"oss_sample_1", b"oss_sample_2", b"oss_sample_3", b"oss_sample_4"]),
        ("research_dataset", "CC-BY-NC-4.0", [b"res_sample_1", b"res_sample_2"]),
    ]

    batch_idx = 0
    for source_id, license_id, samples in sample_data:
        sample_fingerprints = []

        for sample in samples:
            fingerprint = fingerprint_sample(sample)
            sample_fingerprints.append(fingerprint)

            record = ProvenanceRecord(
                sample_id=fingerprint,
                source_id=source_id,
                content_type="bytes",
                byte_size=len(sample),
                timestamp=datetime.now().isoformat(),
                metadata={"batch": batch_idx, "license_id": license_id}
            )
            db.record_sample(record)

        batch_merkle = compute_merkle_root(sample_fingerprints)
        batch = BatchRecord(
            batch_id=batch_merkle,
            session_id=session.session_id,
            batch_index=batch_idx,
            sample_count=len(sample_fingerprints),
            created_at=datetime.now().isoformat(),
            sample_ids=tuple(sample_fingerprints)
        )
        db.record_batch(batch)
        batch_idx += 1

    db.end_session(session.session_id, status="completed")
    return session.session_id


def main():
    """Demonstrate compliance auditing with Origin."""

    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "audit_provenance.db")

        print("=" * 70)
        print("Origin Provenance Tracking - Compliance Audit Example")
        print("=" * 70)

        # Setup
        print("\n[Setup] Creating demo database with training history...")
        db = ProvenanceDatabase(db_path)
        session_id = setup_demo_database(db)
        print(f"    Database: {db_path}")
        print(f"    Session to audit: {session_id}")

        # Initialize query engine
        engine = QueryEngine(db)
        analyzer = LicenseAnalyzer()

        # ─────────────────────────────────────────────────────────────
        # Audit 1: Session Overview
        # ─────────────────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("AUDIT 1: Session Overview")
        print("=" * 70)

        session = db.get_session(session_id)
        batches = db.list_batches(session_id)
        source_breakdown = engine.get_source_breakdown(session_id)
        total_samples = sum(source_breakdown.values())

        print(f"""
    Training Session: {session_id}
    ─────────────────────────────────────────
    Status:          {session.status if session else 'unknown'}
    Config Hash:     {session.config_hash if session else 'N/A'}
    Started:         {session.created_at if session else 'N/A'}
    ─────────────────────────────────────────
    Total Batches:   {len(batches)}
    Total Samples:   {total_samples}
        """)

        # ─────────────────────────────────────────────────────────────
        # Audit 2: Data Source Analysis
        # ─────────────────────────────────────────────────────────────
        print("=" * 70)
        print("AUDIT 2: Data Source Analysis")
        print("=" * 70)

        # Get sources from source breakdown
        print("\n    Data Sources Used in Training:")
        print("    " + "-" * 60)
        print(f"    {'Source ID':<20} {'Type':<12} {'License':<15}")
        print("    " + "-" * 60)

        for source_id in source_breakdown.keys():
            source = db.get_source(source_id)
            if source:
                print(f"    {source.source_id:<20} "
                      f"{source.source_type:<12} "
                      f"{source.license_id or 'unknown':<15}")

        # ─────────────────────────────────────────────────────────────
        # Audit 3: License Breakdown
        # ─────────────────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("AUDIT 3: License Breakdown")
        print("=" * 70)

        license_breakdown = engine.get_license_breakdown(session_id)
        total_samples = sum(license_breakdown.values())

        print("\n    License Distribution:")
        print("    " + "-" * 50)
        for license_id, count in sorted(license_breakdown.items()):
            percentage = (count / total_samples) * 100 if total_samples > 0 else 0
            bar = "█" * int(percentage / 5)
            print(f"    {license_id:<20} {count:>5} ({percentage:>5.1f}%) {bar}")

        # ─────────────────────────────────────────────────────────────
        # Audit 4: License Compatibility Check
        # ─────────────────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("AUDIT 4: License Compatibility Analysis")
        print("=" * 70)

        analysis = analyzer.analyze_session(db, session_id)

        print(f"""
    Effective License: {analysis.get('result_license', 'unknown')}

    Permissions Granted:
        {', '.join(analysis.get('permissions', [])) or 'None specified'}

    Restrictions Applied:
        {', '.join(analysis.get('restrictions', [])) or 'None specified'}

    Conditions Required:
        {', '.join(analysis.get('conditions', [])) or 'None specified'}
        """)

        # Check for conflicts
        conflicts = analysis.get("conflicts", [])
        if conflicts:
            print("    ⚠️  LICENSE CONFLICTS DETECTED:")
            print("    " + "-" * 50)
            for conflict in conflicts:
                print(f"    - {conflict.get('license_a')} + {conflict.get('license_b')}")
                print(f"      Reason: {conflict.get('reason')}")
        else:
            print("    ✓ No license conflicts detected")

        # Check for restrictions
        restrictions = analysis.get("restrictions", [])
        if "non-commercial" in restrictions:
            print("\n    ⚠️  COMMERCIAL USE RESTRICTION:")
            print("    This model includes CC-BY-NC data and cannot be used commercially.")

        if "proprietary" in str(license_breakdown.keys()).lower():
            print("\n    ⚠️  PROPRIETARY DATA INCLUDED:")
            print("    This model includes proprietary licensed data.")
            print("    Verify vendor agreement allows this use case.")

        # ─────────────────────────────────────────────────────────────
        # Audit 5: Sample Tracing
        # ─────────────────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("AUDIT 5: Sample Tracing (Demonstrating Traceability)")
        print("=" * 70)

        # Get a sample fingerprint to trace
        batches = db.list_batches(session_id)
        if batches:
            sample_ids = batches[0].sample_ids
            if sample_ids:
                sample_id = sample_ids[0]

                print(f"\n    Tracing sample: {sample_id[:32]}...")

                trace = engine.trace_sample(sample_id)
                sample_info = trace.get('sample', {})
                source_info = trace.get('source', {})
                first_batch = trace.get('batches', [{}])[0] if trace.get('batches') else {}
                print(f"""
    Sample Trace:
    ─────────────────────────────────────────
    Fingerprint:    {sample_info.get('sample_id', 'N/A')[:48]}...
    Source:         {sample_info.get('source_id', 'N/A')}
    License:        {source_info.get('license_id', 'N/A')}
    First Observed: {sample_info.get('timestamp', 'N/A')}
    Batch:          {first_batch.get('batch_id', 'N/A')[:32] if first_batch.get('batch_id') else 'N/A'}...
                """)

        # ─────────────────────────────────────────────────────────────
        # Audit 6: Export for Regulators
        # ─────────────────────────────────────────────────────────────
        print("=" * 70)
        print("AUDIT 6: Regulatory Export")
        print("=" * 70)

        exporter = ProvenanceExporter(db)

        # JSON export
        json_path = os.path.join(temp_dir, "audit_export.json")
        json_data = exporter.to_json(session_id, pretty=True)
        with open(json_path, "w") as f:
            f.write(json_data)
        print(f"\n    JSON Export: {json_path}")

        # Preview JSON
        parsed = json.loads(json_data)
        stats = parsed.get('statistics', {})
        print(f"    Export contains:")
        print(f"      - session: {parsed.get('session', {}).get('session_id', 'N/A')[:32]}...")
        print(f"      - total_samples: {stats.get('total_samples', 0)}")
        print(f"      - licenses_used: {stats.get('licenses_used', [])}")

        # ─────────────────────────────────────────────────────────────
        # Audit 7: Provenance Card Generation
        # ─────────────────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("AUDIT 7: Provenance Card (Human-Readable Report)")
        print("=" * 70)

        generator = ProvenanceCardGenerator(db)
        card = generator.generate(session_id)

        card_path = os.path.join(temp_dir, "PROVENANCE_CARD.md")
        with open(card_path, "w") as f:
            f.write(card)
        print(f"\n    Provenance Card: {card_path}")

        # Preview card
        print("\n    Card Preview:")
        print("    " + "-" * 60)
        for line in card.split("\n")[:20]:
            print(f"    {line}")
        print("    ...")

        # ─────────────────────────────────────────────────────────────
        # Audit Summary
        # ─────────────────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("AUDIT SUMMARY")
        print("=" * 70)

        findings = []
        if conflicts:
            findings.append("❌ License conflicts detected - requires legal review")
        if "non-commercial" in str(restrictions):
            findings.append("⚠️  Non-commercial restriction applies")
        if "proprietary" in str(license_breakdown.keys()).lower():
            findings.append("⚠️  Proprietary data included - verify license terms")
        if "unknown" in str(license_breakdown.keys()).lower():
            findings.append("⚠️  Unknown licenses present - investigate sources")

        if findings:
            print("\n    Findings:")
            for finding in findings:
                print(f"      {finding}")
        else:
            print("\n    ✓ No issues found")

        print(f"""
    Audit Artifacts Generated:
      - {json_path}
      - {card_path}

    These files can be provided to auditors or regulators
    as evidence of training data provenance.
        """)

        db.close()

        print("=" * 70)
        print("Compliance audit example completed!")
        print("=" * 70)
        print("""
Key compliance capabilities demonstrated:
  1. Session overview - what was trained, when
  2. Data source analysis - where data came from
  3. License breakdown - what licenses apply
  4. Compatibility check - potential conflicts
  5. Sample tracing - follow data lineage
  6. Regulatory export - machine-readable records
  7. Provenance cards - human-readable reports

Origin provides the technical foundation for AI transparency
and compliance with regulations like the EU AI Act.
        """)


if __name__ == "__main__":
    main()
