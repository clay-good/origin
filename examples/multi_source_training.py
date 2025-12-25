#!/usr/bin/env python3
"""
Multi-Source Training Example - Origin Provenance Tracking

This example demonstrates how to track provenance when training on data
from multiple sources with different licenses. This is a common scenario
when combining:

- Public datasets (various CC licenses)
- Proprietary datasets (vendor licenses)
- User-generated data (terms of service)
- Synthetic data (generated, no license restrictions)

No external dependencies required - uses only Python standard library.

Run this example:
    python examples/multi_source_training.py
"""

import os
import tempfile
from datetime import datetime
from typing import Any, Dict, Iterator, List, Tuple

# Origin imports
from origin.hooks.base import BaseHook
from origin.storage.database import ProvenanceDatabase
from origin.core.fingerprint import fingerprint_sample, merkle_root as compute_merkle_root
from origin.core.record import ProvenanceRecord, BatchRecord, SourceRecord, LicenseRecord
from origin.core.license import LicenseAnalyzer
from origin.query.engine import QueryEngine
from origin.cards.generator import ProvenanceCardGenerator


class MultiSourceHook(BaseHook):
    """
    Hook that tracks samples from multiple sources in a single batch.

    This is useful when your batches contain mixed data from different
    sources, each with their own license.
    """

    def __init__(
        self,
        db: ProvenanceDatabase,
        session_id: str,
    ):
        # Initialize without a fixed source/license since we'll handle multiple
        super().__init__(db, session_id, source_id="multi_source", license_id=None)

    def _extract_samples(self, batch: Any) -> List[Tuple[Any, Dict[str, Any]]]:
        """Extract samples with source and license metadata."""
        samples = []

        if isinstance(batch, list):
            for i, item in enumerate(batch):
                if isinstance(item, dict):
                    # Extract source and license from item metadata
                    sample_data = item.get("data", item)
                    metadata = {
                        "index": i,
                        "source_id": item.get("source_id", "unknown"),
                        "license_id": item.get("license_id", "unknown"),
                    }
                    samples.append((sample_data, metadata))
                else:
                    samples.append((item, {"index": i}))
        else:
            samples.append((batch, {}))

        return samples

    def observe_with_source(
        self,
        batch: List[Dict],
        source_id: str,
        license_id: str
    ) -> str:
        """
        Observe a batch where all items share the same source/license.

        Args:
            batch: List of data items.
            source_id: Source identifier for all items.
            license_id: License for all items.

        Returns:
            Batch ID (Merkle root).
        """
        # Add source/license to each item
        enriched_batch = [
            {"data": item, "source_id": source_id, "license_id": license_id}
            for item in batch
        ]
        return self.observe(enriched_batch)

    def observe_mixed(self, batch: List[Dict]) -> str:
        """
        Observe a batch with mixed sources and licenses.

        Each item should have 'data', 'source_id', and 'license_id' keys.

        Args:
            batch: List of dicts with data, source_id, and license_id.

        Returns:
            Batch ID (Merkle root).
        """
        return self.observe(batch)


def simulate_data_sources():
    """
    Simulate data from multiple sources.

    In real scenarios, this would be loading from different datasets,
    APIs, databases, etc.
    """
    sources = {
        "wikipedia": {
            "license": "CC-BY-SA-4.0",
            "type": "public",
            "samples": [
                {"text": "Machine learning is a subset of artificial intelligence.", "id": "wiki_001"},
                {"text": "Neural networks are inspired by biological neurons.", "id": "wiki_002"},
                {"text": "Deep learning uses multiple layers of neural networks.", "id": "wiki_003"},
            ]
        },
        "common_crawl": {
            "license": "CC0-1.0",
            "type": "public",
            "samples": [
                {"text": "The quick brown fox jumps over the lazy dog.", "id": "cc_001"},
                {"text": "Lorem ipsum dolor sit amet consectetur.", "id": "cc_002"},
            ]
        },
        "vendor_dataset": {
            "license": "proprietary",
            "type": "purchased",
            "samples": [
                {"text": "Premium content from licensed vendor.", "id": "vendor_001"},
                {"text": "High-quality curated training examples.", "id": "vendor_002"},
                {"text": "Expert-annotated dataset samples.", "id": "vendor_003"},
                {"text": "Professionally verified data points.", "id": "vendor_004"},
            ]
        },
        "user_feedback": {
            "license": "CC-BY-4.0",
            "type": "user_generated",
            "samples": [
                {"text": "User submitted helpful feedback.", "id": "user_001"},
                {"text": "Community contribution from users.", "id": "user_002"},
            ]
        },
        "synthetic_data": {
            "license": "MIT",
            "type": "generated",
            "samples": [
                {"text": "Synthetically generated example 1.", "id": "syn_001"},
                {"text": "Synthetically generated example 2.", "id": "syn_002"},
                {"text": "Synthetically generated example 3.", "id": "syn_003"},
            ]
        },
    }
    return sources


def main():
    """Demonstrate multi-source training with provenance tracking."""

    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "multi_source_provenance.db")

        print("=" * 70)
        print("Origin Provenance Tracking - Multi-Source Training Example")
        print("=" * 70)

        # Initialize database
        db = ProvenanceDatabase(db_path)
        print(f"\n[Setup] Database: {db_path}")

        # Get simulated data sources
        data_sources = simulate_data_sources()

        # ─────────────────────────────────────────────────────────────
        # Step 1: Register all data sources
        # ─────────────────────────────────────────────────────────────
        print("\n[1] Registering data sources...")
        print("    " + "-" * 60)
        print(f"    {'Source':<20} {'Type':<15} {'License':<15} {'Samples':<10}")
        print("    " + "-" * 60)

        for source_name, source_info in data_sources.items():
            source = SourceRecord(
                source_id=source_name,
                source_type=source_info["type"],
                source_path=f"data://{source_name}",
                license_id=source_info["license"],
                first_seen=datetime.now().isoformat()
            )
            db.record_source(source)
            print(f"    {source_name:<20} {source_info['type']:<15} "
                  f"{source_info['license']:<15} {len(source_info['samples']):<10}")

        # ─────────────────────────────────────────────────────────────
        # Step 2: Start training session
        # ─────────────────────────────────────────────────────────────
        print("\n[2] Starting training session...")
        session = db.begin_session(config_hash="multi_source_model_v1")
        print(f"    Session ID: {session.session_id}")

        # ─────────────────────────────────────────────────────────────
        # Step 3: Process data from each source
        # ─────────────────────────────────────────────────────────────
        print("\n[3] Processing data from multiple sources...")

        # Best practice: Create a separate hook for each source
        # This ensures proper source and license tracking per sample
        total_samples = 0
        for source_name, source_info in data_sources.items():
            # Create a hook for this specific source
            from origin.hooks.base import BaseHook

            class SimpleHook(BaseHook):
                def _extract_samples(self, batch):
                    return [(item, {"index": i}) for i, item in enumerate(batch)]

            hook = SimpleHook(
                db=db,
                session_id=session.session_id,
                source_id=source_name,
                license_id=source_info["license"]
            )

            # Process this source's data
            batch = source_info["samples"]
            hook.observe(batch)
            total_samples += len(batch)
            print(f"    Processed {len(batch)} samples from {source_name} ({source_info['license']})")

        # ─────────────────────────────────────────────────────────────
        # Step 4: End session
        # ─────────────────────────────────────────────────────────────
        print("\n[4] Ending training session...")
        db.end_session(session.session_id, status="completed")

        batches = db.list_batches(session.session_id)
        print(f"    Total batches: {len(batches)}")
        print(f"    Total samples: {total_samples}")

        # ─────────────────────────────────────────────────────────────
        # Step 5: Analyze license composition
        # ─────────────────────────────────────────────────────────────
        print("\n[5] Analyzing license composition...")

        engine = QueryEngine(db)
        analyzer = LicenseAnalyzer()

        # License breakdown
        breakdown = engine.get_license_breakdown(session.session_id)
        total = sum(breakdown.values())

        print("\n    License Distribution:")
        print("    " + "-" * 50)
        for license_id, count in sorted(breakdown.items(), key=lambda x: -x[1]):
            pct = (count / total * 100) if total > 0 else 0
            bar = "█" * int(pct / 5)
            print(f"    {license_id:<20} {count:>4} ({pct:>5.1f}%) {bar}")

        # License analysis
        print("\n    License Compatibility Analysis:")
        print("    " + "-" * 50)

        analysis = analyzer.analyze_session(db, session.session_id)
        print(f"    Result License: {analysis.get('result_license', 'unknown')}")

        permissions = analysis.get("permissions", [])
        if permissions:
            print(f"    Permissions: {', '.join(sorted(permissions))}")

        restrictions = analysis.get("restrictions", [])
        if restrictions:
            print(f"    Restrictions: {', '.join(sorted(restrictions))}")

        conditions = analysis.get("conditions", [])
        if conditions:
            print(f"    Conditions: {', '.join(sorted(conditions))}")

        conflicts = analysis.get("conflicts", [])
        print(f"\n    Conflicts Detected: {len(conflicts)}")
        if conflicts:
            for conflict in conflicts:
                print(f"      - {conflict.get('license_a')} + {conflict.get('license_b')}: "
                      f"{conflict.get('reason')}")

        # ─────────────────────────────────────────────────────────────
        # Step 6: Source-level statistics
        # ─────────────────────────────────────────────────────────────
        print("\n[6] Source-level statistics...")

        source_breakdown = engine.get_source_breakdown(session.session_id)
        print("\n    Data sources contributing to this model:")
        print("    " + "-" * 60)
        for source_id, count in source_breakdown.items():
            source = db.get_source(source_id)
            if source:
                print(f"    - {source.source_id}: {source.license_id} "
                      f"({source.source_type}) - {count} samples")

        # ─────────────────────────────────────────────────────────────
        # Step 7: Important compliance notes
        # ─────────────────────────────────────────────────────────────
        print("\n[7] Compliance notes for this model...")

        notes = []

        # Check for copyleft
        copyleft_licenses = ["GPL-3.0-only", "GPL-2.0-only", "CC-BY-SA-4.0"]
        for lic in copyleft_licenses:
            if lic in breakdown:
                notes.append(f"Contains {lic} (copyleft) - derivatives must use same license")

        # Check for NC
        if "CC-BY-NC-4.0" in breakdown:
            notes.append("Contains CC-BY-NC-4.0 - model cannot be used commercially")

        # Check for proprietary
        if "proprietary" in breakdown:
            notes.append("Contains proprietary data - verify vendor license permits this use")

        # Check for unknown
        if "unknown" in breakdown:
            notes.append("Contains unknown license - investigate before deployment")

        if notes:
            print("\n    ⚠️  Important Notes:")
            for note in notes:
                print(f"      - {note}")
        else:
            print("\n    ✓ No special compliance notes")

        # ─────────────────────────────────────────────────────────────
        # Step 8: Generate provenance card
        # ─────────────────────────────────────────────────────────────
        print("\n[8] Generating provenance card...")

        generator = ProvenanceCardGenerator(db)
        card = generator.generate(session.session_id)

        card_path = os.path.join(temp_dir, "PROVENANCE_CARD.md")
        with open(card_path, "w") as f:
            f.write(card)

        print(f"    Card saved: {card_path}")
        print("\n    Preview:")
        print("    " + "-" * 50)
        for line in card.split("\n")[:15]:
            print(f"    {line}")
        print("    ...")

        db.close()

        print("\n" + "=" * 70)
        print("Multi-source training example completed!")
        print("=" * 70)
        print("""
Key patterns for multi-source training:
  1. Register each data source with its license upfront
  2. Use observe_with_source() for homogeneous batches
  3. Use observe_mixed() for heterogeneous batches
  4. Origin automatically tracks which licenses apply
  5. License analysis shows combined permissions/restrictions
  6. Provenance cards document all data sources for compliance

This enables full transparency about training data composition
for regulatory compliance and model documentation.
        """)


if __name__ == "__main__":
    main()
