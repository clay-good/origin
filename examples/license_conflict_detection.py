#!/usr/bin/env python3
"""
License Conflict Detection Example - Origin Provenance Tracking

This example demonstrates Origin's license conflict detection capabilities.
It shows how to:

1. Detect conflicts between incompatible licenses
2. Understand which license combinations are problematic
3. Analyze the effective license terms for mixed datasets
4. Generate compliance reports highlighting issues

This is essential for legal compliance when training on data from
multiple sources with different licensing terms.

No external dependencies required - uses only Python standard library.

Run this example:
    python examples/license_conflict_detection.py
"""

import os
import tempfile
from datetime import datetime

# Origin imports
from origin.storage.database import ProvenanceDatabase
from origin.core.fingerprint import fingerprint_sample, merkle_root
from origin.core.record import ProvenanceRecord, BatchRecord, SourceRecord
from origin.core.license import LicenseAnalyzer, LicenseRegistry
from origin.query.engine import QueryEngine


def create_scenario(db: ProvenanceDatabase, name: str, sources: list) -> str:
    """Create a training scenario with given sources."""
    # Register sources
    for source_id, license_id in sources:
        db.record_source(SourceRecord(
            source_id=source_id,
            source_type="dataset",
            source_path=f"/data/{source_id}",
            license_id=license_id,
            first_seen=datetime.now().isoformat()
        ))

    # Create session
    session = db.begin_session(config_hash=f"scenario_{name}")

    # Add samples from each source
    batch_idx = 0
    for source_id, license_id in sources:
        fingerprints = []
        for i in range(3):
            sample = f"{source_id}_sample_{i}".encode()
            fp = fingerprint_sample(sample)
            fingerprints.append(fp)

            db.record_sample(ProvenanceRecord(
                sample_id=fp,
                source_id=source_id,
                content_type="data",
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
    return session.session_id


def analyze_scenario(
    db: ProvenanceDatabase,
    session_id: str,
    scenario_name: str,
    description: str
):
    """Analyze a scenario for license conflicts."""
    print(f"\n{'─' * 60}")
    print(f"Scenario: {scenario_name}")
    print(f"{'─' * 60}")
    print(f"Description: {description}")

    analyzer = LicenseAnalyzer()
    engine = QueryEngine(db)

    # Get license breakdown
    breakdown = engine.get_license_breakdown(session_id)
    print(f"\nLicenses present:")
    for lic, count in breakdown.items():
        print(f"  - {lic}: {count} samples")

    # Analyze session
    analysis = analyzer.analyze_session(db, session_id)

    # Check for conflicts
    conflicts = analysis.get("conflicts", [])
    if conflicts:
        print(f"\n⚠️  CONFLICTS DETECTED: {len(conflicts)}")
        for conflict in conflicts:
            print(f"  ❌ {conflict.get('license_a')} + {conflict.get('license_b')}")
            print(f"     Reason: {conflict.get('reason')}")
    else:
        print(f"\n✓ No conflicts detected")

    # Show effective terms
    print(f"\nEffective License Terms:")
    print(f"  Result: {analysis.get('result_license', 'unknown')}")

    permissions = analysis.get("permissions", [])
    if permissions:
        print(f"  Permissions: {', '.join(sorted(permissions))}")
    else:
        print(f"  Permissions: None (most restrictive combination)")

    restrictions = analysis.get("restrictions", [])
    if restrictions:
        print(f"  Restrictions: {', '.join(sorted(restrictions))}")

    conditions = analysis.get("conditions", [])
    if conditions:
        print(f"  Conditions: {', '.join(sorted(conditions))}")

    return len(conflicts) > 0


def main():
    """Demonstrate license conflict detection."""

    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "license_conflicts.db")

        print("=" * 70)
        print("Origin Provenance Tracking - License Conflict Detection")
        print("=" * 70)

        # ─────────────────────────────────────────────────────────────
        # Show built-in licenses
        # ─────────────────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("BUILT-IN LICENSE DEFINITIONS")
        print("=" * 70)

        registry = LicenseRegistry()
        print("\nSupported Licenses:")
        print("-" * 60)
        print(f"{'License ID':<20} {'Copyleft':<10} {'Key Characteristics'}")
        print("-" * 60)

        license_info = [
            ("MIT", False, "Permissive, minimal restrictions"),
            ("Apache-2.0", False, "Permissive, patent grant, no trademark"),
            ("GPL-3.0-only", True, "Copyleft, disclose source required"),
            ("GPL-2.0-only", True, "Copyleft, older version"),
            ("CC-BY-4.0", False, "Attribution required"),
            ("CC-BY-SA-4.0", True, "Attribution + ShareAlike (copyleft)"),
            ("CC-BY-NC-4.0", False, "Attribution + NonCommercial"),
            ("CC0-1.0", False, "Public domain, no restrictions"),
            ("proprietary", False, "All rights reserved"),
        ]

        for lic_id, copyleft, desc in license_info:
            print(f"{lic_id:<20} {'Yes' if copyleft else 'No':<10} {desc}")

        # ─────────────────────────────────────────────────────────────
        # Scenario 1: Compatible permissive licenses
        # ─────────────────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("SCENARIO ANALYSIS")
        print("=" * 70)

        db = ProvenanceDatabase(db_path)

        session1 = create_scenario(db, "permissive_mix", [
            ("dataset_a", "MIT"),
            ("dataset_b", "Apache-2.0"),
            ("dataset_c", "BSD-3-Clause"),
        ])
        analyze_scenario(
            db, session1,
            "Permissive License Mix",
            "Combining MIT, Apache-2.0, and BSD-3-Clause datasets"
        )

        # ─────────────────────────────────────────────────────────────
        # Scenario 2: Permissive + Copyleft (compatible)
        # ─────────────────────────────────────────────────────────────
        db2 = ProvenanceDatabase(os.path.join(temp_dir, "scenario2.db"))
        session2 = create_scenario(db2, "permissive_copyleft", [
            ("mit_data", "MIT"),
            ("gpl_data", "GPL-3.0-only"),
        ])
        analyze_scenario(
            db2, session2,
            "Permissive + Copyleft",
            "MIT data combined with GPL-3.0 data (permissive can flow into copyleft)"
        )
        db2.close()

        # ─────────────────────────────────────────────────────────────
        # Scenario 3: Different copyleft versions (CONFLICT)
        # ─────────────────────────────────────────────────────────────
        db3 = ProvenanceDatabase(os.path.join(temp_dir, "scenario3.db"))
        session3 = create_scenario(db3, "copyleft_conflict", [
            ("gpl2_data", "GPL-2.0-only"),
            ("gpl3_data", "GPL-3.0-only"),
        ])
        analyze_scenario(
            db3, session3,
            "Copyleft Version Conflict",
            "GPL-2.0-only + GPL-3.0-only (different copyleft versions conflict)"
        )
        db3.close()

        # ─────────────────────────────────────────────────────────────
        # Scenario 4: Copyleft + Proprietary (CONFLICT)
        # ─────────────────────────────────────────────────────────────
        db4 = ProvenanceDatabase(os.path.join(temp_dir, "scenario4.db"))
        session4 = create_scenario(db4, "copyleft_proprietary", [
            ("gpl_data", "GPL-3.0-only"),
            ("vendor_data", "proprietary"),
        ])
        analyze_scenario(
            db4, session4,
            "Copyleft + Proprietary Conflict",
            "GPL-3.0 + proprietary data (copyleft conflicts with proprietary)"
        )
        db4.close()

        # ─────────────────────────────────────────────────────────────
        # Scenario 5: Creative Commons mix
        # ─────────────────────────────────────────────────────────────
        db5 = ProvenanceDatabase(os.path.join(temp_dir, "scenario5.db"))
        session5 = create_scenario(db5, "cc_mix", [
            ("cc_by_data", "CC-BY-4.0"),
            ("cc_nc_data", "CC-BY-NC-4.0"),
            ("cc0_data", "CC0-1.0"),
        ])
        analyze_scenario(
            db5, session5,
            "Creative Commons Mix",
            "CC-BY + CC-BY-NC + CC0 (NC restriction inherited, but no conflict)"
        )
        db5.close()

        # ─────────────────────────────────────────────────────────────
        # Scenario 6: CC-BY-SA + CC-BY-NC (different restrictions)
        # ─────────────────────────────────────────────────────────────
        db6 = ProvenanceDatabase(os.path.join(temp_dir, "scenario6.db"))
        session6 = create_scenario(db6, "cc_sa_nc", [
            ("cc_sa_data", "CC-BY-SA-4.0"),
            ("cc_nc_data", "CC-BY-NC-4.0"),
        ])
        analyze_scenario(
            db6, session6,
            "CC ShareAlike + NonCommercial",
            "CC-BY-SA (copyleft) + CC-BY-NC (non-commercial)"
        )
        db6.close()

        # ─────────────────────────────────────────────────────────────
        # Scenario 7: Unknown license present
        # ─────────────────────────────────────────────────────────────
        db7 = ProvenanceDatabase(os.path.join(temp_dir, "scenario7.db"))
        session7 = create_scenario(db7, "unknown_license", [
            ("known_data", "MIT"),
            ("scraped_data", "unknown"),
        ])
        analyze_scenario(
            db7, session7,
            "Unknown License Present",
            "MIT + unknown (flags for review but not a hard conflict)"
        )
        db7.close()

        # ─────────────────────────────────────────────────────────────
        # Summary
        # ─────────────────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("LICENSE COMPATIBILITY SUMMARY")
        print("=" * 70)

        print("""
┌─────────────────────────────────────────────────────────────────────┐
│                    LICENSE COMPATIBILITY MATRIX                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ALWAYS COMPATIBLE:                                                   │
│  ✓ Permissive + Permissive (MIT, Apache, BSD)                        │
│  ✓ Permissive + Copyleft (permissive flows into copyleft)            │
│  ✓ Public Domain + Anything (CC0, Unlicense)                         │
│                                                                       │
│  CONFLICTS:                                                           │
│  ✗ GPL-2.0-only + GPL-3.0-only (different copyleft versions)         │
│  ✗ GPL + Proprietary (copyleft requires open source)                 │
│  ✗ CC-BY-SA + Proprietary (ShareAlike is copyleft)                   │
│                                                                       │
│  RESTRICTIONS (not conflicts, but inherited):                         │
│  ⚠ CC-BY-NC restricts commercial use                                 │
│  ⚠ Unknown licenses require manual review                            │
│  ⚠ Proprietary terms must be verified with vendor                    │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘

KEY PRINCIPLES:

1. Permissive licenses combine freely
   - MIT, Apache-2.0, BSD variants are all compatible
   - The combined work inherits all attribution requirements

2. Copyleft licenses propagate requirements
   - GPL/LGPL/CC-BY-SA require derivatives to use same license
   - Permissive can flow INTO copyleft, not the reverse

3. Different copyleft versions often conflict
   - GPL-2.0-only and GPL-3.0-only cannot be combined
   - "GPL-3.0-or-later" is more flexible than "-only"

4. Proprietary conflicts with copyleft
   - Cannot combine GPL data with proprietary data
   - Permissive data CAN be used with proprietary

5. Non-commercial restrictions are inherited
   - CC-BY-NC data makes the entire model NC
   - This is a restriction, not a conflict

WHEN TO SEEK LEGAL REVIEW:

- Any conflicts detected by Origin
- Significant amounts of unknown-license data
- Proprietary data included in the training set
- Model will be distributed commercially
- Regulatory compliance required (EU AI Act)
        """)

        db.close()

        print("=" * 70)
        print("License Conflict Detection Example Complete!")
        print("=" * 70)


if __name__ == "__main__":
    main()
