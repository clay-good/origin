"""
Tests for Origin license tracking system.

Tests cover:
- LicenseRegistry operations
- LicenseAnalyzer session analysis
- Conflict detection
- License string parsing
- SPDX license definitions
"""

import os
import tempfile
import unittest
from pathlib import Path

from origin.core.license import (
    LicenseRegistry,
    LicenseAnalyzer,
    parse_license_string,
    SPDX_LICENSES,
)
from origin.core.record import LicenseRecord, SourceRecord
from origin.storage.database import ProvenanceDatabase
from origin.core.fingerprint import fingerprint_bytes
from tests.fixtures.sample_licenses import (
    TEST_MIT_LICENSE,
    TEST_GPL_LICENSE,
    TEST_APACHE_LICENSE,
    TEST_UNKNOWN_LICENSE,
)
from tests.fixtures.sample_data import (
    create_test_source_record,
    create_test_provenance_record,
    create_test_batch_record,
)


class TestSPDXLicenses(unittest.TestCase):
    """Tests for built-in SPDX license definitions."""

    def test_mit_license_defined(self):
        """MIT license should be in SPDX_LICENSES."""
        self.assertIn("MIT", SPDX_LICENSES)

    def test_gpl_license_defined(self):
        """GPL-3.0-only license should be in SPDX_LICENSES."""
        self.assertIn("GPL-3.0-only", SPDX_LICENSES)

    def test_apache_license_defined(self):
        """Apache-2.0 license should be in SPDX_LICENSES."""
        self.assertIn("Apache-2.0", SPDX_LICENSES)

    def test_all_licenses_have_required_fields(self):
        """All SPDX licenses should have required fields."""
        required_fields = ["license_name", "permissions", "restrictions", "conditions", "copyleft"]
        for license_id, license_data in SPDX_LICENSES.items():
            for field in required_fields:
                self.assertIn(
                    field, license_data,
                    f"License {license_id} missing field {field}"
                )


class TestLicenseRegistry(unittest.TestCase):
    """Tests for LicenseRegistry class."""

    def setUp(self):
        """Create a fresh registry for each test."""
        self.registry = LicenseRegistry()

    def test_get_builtin_license(self):
        """Getting built-in license should work."""
        result = self.registry.get("MIT")
        self.assertIsNotNone(result)
        self.assertEqual(result.license_name, "MIT License")

    def test_register_license(self):
        """Registering a license should make it retrievable."""
        custom = LicenseRecord(
            license_id="CUSTOM",
            license_name="Custom License",
            license_url=None,
            permissions=(),
            restrictions=(),
            conditions=(),
            copyleft=False,
        )
        self.registry.register(custom)
        retrieved = self.registry.get("CUSTOM")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.license_name, "Custom License")

    def test_get_unknown_license(self):
        """Getting unknown license should return None."""
        result = self.registry.get("NonExistentLicense")
        self.assertIsNone(result)

    def test_register_overwrites(self):
        """Registering same license ID should overwrite."""
        license1 = LicenseRecord(
            license_id="TEST",
            license_name="Test License 1",
            license_url=None,
            permissions=(),
            restrictions=(),
            conditions=(),
            copyleft=False,
        )
        license2 = LicenseRecord(
            license_id="TEST",
            license_name="Test License 2",
            license_url=None,
            permissions=(),
            restrictions=(),
            conditions=(),
            copyleft=False,
        )
        self.registry.register(license1)
        self.registry.register(license2)
        retrieved = self.registry.get("TEST")
        self.assertEqual(retrieved.license_name, "Test License 2")

    def test_list_all(self):
        """List all should return all registered licenses."""
        all_licenses = self.registry.list_all()
        # Should have at least the built-in SPDX licenses
        self.assertGreater(len(all_licenses), 5)
        self.assertIn("MIT", all_licenses)

    def test_case_insensitive_lookup(self):
        """Lookups should be case-insensitive."""
        result1 = self.registry.get("MIT")
        result2 = self.registry.get("mit")
        result3 = self.registry.get("Mit")
        self.assertIsNotNone(result1)
        self.assertIsNotNone(result2)
        self.assertIsNotNone(result3)

    def test_resolve_known_license(self):
        """Resolve should return known license."""
        result = self.registry.resolve("MIT")
        self.assertEqual(result.license_id, "MIT")

    def test_resolve_unknown_returns_unknown(self):
        """Resolve should return 'unknown' license for unknown identifiers."""
        result = self.registry.resolve("NonExistent")
        self.assertEqual(result.license_id.lower(), "unknown")


class TestParseLicenseString(unittest.TestCase):
    """Tests for parse_license_string function."""

    def test_parse_mit(self):
        """MIT should parse correctly."""
        result = parse_license_string("MIT")
        self.assertEqual(result, "MIT")

    def test_parse_mit_license(self):
        """MIT License variation should parse to MIT."""
        result = parse_license_string("MIT License")
        self.assertEqual(result, "MIT")

    def test_parse_apache(self):
        """Apache 2.0 should parse correctly."""
        result = parse_license_string("Apache 2.0")
        self.assertEqual(result, "Apache-2.0")

    def test_parse_gpl_variation(self):
        """GPL variations should parse correctly."""
        result = parse_license_string("GPLv3")
        self.assertEqual(result, "GPL-3.0-only")

    def test_parse_bsd(self):
        """BSD should default to BSD-3-Clause."""
        result = parse_license_string("BSD")
        self.assertEqual(result, "BSD-3-Clause")

    def test_parse_cc_by(self):
        """CC BY 4.0 should parse correctly."""
        result = parse_license_string("CC BY 4.0")
        self.assertEqual(result, "CC-BY-4.0")

    def test_parse_unknown_returns_lowercased(self):
        """Unknown license string should return lowercased."""
        result = parse_license_string("CustomLicense-1.0")
        self.assertEqual(result, "customlicense-1.0")

    def test_parse_empty_string(self):
        """Empty string should return empty string."""
        result = parse_license_string("")
        self.assertEqual(result, "")


class TestLicenseAnalyzer(unittest.TestCase):
    """Tests for LicenseAnalyzer class."""

    def setUp(self):
        """Create temporary database with test data."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"
        self.db = ProvenanceDatabase(self.db_path)
        self.session = self.db.begin_session("test_config")
        self.analyzer = LicenseAnalyzer()

    def tearDown(self):
        """Clean up."""
        self.db.close()
        if self.db_path.exists():
            os.remove(self.db_path)
        os.rmdir(self.temp_dir)

    def _add_sample_with_license(self, license_id: str, content: bytes = None):
        """Helper to add a sample with a specific license."""
        if content is None:
            content = f"sample_{license_id}".encode()

        source = SourceRecord(
            source_id=fingerprint_bytes(content + b"_source"),
            source_type="file",
            source_path=f"/data/{license_id}.csv",
            license_id=license_id,
            first_seen="2025-01-01T00:00:00Z",
        )
        self.db.record_source(source)

        sample = create_test_provenance_record(
            content=content,
            source_id=source.source_id,
        )
        self.db.record_sample(sample)

        batch = create_test_batch_record(
            session_id=self.session.session_id,
            sample_ids=(sample.sample_id,),
        )
        self.db.record_batch(batch)

        return sample

    def test_analyze_session_single_license(self):
        """Session with single license should report that license."""
        self._add_sample_with_license("MIT")
        result = self.analyzer.analyze_session(self.db, self.session.session_id)
        self.assertEqual(result.get("result_license"), "MIT")
        self.assertEqual(len(result.get("conflicts", [])), 0)

    def test_analyze_session_multiple_compatible(self):
        """Multiple compatible permissive licenses should not conflict."""
        self._add_sample_with_license("MIT", b"sample1")
        self._add_sample_with_license("Apache-2.0", b"sample2")
        result = self.analyzer.analyze_session(self.db, self.session.session_id)
        self.assertEqual(len(result.get("conflicts", [])), 0)

    def test_analyze_session_copyleft_mix(self):
        """Mixing different copyleft licenses should detect conflict."""
        self._add_sample_with_license("GPL-3.0-only", b"sample1")
        self._add_sample_with_license("GPL-2.0-only", b"sample2")
        result = self.analyzer.analyze_session(self.db, self.session.session_id)
        # Different copyleft licenses are incompatible
        self.assertGreater(len(result.get("conflicts", [])), 0)

    def test_analyze_session_permissive_with_copyleft(self):
        """Mixing permissive with copyleft should NOT be a conflict."""
        self._add_sample_with_license("MIT", b"sample1")
        self._add_sample_with_license("GPL-3.0-only", b"sample2")
        result = self.analyzer.analyze_session(self.db, self.session.session_id)
        # Permissive licenses can be incorporated into copyleft projects
        self.assertEqual(len(result.get("conflicts", [])), 0)

    def test_analyze_empty_session(self):
        """Empty session should have no conflicts."""
        result = self.analyzer.analyze_session(self.db, self.session.session_id)
        self.assertEqual(len(result.get("conflicts", [])), 0)

    def test_check_compatibility_same_license(self):
        """Same license should be compatible."""
        compatible, reason = self.analyzer.check_compatibility("MIT", "MIT")
        self.assertTrue(compatible)

    def test_check_compatibility_permissive_licenses(self):
        """Permissive licenses should be compatible."""
        compatible, reason = self.analyzer.check_compatibility("MIT", "Apache-2.0")
        self.assertTrue(compatible)

    def test_check_compatibility_copyleft_with_copyleft(self):
        """Different copyleft licenses should be incompatible."""
        compatible, reason = self.analyzer.check_compatibility("GPL-3.0-only", "GPL-2.0-only")
        self.assertFalse(compatible)
        self.assertIn("copyleft", reason.lower())


class TestLicensePropagation(unittest.TestCase):
    """Tests for license propagation analysis."""

    def setUp(self):
        """Create analyzer."""
        self.analyzer = LicenseAnalyzer()

    def test_propagate_single_license(self):
        """Single license should propagate unchanged."""
        result = self.analyzer.propagate(["MIT"])
        self.assertEqual(result["result_license"], "MIT")

    def test_propagate_multiple_same_license(self):
        """Multiple same licenses should be same."""
        result = self.analyzer.propagate(["MIT", "MIT", "MIT"])
        self.assertEqual(result["result_license"], "MIT")

    def test_propagate_mixed_licenses(self):
        """Mixed licenses should return 'mixed'."""
        result = self.analyzer.propagate(["MIT", "Apache-2.0"])
        self.assertEqual(result["result_license"], "mixed")

    def test_propagate_permissions_intersection(self):
        """Permissions should be intersection of all."""
        result = self.analyzer.propagate(["MIT", "Apache-2.0"])
        # Both have commercial, so it should be in permissions
        self.assertIn("commercial", result["permissions"])

    def test_propagate_restrictions_union(self):
        """Restrictions should be union of all."""
        result = self.analyzer.propagate(["MIT", "Apache-2.0"])
        # Apache-2.0 has no-trademark restriction
        self.assertIn("no-trademark", result["restrictions"])

    def test_propagate_empty_list(self):
        """Empty list should return unknown."""
        result = self.analyzer.propagate([])
        self.assertEqual(result["result_license"], "unknown")

    def test_propagate_detects_conflicts(self):
        """Propagation should detect conflicts between incompatible copyleft."""
        result = self.analyzer.propagate(["GPL-3.0-only", "GPL-2.0-only"])
        self.assertGreater(len(result["conflicts"]), 0)

    def test_propagate_no_conflict_permissive_copyleft(self):
        """Propagation should not flag permissive + copyleft as conflict."""
        result = self.analyzer.propagate(["MIT", "GPL-3.0-only"])
        self.assertEqual(len(result["conflicts"]), 0)


class TestLicenseRecordValidation(unittest.TestCase):
    """Tests for LicenseRecord validation."""

    def test_valid_license_record(self):
        """Valid license record should be created."""
        license_rec = LicenseRecord(
            license_id="TEST",
            license_name="Test License",
            license_url="https://example.com/license",
            permissions=("use", "modify"),
            restrictions=("no-warranty",),
            conditions=("attribution",),
            copyleft=False,
        )
        self.assertEqual(license_rec.license_id, "TEST")

    def test_license_record_empty_id_fails(self):
        """Empty license ID should raise ValueError."""
        with self.assertRaises(ValueError):
            LicenseRecord(
                license_id="",
                license_name="Test",
                license_url=None,
                permissions=(),
                restrictions=(),
                conditions=(),
                copyleft=False,
            )

    def test_license_record_to_dict(self):
        """LicenseRecord should convert to dict correctly."""
        license_dict = TEST_MIT_LICENSE.to_dict()
        self.assertEqual(license_dict["license_id"], "MIT")
        self.assertIsInstance(license_dict["permissions"], list)

    def test_license_record_from_dict(self):
        """LicenseRecord should be created from dict."""
        data = {
            "license_id": "TEST",
            "license_name": "Test",
            "license_url": None,
            "permissions": ["use"],
            "restrictions": [],
            "conditions": [],
            "copyleft": False,
        }
        license_rec = LicenseRecord.from_dict(data)
        self.assertEqual(license_rec.license_id, "TEST")
        self.assertEqual(license_rec.permissions, ("use",))


if __name__ == "__main__":
    unittest.main()
