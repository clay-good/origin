"""
Tests for Origin provenance card generation.

Tests cover:
- ProvenanceCardGenerator
- Template formatting helpers
- Card content and structure
- Edge cases (empty sessions, missing data)
"""

import os
import tempfile
import unittest
from pathlib import Path

from origin.cards.generator import ProvenanceCardGenerator
from origin.cards.templates import (
    format_timestamp,
    format_bytes,
    format_license_summary,
    format_table,
    truncate_hash,
)
from origin.storage.database import ProvenanceDatabase
from origin.core.record import SourceRecord, BatchRecord
from origin.core.fingerprint import fingerprint_bytes, merkle_root
from tests.fixtures.sample_data import (
    create_test_source_record,
    create_test_provenance_record,
)
from tests.fixtures.sample_licenses import TEST_MIT_LICENSE


class TestFormatTimestamp(unittest.TestCase):
    """Tests for format_timestamp helper."""

    def test_iso_format(self):
        """Should format ISO timestamp to readable string."""
        result = format_timestamp("2025-01-15T14:30:00+00:00")
        self.assertIn("January", result)
        self.assertIn("2025", result)

    def test_handles_utc(self):
        """Should handle UTC timestamps."""
        result = format_timestamp("2025-06-15T10:00:00Z")
        self.assertIn("June", result)
        self.assertIn("2025", result)

    def test_invalid_format(self):
        """Should handle invalid timestamp gracefully."""
        result = format_timestamp("not-a-timestamp")
        # Should return something, not crash
        self.assertIsInstance(result, str)


class TestFormatBytes(unittest.TestCase):
    """Tests for format_bytes helper."""

    def test_bytes(self):
        """Small values should show as bytes."""
        result = format_bytes(100)
        self.assertIn("B", result)

    def test_kilobytes(self):
        """KB values should format correctly."""
        result = format_bytes(1024)
        self.assertIn("KB", result)

    def test_megabytes(self):
        """MB values should format correctly."""
        result = format_bytes(1024 * 1024)
        self.assertIn("MB", result)

    def test_gigabytes(self):
        """GB values should format correctly."""
        result = format_bytes(1024 * 1024 * 1024)
        self.assertIn("GB", result)

    def test_zero_bytes(self):
        """Zero bytes should format correctly."""
        result = format_bytes(0)
        self.assertIn("0", result)


class TestTruncateHash(unittest.TestCase):
    """Tests for truncate_hash helper."""

    def test_truncates_long_hash(self):
        """Long hash should be truncated."""
        long_hash = "a" * 64
        result = truncate_hash(long_hash, length=16)
        self.assertEqual(len(result), 19)  # 16 chars + "..."
        self.assertTrue(result.endswith("..."))

    def test_short_hash_unchanged(self):
        """Short hash should not be changed."""
        short_hash = "abc123"
        result = truncate_hash(short_hash, length=16)
        self.assertEqual(result, short_hash)


class TestFormatTable(unittest.TestCase):
    """Tests for format_table helper."""

    def test_simple_table(self):
        """Should create markdown table."""
        headers = ["Name", "Value"]
        rows = [["foo", "1"], ["bar", "2"]]
        result = format_table(headers, rows)
        self.assertIn("| Name | Value |", result)
        self.assertIn("| foo | 1 |", result)
        self.assertIn("| bar | 2 |", result)

    def test_empty_table(self):
        """Empty table should have headers only."""
        headers = ["A", "B"]
        rows = []
        result = format_table(headers, rows)
        self.assertIn("| A | B |", result)


class TestFormatLicenseSummary(unittest.TestCase):
    """Tests for format_license_summary helper."""

    def test_single_license(self):
        """Single license should be formatted."""
        analysis = {
            "result_license": "MIT",
            "permissions": ("commercial", "modification"),
            "restrictions": ("no-warranty",),
            "conditions": ("attribution",),
            "conflicts": [],
        }
        result = format_license_summary(analysis)
        self.assertIn("MIT", result)
        self.assertIn("commercial", result)

    def test_multiple_licenses(self):
        """Multiple licenses with conflict should all appear."""
        analysis = {
            "result_license": "mixed",
            "permissions": ("commercial",),
            "restrictions": (),
            "conditions": (),
            "conflicts": [
                {"license_a": "MIT", "license_b": "GPL-3.0", "reason": "copyleft mix"}
            ],
        }
        result = format_license_summary(analysis)
        self.assertIn("MIT", result)
        self.assertIn("GPL-3.0", result)
        self.assertIn("mixed", result)


class TestProvenanceCardGenerator(unittest.TestCase):
    """Tests for ProvenanceCardGenerator class."""

    def setUp(self):
        """Create temporary database with test data."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"
        self.db = ProvenanceDatabase(self.db_path)
        self.session = self.db.begin_session("test_config_hash")

        # Add test data
        self.source = create_test_source_record()
        self.db.record_source(self.source)
        self.db.record_license(TEST_MIT_LICENSE)

        # Add samples and batch
        sample_ids = []
        for i in range(5):
            sample = create_test_provenance_record(
                content=f"sample_{i}".encode(),
                source_id=self.source.source_id,
            )
            self.db.record_sample(sample)
            sample_ids.append(sample.sample_id)

        batch = BatchRecord(
            batch_id=merkle_root(sample_ids),
            session_id=self.session.session_id,
            batch_index=0,
            sample_count=len(sample_ids),
            created_at="2025-01-15T14:30:00+00:00",
            sample_ids=tuple(sample_ids),
        )
        self.db.record_batch(batch)

        self.generator = ProvenanceCardGenerator(self.db)

    def tearDown(self):
        """Clean up."""
        self.db.close()
        if self.db_path.exists():
            os.remove(self.db_path)
        os.rmdir(self.temp_dir)

    def test_generate_returns_string(self):
        """Generate should return markdown string."""
        card = self.generator.generate(self.session.session_id)
        self.assertIsInstance(card, str)

    def test_card_has_session_info(self):
        """Card should contain session information."""
        card = self.generator.generate(self.session.session_id)
        self.assertIn("Session", card)
        self.assertIn(self.session.session_id, card)

    def test_card_has_source_info(self):
        """Card should contain source information."""
        card = self.generator.generate(self.session.session_id)
        self.assertIn("Source", card)

    def test_card_has_license_info(self):
        """Card should contain license information."""
        card = self.generator.generate(self.session.session_id)
        self.assertIn("License", card)
        self.assertIn("MIT", card)

    def test_card_has_statistics(self):
        """Card should contain sample statistics."""
        card = self.generator.generate(self.session.session_id)
        self.assertIn("Sample", card)
        self.assertIn("5", card)  # 5 samples

    def test_card_has_batch_summary(self):
        """Card should contain batch summary."""
        card = self.generator.generate(self.session.session_id)
        self.assertIn("Batch", card)

    def test_card_is_markdown(self):
        """Card should be valid markdown."""
        card = self.generator.generate(self.session.session_id)
        # Check for markdown elements
        self.assertIn("#", card)  # Headers
        self.assertIn("|", card)  # Tables

    def test_nonexistent_session_raises(self):
        """Generating card for nonexistent session should raise."""
        with self.assertRaises(ValueError):
            self.generator.generate("nonexistent-session-id")


class TestProvenanceCardEmptySession(unittest.TestCase):
    """Tests for card generation with empty session."""

    def setUp(self):
        """Create temporary database with empty session."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"
        self.db = ProvenanceDatabase(self.db_path)
        self.session = self.db.begin_session("empty_config")
        self.generator = ProvenanceCardGenerator(self.db)

    def tearDown(self):
        """Clean up."""
        self.db.close()
        if self.db_path.exists():
            os.remove(self.db_path)
        os.rmdir(self.temp_dir)

    def test_empty_session_generates_card(self):
        """Empty session should still generate a card."""
        card = self.generator.generate(self.session.session_id)
        self.assertIsInstance(card, str)
        self.assertIn("Session", card)

    def test_empty_session_shows_zero_samples(self):
        """Empty session should show zero samples."""
        card = self.generator.generate(self.session.session_id)
        self.assertIn("0", card)


class TestProvenanceCardWithConflicts(unittest.TestCase):
    """Tests for card generation with license conflicts."""

    def setUp(self):
        """Create temporary database with conflicting licenses."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"
        self.db = ProvenanceDatabase(self.db_path)
        self.session = self.db.begin_session("conflict_config")

        # Add a conflict
        self.db.record_conflict(
            session_id=self.session.session_id,
            license_a="MIT",
            license_b="GPL-3.0",
            conflict_type="copyleft_mix",
        )

        self.generator = ProvenanceCardGenerator(self.db)

    def tearDown(self):
        """Clean up."""
        self.db.close()
        if self.db_path.exists():
            os.remove(self.db_path)
        os.rmdir(self.temp_dir)

    def test_conflicts_shown_in_card(self):
        """License conflicts should appear in card."""
        card = self.generator.generate(self.session.session_id)
        # Card should mention conflicts
        self.assertIn("conflict", card.lower())


if __name__ == "__main__":
    unittest.main()
