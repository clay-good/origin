"""
Tests for Origin query engine.

Tests cover:
- License queries
- Source pattern matching
- Batch queries
- Sample tracing
- Session comparison
- Edge cases
"""

import os
import tempfile
import unittest
from pathlib import Path

from origin.query.engine import QueryEngine
from origin.storage.database import ProvenanceDatabase
from origin.core.record import SourceRecord, BatchRecord
from origin.core.fingerprint import fingerprint_bytes, merkle_root
from tests.fixtures.sample_data import (
    create_test_source_record,
    create_test_provenance_record,
)
from tests.fixtures.sample_licenses import TEST_MIT_LICENSE, TEST_GPL_LICENSE


class QueryTestCase(unittest.TestCase):
    """Base class for query tests with database setup."""

    def setUp(self):
        """Create temporary database with test data."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"
        self.db = ProvenanceDatabase(self.db_path)
        self.session = self.db.begin_session("test_config")
        self.engine = QueryEngine(self.db)

    def tearDown(self):
        """Clean up."""
        self.db.close()
        if self.db_path.exists():
            os.remove(self.db_path)
        os.rmdir(self.temp_dir)

    def _add_sample_with_license(
        self,
        license_id: str,
        content: bytes = None,
        source_path: str = None,
    ):
        """Helper to add a sample with specific license."""
        if content is None:
            content = f"sample_{license_id}_{id(self)}".encode()
        if source_path is None:
            source_path = f"/data/{license_id}_{fingerprint_bytes(content)[:8]}.csv"

        source = SourceRecord(
            source_id=fingerprint_bytes(source_path.encode()),
            source_type="file",
            source_path=source_path,
            license_id=license_id,
            first_seen="2025-01-01T00:00:00Z",
        )
        self.db.record_source(source)

        sample = create_test_provenance_record(
            content=content,
            source_id=source.source_id,
        )
        self.db.record_sample(sample)

        batch = BatchRecord(
            batch_id=fingerprint_bytes(f"batch_{content}".encode()),
            session_id=self.session.session_id,
            batch_index=0,
            sample_count=1,
            created_at="2025-01-01T00:00:00Z",
            sample_ids=(sample.sample_id,),
        )
        self.db.record_batch(batch)

        return sample, source


class TestFindSamplesByLicense(QueryTestCase):
    """Tests for find_samples_by_license method."""

    def test_find_existing_license(self):
        """Should find samples with matching license."""
        self._add_sample_with_license("MIT")
        samples = self.engine.find_samples_by_license("MIT")
        self.assertEqual(len(samples), 1)

    def test_find_nonexistent_license(self):
        """Should return empty for nonexistent license."""
        self._add_sample_with_license("MIT")
        samples = self.engine.find_samples_by_license("GPL-3.0")
        self.assertEqual(len(samples), 0)

    def test_find_multiple_samples(self):
        """Should find all samples with license."""
        self._add_sample_with_license("MIT", b"sample1")
        self._add_sample_with_license("MIT", b"sample2")
        self._add_sample_with_license("Apache-2.0", b"sample3")
        samples = self.engine.find_samples_by_license("MIT")
        self.assertEqual(len(samples), 2)

    def test_filter_by_session(self):
        """Should filter by session when specified."""
        self._add_sample_with_license("MIT", b"sample1")

        # Create another session with MIT sample
        session2 = self.db.begin_session("other_config")
        source = SourceRecord(
            source_id="other_source",
            source_type="file",
            source_path="/other/path.csv",
            license_id="MIT",
            first_seen="2025-01-01T00:00:00Z",
        )
        self.db.record_source(source)
        sample = create_test_provenance_record(
            content=b"other_sample",
            source_id=source.source_id,
        )
        self.db.record_sample(sample)
        batch = BatchRecord(
            batch_id=fingerprint_bytes(b"other_batch"),
            session_id=session2.session_id,
            batch_index=0,
            sample_count=1,
            created_at="2025-01-01T00:00:00Z",
            sample_ids=(sample.sample_id,),
        )
        self.db.record_batch(batch)

        # Should find only one when filtering by session
        samples = self.engine.find_samples_by_license(
            "MIT", session_id=self.session.session_id
        )
        self.assertEqual(len(samples), 1)


class TestFindSamplesBySource(QueryTestCase):
    """Tests for find_samples_by_source method."""

    def test_find_exact_path(self):
        """Should find samples with exact path match."""
        self._add_sample_with_license("MIT", source_path="/data/train.csv")
        samples = self.engine.find_samples_by_source("/data/train.csv")
        self.assertEqual(len(samples), 1)

    def test_find_glob_pattern(self):
        """Should find samples matching glob pattern."""
        self._add_sample_with_license("MIT", b"s1", source_path="/data/file1.csv")
        self._add_sample_with_license("MIT", b"s2", source_path="/data/file2.csv")
        self._add_sample_with_license("MIT", b"s3", source_path="/other/file3.csv")

        samples = self.engine.find_samples_by_source("/data/*.csv")
        self.assertEqual(len(samples), 2)

    def test_no_match(self):
        """Should return empty for no matches."""
        self._add_sample_with_license("MIT", source_path="/data/train.csv")
        samples = self.engine.find_samples_by_source("/other/*.csv")
        self.assertEqual(len(samples), 0)


class TestCheckLicensePresence(QueryTestCase):
    """Tests for check_license_presence method."""

    def test_license_present(self):
        """Should return True when license is present."""
        self._add_sample_with_license("MIT")
        result = self.engine.check_license_presence(
            self.session.session_id, "MIT"
        )
        self.assertTrue(result)

    def test_license_not_present(self):
        """Should return False when license is not present."""
        self._add_sample_with_license("MIT")
        result = self.engine.check_license_presence(
            self.session.session_id, "GPL-3.0"
        )
        self.assertFalse(result)

    def test_empty_session(self):
        """Should return False for empty session."""
        result = self.engine.check_license_presence(
            self.session.session_id, "MIT"
        )
        self.assertFalse(result)


class TestGetLicenseBreakdown(QueryTestCase):
    """Tests for get_license_breakdown method."""

    def test_single_license(self):
        """Should count samples for single license."""
        self._add_sample_with_license("MIT", b"s1")
        self._add_sample_with_license("MIT", b"s2")
        breakdown = self.engine.get_license_breakdown(self.session.session_id)
        self.assertEqual(breakdown.get("MIT"), 2)

    def test_multiple_licenses(self):
        """Should count samples for each license."""
        self._add_sample_with_license("MIT", b"s1")
        self._add_sample_with_license("MIT", b"s2")
        self._add_sample_with_license("Apache-2.0", b"s3")
        breakdown = self.engine.get_license_breakdown(self.session.session_id)
        self.assertEqual(breakdown.get("MIT"), 2)
        self.assertEqual(breakdown.get("Apache-2.0"), 1)

    def test_empty_session(self):
        """Empty session should return empty breakdown."""
        breakdown = self.engine.get_license_breakdown(self.session.session_id)
        self.assertEqual(len(breakdown), 0)


class TestFindBatchesWithLicense(QueryTestCase):
    """Tests for find_batches_with_license method."""

    def test_find_batches(self):
        """Should find batches containing license."""
        self._add_sample_with_license("MIT")
        batches = self.engine.find_batches_with_license(
            self.session.session_id, "MIT"
        )
        self.assertEqual(len(batches), 1)

    def test_no_batches(self):
        """Should return empty when no matching batches."""
        self._add_sample_with_license("MIT")
        batches = self.engine.find_batches_with_license(
            self.session.session_id, "GPL-3.0"
        )
        self.assertEqual(len(batches), 0)


class TestTraceSample(QueryTestCase):
    """Tests for trace_sample method."""

    def test_trace_existing_sample(self):
        """Should trace sample back to source."""
        sample, source = self._add_sample_with_license("MIT")
        trace = self.engine.trace_sample(sample.sample_id)

        self.assertIsNotNone(trace["sample"])
        self.assertIsNotNone(trace["source"])
        self.assertEqual(trace["sample"]["sample_id"], sample.sample_id)
        self.assertEqual(trace["source"]["source_id"], source.source_id)

    def test_trace_nonexistent_sample(self):
        """Should return None for nonexistent sample."""
        trace = self.engine.trace_sample("nonexistent_sample_id")
        self.assertIsNone(trace["sample"])

    def test_trace_includes_batches(self):
        """Trace should include batches containing sample."""
        sample, _ = self._add_sample_with_license("MIT")
        trace = self.engine.trace_sample(sample.sample_id)
        self.assertGreater(len(trace["batches"]), 0)

    def test_trace_includes_sessions(self):
        """Trace should include sessions containing sample."""
        sample, _ = self._add_sample_with_license("MIT")
        trace = self.engine.trace_sample(sample.sample_id)
        self.assertIn(self.session.session_id, trace["sessions"])


class TestFindConflicts(QueryTestCase):
    """Tests for find_conflicts method."""

    def test_find_conflicts_for_session(self):
        """Should find conflicts for specific session."""
        self.db.record_conflict(
            self.session.session_id, "MIT", "GPL-3.0", "copyleft_mix"
        )
        conflicts = self.engine.find_conflicts(self.session.session_id)
        self.assertEqual(len(conflicts), 1)

    def test_find_all_conflicts(self):
        """Should find all conflicts when no session specified."""
        self.db.record_conflict(
            self.session.session_id, "MIT", "GPL-3.0", "copyleft_mix"
        )
        session2 = self.db.begin_session("other")
        self.db.record_conflict(
            session2.session_id, "Apache-2.0", "AGPL-3.0", "copyleft_mix"
        )

        conflicts = self.engine.find_conflicts()
        self.assertEqual(len(conflicts), 2)

    def test_no_conflicts(self):
        """Should return empty when no conflicts."""
        conflicts = self.engine.find_conflicts(self.session.session_id)
        self.assertEqual(len(conflicts), 0)


class TestCompareSessionsQuery(QueryTestCase):
    """Tests for compare_sessions method."""

    def test_compare_overlapping_sessions(self):
        """Should detect shared samples between sessions."""
        # Add sample to first session
        sample, source = self._add_sample_with_license("MIT", b"shared")

        # Create second session with same sample
        session2 = self.db.begin_session("config2")
        batch2 = BatchRecord(
            batch_id=fingerprint_bytes(b"batch2"),
            session_id=session2.session_id,
            batch_index=0,
            sample_count=1,
            created_at="2025-01-01T00:00:00Z",
            sample_ids=(sample.sample_id,),
        )
        self.db.record_batch(batch2)

        comparison = self.engine.compare_sessions(
            self.session.session_id, session2.session_id
        )
        self.assertEqual(comparison["shared_samples"], 1)

    def test_compare_disjoint_sessions(self):
        """Should detect unique samples in each session."""
        self._add_sample_with_license("MIT", b"sample1")

        # Create second session with different sample
        session2 = self.db.begin_session("config2")
        source2 = SourceRecord(
            source_id="source2",
            source_type="file",
            source_path="/data/other.csv",
            license_id="Apache-2.0",
            first_seen="2025-01-01T00:00:00Z",
        )
        self.db.record_source(source2)
        sample2 = create_test_provenance_record(
            content=b"sample2", source_id="source2"
        )
        self.db.record_sample(sample2)
        batch2 = BatchRecord(
            batch_id=fingerprint_bytes(b"batch2"),
            session_id=session2.session_id,
            batch_index=0,
            sample_count=1,
            created_at="2025-01-01T00:00:00Z",
            sample_ids=(sample2.sample_id,),
        )
        self.db.record_batch(batch2)

        comparison = self.engine.compare_sessions(
            self.session.session_id, session2.session_id
        )
        self.assertEqual(comparison["shared_samples"], 0)
        self.assertEqual(comparison["unique_to_a"], 1)
        self.assertEqual(comparison["unique_to_b"], 1)


class TestExportResult(QueryTestCase):
    """Tests for export_result method."""

    def test_export_json(self):
        """Should export result as JSON."""
        data = {"key": "value", "number": 42}
        result = self.engine.export_result(data, format="json")
        self.assertIn('"key"', result)
        self.assertIn('"value"', result)

    def test_export_jsonl(self):
        """Should export list as JSONL."""
        data = [{"a": 1}, {"b": 2}]
        result = self.engine.export_result(data, format="jsonl")
        lines = result.strip().split("\n")
        self.assertEqual(len(lines), 2)

    def test_export_invalid_format(self):
        """Should raise for invalid format."""
        with self.assertRaises(ValueError):
            self.engine.export_result({}, format="invalid")

    def test_export_records(self):
        """Should convert record objects to dicts."""
        sample, _ = self._add_sample_with_license("MIT")
        samples = self.engine.find_samples_by_license("MIT")
        result = self.engine.export_result(samples, format="json")
        self.assertIn("sample_id", result)


if __name__ == "__main__":
    unittest.main()
