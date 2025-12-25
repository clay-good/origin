"""
Tests for Origin storage layer (ProvenanceDatabase).

Tests cover:
- Database creation and initialization
- Session lifecycle management
- Record storage and retrieval
- Read-only mode
- SQL injection prevention
- Batch operations
- Statistics
"""

import os
import tempfile
import unittest
from pathlib import Path

from origin.storage.database import ProvenanceDatabase
from origin.core.record import (
    ProvenanceRecord,
    SessionRecord,
    BatchRecord,
    SourceRecord,
    LicenseRecord,
)
from origin.core.fingerprint import fingerprint_bytes, merkle_root
from tests.fixtures.sample_data import (
    create_test_source_record,
    create_test_provenance_record,
    create_test_batch_record,
)
from tests.fixtures.sample_licenses import TEST_MIT_LICENSE, TEST_GPL_LICENSE


class TestProvenanceDatabaseCreation(unittest.TestCase):
    """Tests for database creation and initialization."""

    def setUp(self):
        """Create temporary directory for test databases."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"

    def tearDown(self):
        """Clean up temporary files."""
        if self.db_path.exists():
            os.remove(self.db_path)
        os.rmdir(self.temp_dir)

    def test_create_database(self):
        """Database file should be created on initialization."""
        db = ProvenanceDatabase(self.db_path)
        self.assertTrue(self.db_path.exists())
        db.close()

    def test_create_database_with_path_object(self):
        """Should accept Path objects."""
        db = ProvenanceDatabase(Path(self.db_path))
        self.assertTrue(self.db_path.exists())
        db.close()

    def test_context_manager(self):
        """Database should work as context manager."""
        with ProvenanceDatabase(self.db_path) as db:
            self.assertTrue(self.db_path.exists())
            session = db.begin_session("test")
            self.assertIsNotNone(session)

    def test_schema_version_set(self):
        """Schema version should be set on creation."""
        with ProvenanceDatabase(self.db_path) as db:
            stats = db.get_statistics()
            self.assertEqual(stats['schema_version'], 1)


class TestSessionLifecycle(unittest.TestCase):
    """Tests for session management."""

    def setUp(self):
        """Create temporary database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"

    def tearDown(self):
        """Clean up temporary files."""
        if self.db_path.exists():
            os.remove(self.db_path)
        os.rmdir(self.temp_dir)

    def test_begin_session(self):
        """Begin session should create a new session record."""
        with ProvenanceDatabase(self.db_path) as db:
            session = db.begin_session("config_hash_123")
            self.assertIsNotNone(session.session_id)
            self.assertEqual(session.status, "running")
            self.assertEqual(session.config_hash, "config_hash_123")

    def test_begin_session_with_metadata(self):
        """Session should store metadata."""
        with ProvenanceDatabase(self.db_path) as db:
            session = db.begin_session("hash", metadata={"key": "value"})
            retrieved = db.get_session(session.session_id)
            self.assertEqual(retrieved.metadata, {"key": "value"})

    def test_end_session_completed(self):
        """Ending session should update status to completed."""
        with ProvenanceDatabase(self.db_path) as db:
            session = db.begin_session("config_hash_123")
            db.end_session(session.session_id, "completed")
            retrieved = db.get_session(session.session_id)
            self.assertEqual(retrieved.status, "completed")

    def test_end_session_failed(self):
        """Ending session should update status to failed."""
        with ProvenanceDatabase(self.db_path) as db:
            session = db.begin_session("config_hash_123")
            db.end_session(session.session_id, "failed")
            retrieved = db.get_session(session.session_id)
            self.assertEqual(retrieved.status, "failed")

    def test_end_session_invalid_status(self):
        """Invalid status should raise ValueError."""
        with ProvenanceDatabase(self.db_path) as db:
            session = db.begin_session("config_hash_123")
            with self.assertRaises(ValueError):
                db.end_session(session.session_id, "invalid")

    def test_list_sessions(self):
        """List sessions should return all sessions."""
        with ProvenanceDatabase(self.db_path) as db:
            db.begin_session("hash1")
            db.begin_session("hash2")
            db.begin_session("hash3")
            sessions = db.list_sessions()
            self.assertEqual(len(sessions), 3)

    def test_list_sessions_limit(self):
        """List sessions should respect limit parameter."""
        with ProvenanceDatabase(self.db_path) as db:
            for i in range(5):
                db.begin_session(f"hash{i}")
            sessions = db.list_sessions(limit=3)
            self.assertEqual(len(sessions), 3)


class TestRecordStorage(unittest.TestCase):
    """Tests for recording and retrieving records."""

    def setUp(self):
        """Create temporary database with session."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"
        self.db = ProvenanceDatabase(self.db_path)
        self.session = self.db.begin_session("test_config")

    def tearDown(self):
        """Clean up."""
        self.db.close()
        if self.db_path.exists():
            os.remove(self.db_path)
        os.rmdir(self.temp_dir)

    def test_record_source(self):
        """Source should be recorded and retrievable."""
        source = create_test_source_record()
        self.db.record_source(source)
        retrieved = self.db.get_source(source.source_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.source_path, source.source_path)

    def test_record_source_idempotent(self):
        """Recording same source twice should not fail."""
        source = create_test_source_record()
        self.db.record_source(source)
        self.db.record_source(source)  # Should not raise
        retrieved = self.db.get_source(source.source_id)
        self.assertIsNotNone(retrieved)

    def test_record_sample(self):
        """Sample should be recorded and retrievable."""
        source = create_test_source_record()
        self.db.record_source(source)
        sample = create_test_provenance_record(source_id=source.source_id)
        self.db.record_sample(sample)
        retrieved = self.db.get_sample(sample.sample_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.byte_size, sample.byte_size)

    def test_record_batch(self):
        """Batch should be recorded with sample associations."""
        source = create_test_source_record()
        self.db.record_source(source)

        # Create and record samples
        sample_ids = []
        for i in range(3):
            sample = create_test_provenance_record(
                content=f"sample_{i}".encode(),
                source_id=source.source_id,
            )
            self.db.record_sample(sample)
            sample_ids.append(sample.sample_id)

        # Create and record batch
        batch = create_test_batch_record(
            session_id=self.session.session_id,
            sample_ids=tuple(sample_ids),
        )
        self.db.record_batch(batch)

        # Retrieve and verify
        retrieved = self.db.get_batch(batch.batch_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.sample_count, 3)
        self.assertEqual(len(retrieved.sample_ids), 3)

    def test_record_license(self):
        """License should be recorded and retrievable."""
        self.db.record_license(TEST_MIT_LICENSE)
        retrieved = self.db.get_license(TEST_MIT_LICENSE.license_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.license_name, "MIT License")

    def test_record_conflict(self):
        """Conflict should be recorded."""
        self.db.record_conflict(
            session_id=self.session.session_id,
            license_a="MIT",
            license_b="GPL-3.0",
            conflict_type="copyleft_mix",
        )
        conflicts = self.db.list_conflicts(self.session.session_id)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]['license_a'], "MIT")

    def test_list_batches(self):
        """List batches should return session batches in order."""
        source = create_test_source_record()
        self.db.record_source(source)

        for i in range(3):
            sample = create_test_provenance_record(
                content=f"batch_{i}".encode(),
                source_id=source.source_id,
            )
            self.db.record_sample(sample)
            batch = BatchRecord(
                batch_id=fingerprint_bytes(f"batch_{i}".encode()),
                session_id=self.session.session_id,
                batch_index=i,
                sample_count=1,
                created_at="2025-01-01T00:00:00Z",
                sample_ids=(sample.sample_id,),
            )
            self.db.record_batch(batch)

        batches = self.db.list_batches(self.session.session_id)
        self.assertEqual(len(batches), 3)
        self.assertEqual(batches[0].batch_index, 0)
        self.assertEqual(batches[1].batch_index, 1)
        self.assertEqual(batches[2].batch_index, 2)


class TestReadOnlyMode(unittest.TestCase):
    """Tests for read-only database access."""

    def setUp(self):
        """Create temporary database with data."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"

        # Create and populate database
        with ProvenanceDatabase(self.db_path) as db:
            db.begin_session("test")

    def tearDown(self):
        """Clean up."""
        # Clean up all files in temp dir (including -wal, -shm journal files)
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_read_only_open(self):
        """Should open existing database in read-only mode."""
        with ProvenanceDatabase(self.db_path, read_only=True) as db:
            sessions = db.list_sessions()
            self.assertEqual(len(sessions), 1)

    def test_read_only_write_fails(self):
        """Write operations should fail in read-only mode."""
        with ProvenanceDatabase(self.db_path, read_only=True) as db:
            with self.assertRaises(PermissionError):
                db.begin_session("new_session")

    def test_read_only_nonexistent_fails(self):
        """Opening nonexistent database in read-only mode should fail."""
        nonexistent = Path(self.temp_dir) / "nonexistent.db"
        with self.assertRaises(FileNotFoundError):
            ProvenanceDatabase(nonexistent, read_only=True)


class TestSQLInjectionPrevention(unittest.TestCase):
    """Tests verifying SQL injection prevention."""

    def setUp(self):
        """Create temporary database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"

    def tearDown(self):
        """Clean up."""
        if self.db_path.exists():
            os.remove(self.db_path)
        os.rmdir(self.temp_dir)

    def test_sql_injection_in_session_id(self):
        """SQL injection in session ID should not work."""
        with ProvenanceDatabase(self.db_path) as db:
            malicious_id = "'; DROP TABLE sessions; --"
            result = db.get_session(malicious_id)
            self.assertIsNone(result)
            # Database should still work
            session = db.begin_session("safe")
            self.assertIsNotNone(session)

    def test_sql_injection_in_config_hash(self):
        """SQL injection in config hash should not work."""
        with ProvenanceDatabase(self.db_path) as db:
            malicious_hash = "'; DELETE FROM sessions; --"
            session = db.begin_session(malicious_hash)
            self.assertIsNotNone(session)
            # Hash should be stored literally
            retrieved = db.get_session(session.session_id)
            self.assertEqual(retrieved.config_hash, malicious_hash)

    def test_sql_injection_in_source_path(self):
        """SQL injection in source path should not work."""
        with ProvenanceDatabase(self.db_path) as db:
            malicious_source = SourceRecord(
                source_id="test",
                source_type="file",
                source_path="'; DROP TABLE sources; --",
                license_id=None,
                first_seen="2025-01-01T00:00:00Z",
            )
            db.record_source(malicious_source)
            # Database should still work
            retrieved = db.get_source("test")
            self.assertIsNotNone(retrieved)


class TestStatistics(unittest.TestCase):
    """Tests for database statistics."""

    def setUp(self):
        """Create temporary database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"

    def tearDown(self):
        """Clean up."""
        if self.db_path.exists():
            os.remove(self.db_path)
        os.rmdir(self.temp_dir)

    def test_empty_statistics(self):
        """Empty database should have zero counts."""
        with ProvenanceDatabase(self.db_path) as db:
            stats = db.get_statistics()
            self.assertEqual(stats['session_count'], 0)
            self.assertEqual(stats['batch_count'], 0)
            self.assertEqual(stats['sample_count'], 0)

    def test_statistics_after_records(self):
        """Statistics should reflect recorded data."""
        with ProvenanceDatabase(self.db_path) as db:
            # Add some data
            session = db.begin_session("test")
            source = create_test_source_record()
            db.record_source(source)
            sample = create_test_provenance_record(source_id=source.source_id)
            db.record_sample(sample)
            db.record_license(TEST_MIT_LICENSE)

            stats = db.get_statistics()
            self.assertEqual(stats['session_count'], 1)
            self.assertEqual(stats['source_count'], 1)
            self.assertEqual(stats['sample_count'], 1)
            self.assertEqual(stats['license_count'], 1)


if __name__ == "__main__":
    unittest.main()
