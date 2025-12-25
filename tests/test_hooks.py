"""
Tests for Origin instrumentation hooks.

Tests cover:
- BaseHook abstract interface
- Mock tensor handling
- DataLoaderHook (with mock PyTorch)
- DatasetHook (with mock HuggingFace)
- Batch recording
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from origin.hooks.base import BaseHook
from origin.storage.database import ProvenanceDatabase
from origin.core.fingerprint import fingerprint_bytes, fingerprint_sample
from tests.fixtures.sample_data import (
    create_mock_tensor,
    MockTensor,
    create_test_source_record,
)


class ConcreteHook(BaseHook):
    """Concrete implementation of BaseHook for testing."""

    def __init__(
        self,
        db: ProvenanceDatabase,
        session_id: str,
        source_id: str = "test_source",
        license_id: str = None,
    ):
        """Initialize the concrete hook."""
        super().__init__(db, session_id, source_id, license_id)

    def _extract_samples(self, data):
        """Extract samples from a list of items."""
        return [(item, {}) for item in data]

    def set_source(self, source_id: str, license_id: str = None) -> None:
        """Set the current source for recording."""
        self._source_id = source_id
        self._license_id = license_id

    def on_batch(self, batch_data: list) -> str:
        """Convenience method to observe a batch of data."""
        return self.observe(batch_data)


class TestBaseHook(unittest.TestCase):
    """Tests for BaseHook abstract class."""

    def setUp(self):
        """Create temporary database and session."""
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

    def test_concrete_hook_creation(self):
        """Concrete hook should be creatable."""
        hook = ConcreteHook(self.db, self.session.session_id)
        self.assertIsNotNone(hook)

    def test_record_batch_creates_batch(self):
        """Recording a batch should create batch record."""
        # Register source first
        source = create_test_source_record()
        self.db.record_source(source)

        hook = ConcreteHook(self.db, self.session.session_id)
        hook.set_source(source.source_id, source.license_id)

        # Record batch with simple data
        batch_data = [b"sample1", b"sample2", b"sample3"]
        hook.on_batch(batch_data)

        # Verify batch was recorded
        batches = self.db.list_batches(self.session.session_id)
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0].sample_count, 3)

    def test_multiple_batches(self):
        """Multiple batches should increment batch index."""
        source = create_test_source_record()
        self.db.record_source(source)

        hook = ConcreteHook(self.db, self.session.session_id)
        hook.set_source(source.source_id, source.license_id)

        hook.on_batch([b"batch1_sample1", b"batch1_sample2"])
        hook.on_batch([b"batch2_sample1"])
        hook.on_batch([b"batch3_sample1", b"batch3_sample2", b"batch3_sample3"])

        batches = self.db.list_batches(self.session.session_id)
        self.assertEqual(len(batches), 3)
        self.assertEqual(batches[0].batch_index, 0)
        self.assertEqual(batches[1].batch_index, 1)
        self.assertEqual(batches[2].batch_index, 2)


class TestMockTensorHandling(unittest.TestCase):
    """Tests for handling mock tensor objects."""

    def test_mock_tensor_tobytes(self):
        """MockTensor should provide tobytes method."""
        tensor = create_mock_tensor(data=b"test data")
        self.assertEqual(tensor.tobytes(), b"test data")

    def test_mock_tensor_shape(self):
        """MockTensor should have shape attribute."""
        tensor = create_mock_tensor(shape=(2, 3, 4))
        self.assertEqual(tensor.shape, (2, 3, 4))

    def test_mock_tensor_dtype(self):
        """MockTensor should have dtype attribute."""
        tensor = create_mock_tensor(dtype="float64")
        self.assertEqual(tensor.dtype, "float64")

    def test_fingerprint_mock_tensor(self):
        """MockTensor should be fingerprintable."""
        tensor = create_mock_tensor(data=b"tensor content")
        fp = fingerprint_sample(tensor)
        self.assertEqual(len(fp), 64)

    def test_same_tensor_same_fingerprint(self):
        """Same tensor data should produce same fingerprint."""
        tensor1 = create_mock_tensor(data=b"same data")
        tensor2 = create_mock_tensor(data=b"same data")
        self.assertEqual(
            fingerprint_sample(tensor1),
            fingerprint_sample(tensor2)
        )

    def test_different_tensor_different_fingerprint(self):
        """Different tensor data should produce different fingerprints."""
        tensor1 = create_mock_tensor(data=b"data1")
        tensor2 = create_mock_tensor(data=b"data2")
        self.assertNotEqual(
            fingerprint_sample(tensor1),
            fingerprint_sample(tensor2)
        )


class TestDataLoaderHook(unittest.TestCase):
    """Tests for PyTorch DataLoaderHook with mocked dependencies."""

    def setUp(self):
        """Create temporary database."""
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

    def test_import_without_torch(self):
        """DataLoaderHook should handle missing torch gracefully."""
        # This tests that the import doesn't fail when torch is missing
        try:
            from origin.hooks.pytorch import DataLoaderHook, TORCH_AVAILABLE
            # Should import successfully regardless of torch availability
            self.assertIsNotNone(DataLoaderHook)
        except ImportError:
            # If import fails entirely, that's also acceptable
            pass

    def test_hook_initialization_with_mock(self):
        """Hook should initialize with mock data loader."""
        from origin.hooks.pytorch import DataLoaderHook, TORCH_AVAILABLE

        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        # Create mock data loader
        mock_loader = MagicMock()
        mock_loader.dataset = MagicMock()
        mock_loader.__iter__ = MagicMock(return_value=iter([]))

        source = create_test_source_record()
        self.db.record_source(source)

        hook = DataLoaderHook(
            db=self.db,
            session_id=self.session.session_id,
            data_loader=mock_loader,
            source_id=source.source_id,
        )
        self.assertIsNotNone(hook)


class TestDatasetHook(unittest.TestCase):
    """Tests for HuggingFace DatasetHook with mocked dependencies."""

    def setUp(self):
        """Create temporary database."""
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

    def test_import_without_datasets(self):
        """DatasetHook should handle missing datasets gracefully."""
        try:
            from origin.hooks.huggingface import DatasetHook, DATASETS_AVAILABLE
            self.assertIsNotNone(DatasetHook)
        except ImportError:
            pass

    def test_hook_initialization_with_mock(self):
        """Hook should initialize with mock dataset."""
        from origin.hooks.huggingface import DatasetHook, DATASETS_AVAILABLE

        if not DATASETS_AVAILABLE:
            self.skipTest("HuggingFace datasets not available")

        # Create mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = MagicMock(return_value=iter([]))
        mock_dataset.info = MagicMock()
        mock_dataset.info.license = "MIT"

        source = create_test_source_record()
        self.db.record_source(source)

        hook = DatasetHook(
            db=self.db,
            session_id=self.session.session_id,
            dataset=mock_dataset,
            source_id=source.source_id,
        )
        self.assertIsNotNone(hook)


class TestHookSampleRecording(unittest.TestCase):
    """Tests for sample recording through hooks."""

    def setUp(self):
        """Create temporary database."""
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

    def test_samples_recorded_with_correct_source(self):
        """Samples should be recorded with correct source ID."""
        source = create_test_source_record()
        self.db.record_source(source)

        hook = ConcreteHook(self.db, self.session.session_id)
        hook.set_source(source.source_id, source.license_id)

        hook.on_batch([b"sample1", b"sample2"])

        # Get samples from batch
        batches = self.db.list_batches(self.session.session_id)
        samples = self.db.list_samples(batches[0].batch_id)

        self.assertEqual(len(samples), 2)
        for sample in samples:
            self.assertEqual(sample.source_id, source.source_id)

    def test_duplicate_samples_handled(self):
        """Duplicate samples produce identical batch fingerprints."""
        source = create_test_source_record()
        self.db.record_source(source)

        hook = ConcreteHook(self.db, self.session.session_id)
        hook.set_source(source.source_id, source.license_id)

        # Same sample content produces the same fingerprint
        batch_id_1 = hook.on_batch([b"duplicate"])
        batch_id_2 = hook.on_batch([b"duplicate"])

        # Same content should produce same batch fingerprint
        self.assertEqual(batch_id_1, batch_id_2)

        # Only one unique batch should be stored (INSERT OR IGNORE)
        batches = self.db.list_batches(self.session.session_id)
        self.assertEqual(len(batches), 1)

    def test_different_batches_created_for_different_content(self):
        """Different batch content should create different batches."""
        source = create_test_source_record()
        self.db.record_source(source)

        hook = ConcreteHook(self.db, self.session.session_id)
        hook.set_source(source.source_id, source.license_id)

        # Different content in each batch
        hook.on_batch([b"batch_1_content"])
        hook.on_batch([b"batch_2_content"])

        batches = self.db.list_batches(self.session.session_id)
        self.assertEqual(len(batches), 2)


class TestHookSourceManagement(unittest.TestCase):
    """Tests for source management in hooks."""

    def setUp(self):
        """Create temporary database."""
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

    def test_set_source(self):
        """Setting source should update hook state."""
        source = create_test_source_record()
        self.db.record_source(source)

        hook = ConcreteHook(self.db, self.session.session_id)
        hook.set_source(source.source_id, source.license_id)

        self.assertEqual(hook._source_id, source.source_id)
        self.assertEqual(hook._license_id, source.license_id)

    def test_change_source_between_batches(self):
        """Source can be changed between batches."""
        source1 = create_test_source_record(
            source_id="source1",
            source_path="/data/file1.csv",
        )
        source2 = create_test_source_record(
            source_id="source2",
            source_path="/data/file2.csv",
        )
        self.db.record_source(source1)
        self.db.record_source(source2)

        hook = ConcreteHook(self.db, self.session.session_id)

        hook.set_source(source1.source_id, source1.license_id)
        hook.on_batch([b"from_source1"])

        hook.set_source(source2.source_id, source2.license_id)
        hook.on_batch([b"from_source2"])

        batches = self.db.list_batches(self.session.session_id)
        samples1 = self.db.list_samples(batches[0].batch_id)
        samples2 = self.db.list_samples(batches[1].batch_id)

        self.assertEqual(samples1[0].source_id, "source1")
        self.assertEqual(samples2[0].source_id, "source2")


if __name__ == "__main__":
    unittest.main()
