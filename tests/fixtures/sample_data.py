"""
Test fixtures providing sample data and helper functions for Origin tests.

This module contains test constants, mock objects, and factory functions
for creating test record instances.
"""

from datetime import datetime, timezone
from typing import Tuple

from origin.core.record import (
    ProvenanceRecord,
    SessionRecord,
    BatchRecord,
    SourceRecord,
)
from origin.core.fingerprint import fingerprint_bytes, merkle_root


# Test constants
SAMPLE_TEXT = "This is test content for fingerprinting."
SAMPLE_DICT = {"key": "value", "number": 42}
SAMPLE_BYTES = b"raw byte content"


class MockTensor:
    """
    Mock tensor object for testing without PyTorch dependency.

    Simulates the interface of a torch.Tensor for fingerprinting tests.
    """

    def __init__(
        self,
        data: bytes = b"mock tensor data",
        shape: Tuple[int, ...] = (2, 3),
        dtype: str = "float32",
    ) -> None:
        """
        Initialize mock tensor.

        Args:
            data: Raw bytes representing tensor data.
            shape: Shape tuple of the tensor.
            dtype: Data type string.
        """
        self._data = data
        self.shape = shape
        self.dtype = dtype

    def tobytes(self) -> bytes:
        """Return raw byte representation."""
        return self._data

    def numpy(self):
        """Return self as mock numpy array."""
        return self


def create_mock_tensor(
    data: bytes = b"mock tensor data",
    shape: Tuple[int, ...] = (2, 3),
    dtype: str = "float32",
) -> MockTensor:
    """
    Create a mock tensor for testing.

    Args:
        data: Raw bytes for the tensor.
        shape: Shape of the tensor.
        dtype: Data type string.

    Returns:
        MockTensor instance.
    """
    return MockTensor(data=data, shape=shape, dtype=dtype)


def create_test_session_record(
    session_id: str = "test-session-001",
    config_hash: str = "abc123def456",
    status: str = "running",
) -> SessionRecord:
    """
    Create a test SessionRecord.

    Args:
        session_id: Session identifier.
        config_hash: Configuration hash.
        status: Session status.

    Returns:
        SessionRecord instance for testing.
    """
    return SessionRecord(
        session_id=session_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        config_hash=config_hash,
        status=status,
        metadata={"test": True},
    )


def create_test_source_record(
    source_id: str = "src-test-001",
    source_type: str = "file",
    source_path: str = "/data/test.csv",
    license_id: str = "MIT",
) -> SourceRecord:
    """
    Create a test SourceRecord.

    Args:
        source_id: Source identifier.
        source_type: Type of source.
        source_path: Path to source.
        license_id: License identifier.

    Returns:
        SourceRecord instance for testing.
    """
    return SourceRecord(
        source_id=source_id,
        source_type=source_type,
        source_path=source_path,
        license_id=license_id,
        first_seen=datetime.now(timezone.utc).isoformat(),
        metadata={},
    )


def create_test_provenance_record(
    content: bytes = b"test sample content",
    source_id: str = "src-test-001",
    content_type: str = "text",
) -> ProvenanceRecord:
    """
    Create a test ProvenanceRecord with computed fingerprint.

    Args:
        content: Content to fingerprint.
        source_id: Source identifier.
        content_type: Type of content.

    Returns:
        ProvenanceRecord instance for testing.
    """
    sample_id = fingerprint_bytes(content)
    return ProvenanceRecord(
        sample_id=sample_id,
        source_id=source_id,
        content_type=content_type,
        byte_size=len(content),
        timestamp=datetime.now(timezone.utc).isoformat(),
        metadata={},
    )


def create_test_batch_record(
    session_id: str = "test-session-001",
    batch_index: int = 0,
    sample_ids: Tuple[str, ...] = None,
) -> BatchRecord:
    """
    Create a test BatchRecord.

    Args:
        session_id: Session identifier.
        batch_index: Index of batch in session.
        sample_ids: Tuple of sample IDs. If None, creates default samples.

    Returns:
        BatchRecord instance for testing.
    """
    if sample_ids is None:
        # Create some default sample IDs
        sample_ids = tuple(
            fingerprint_bytes(f"sample_{i}".encode())
            for i in range(3)
        )

    batch_id = merkle_root(list(sample_ids)) if sample_ids else fingerprint_bytes(b"empty")

    return BatchRecord(
        batch_id=batch_id,
        session_id=session_id,
        batch_index=batch_index,
        sample_count=len(sample_ids),
        created_at=datetime.now(timezone.utc).isoformat(),
        sample_ids=sample_ids,
    )
