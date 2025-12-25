"""
Base hook for pipeline instrumentation.

This module provides the abstract base class for all instrumentation hooks.
Hooks observe data as it flows through training pipelines without modifying
the data in any way.
"""

import abc
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Tuple

from origin.core.fingerprint import fingerprint_sample, merkle_root
from origin.core.record import ProvenanceRecord, BatchRecord
from origin.storage.database import ProvenanceDatabase


class BaseHook(abc.ABC):
    """
    Abstract base class for pipeline instrumentation hooks.

    Hooks observe data as it passes through training pipelines, recording
    provenance information without modifying the data. Each hook implementation
    handles a specific data format (e.g., PyTorch tensors, HuggingFace datasets).

    Subclasses must implement:
        - _extract_samples(data): Extract individual samples from a batch

    Attributes:
        db: The ProvenanceDatabase for recording observations.
        session_id: The session ID for this training run.
        source_id: Identifier for the data source.
        license_id: Optional license ID for the data source.

    Example:
        >>> class MyHook(BaseHook):
        ...     def _extract_samples(self, data):
        ...         return [(item, {}) for item in data]
        ...
        >>> hook = MyHook(db, session_id, source_id="my_dataset")
        >>> for batch in hook.wrap(data_loader):
        ...     process(batch)  # batch is unchanged
    """

    def __init__(
        self,
        db: ProvenanceDatabase,
        session_id: str,
        source_id: str = "unknown",
        license_id: Optional[str] = None,
    ) -> None:
        """
        Initialize the base hook.

        Args:
            db: ProvenanceDatabase instance for recording observations.
            session_id: Session ID for this training run.
            source_id: Identifier for the data source (default: 'unknown').
            license_id: Optional SPDX license ID for the source data.
        """
        self._db = db
        self._session_id = session_id
        self._source_id = source_id
        self._license_id = license_id

        # Counters and state
        self._batch_count: int = 0
        self._sample_count: int = 0
        self._start_time: Optional[float] = None
        self._fingerprint_cache: Dict[str, str] = {}

    @property
    def db(self) -> ProvenanceDatabase:
        """Return the database instance."""
        return self._db

    @property
    def session_id(self) -> str:
        """Return the session ID."""
        return self._session_id

    @property
    def source_id(self) -> str:
        """Return the source ID."""
        return self._source_id

    @property
    def license_id(self) -> Optional[str]:
        """Return the license ID."""
        return self._license_id

    @abc.abstractmethod
    def _extract_samples(self, data: Any) -> List[Tuple[Any, Dict[str, Any]]]:
        """
        Extract individual samples from a batch of data.

        This method must be implemented by subclasses to handle specific
        data formats (e.g., PyTorch tensors, HuggingFace datasets).

        Args:
            data: A batch of data in the format specific to the hook type.

        Returns:
            A list of tuples, each containing:
                - sample: The individual sample data (for fingerprinting)
                - metadata: A dictionary of metadata about the sample
        """
        pass

    def _get_content_type(self, sample: Any) -> str:
        """
        Determine the content type of a sample.

        Args:
            sample: The sample to classify.

        Returns:
            A string describing the content type.
        """
        if isinstance(sample, bytes):
            return "bytes"
        elif isinstance(sample, str):
            return "text"
        elif isinstance(sample, dict):
            return "dict"
        elif hasattr(sample, "shape") and hasattr(sample, "dtype"):
            return "tensor"
        else:
            return "unknown"

    def _get_byte_size(self, sample: Any) -> int:
        """
        Estimate the byte size of a sample.

        Args:
            sample: The sample to measure.

        Returns:
            Estimated size in bytes.
        """
        if isinstance(sample, bytes):
            return len(sample)
        elif isinstance(sample, str):
            return len(sample.encode("utf-8"))
        elif hasattr(sample, "nbytes"):
            return int(sample.nbytes)
        elif hasattr(sample, "element_size") and hasattr(sample, "numel"):
            # PyTorch tensor
            return sample.element_size() * sample.numel()
        else:
            # Fallback: use string representation length
            return len(str(sample))

    @staticmethod
    def _now_iso() -> str:
        """Return current UTC time as ISO 8601 string."""
        return datetime.now(timezone.utc).isoformat()

    def observe(self, data: Any) -> str:
        """
        Observe a batch of data and record provenance information.

        This method:
        1. Extracts individual samples from the batch
        2. Computes fingerprints for each sample
        3. Records sample and batch information to the database
        4. Returns the batch fingerprint (Merkle root)

        Args:
            data: A batch of data to observe.

        Returns:
            The batch fingerprint (Merkle root of sample fingerprints).
        """
        timestamp = self._now_iso()

        # Extract samples from the batch
        samples = self._extract_samples(data)

        if not samples:
            # Empty batch - create a fingerprint from empty marker
            return fingerprint_sample(b"empty_batch")

        sample_fingerprints: List[str] = []

        for sample_data, metadata in samples:
            # Compute fingerprint
            # We always compute the fingerprint (it's the content-addressable key)
            # and use it to track unique samples
            fp = fingerprint_sample(sample_data)

            # Track unique fingerprints for stats
            # Using fingerprint as key ensures deduplication is content-based
            self._fingerprint_cache[fp] = True

            sample_fingerprints.append(fp)

            # Create and record provenance record
            record = ProvenanceRecord(
                sample_id=fp,
                source_id=self._source_id,
                content_type=self._get_content_type(sample_data),
                byte_size=self._get_byte_size(sample_data),
                timestamp=timestamp,
                metadata=metadata,
            )
            self._db.record_sample(record)
            self._sample_count += 1

        # Compute batch fingerprint as Merkle root
        batch_fingerprint = merkle_root(sample_fingerprints)

        # Create and record batch
        batch = BatchRecord(
            batch_id=batch_fingerprint,
            session_id=self._session_id,
            batch_index=self._batch_count,
            sample_count=len(samples),
            created_at=timestamp,
            sample_ids=tuple(sample_fingerprints),
        )
        self._db.record_batch(batch)
        self._batch_count += 1

        return batch_fingerprint

    def get_stats(self) -> Dict[str, Any]:
        """
        Get observation statistics.

        Returns:
            A dictionary containing:
                - batches_observed: Number of batches observed
                - samples_observed: Total number of samples observed
                - unique_samples: Number of unique sample fingerprints
                - elapsed_seconds: Time elapsed since first observation
        """
        elapsed = 0.0
        if self._start_time is not None:
            elapsed = time.time() - self._start_time

        return {
            "batches_observed": self._batch_count,
            "samples_observed": self._sample_count,
            "unique_samples": len(self._fingerprint_cache),
            "elapsed_seconds": elapsed,
        }

    def wrap(self, iterable: Any) -> Iterator[Any]:
        """
        Wrap an iterable to observe data as it passes through.

        This generator yields each item unchanged while recording provenance
        information. Use this to instrument existing data pipelines.

        Args:
            iterable: Any iterable (e.g., DataLoader, Dataset).

        Yields:
            Each item from the iterable, unchanged.

        Example:
            >>> for batch in hook.wrap(dataloader):
            ...     # batch is unchanged
            ...     train_step(batch)
        """
        self._start_time = time.time()

        for item in iterable:
            self.observe(item)
            yield item

    def reset_stats(self) -> None:
        """
        Reset observation statistics.

        This clears counters and the fingerprint cache but does not
        affect already-recorded data in the database.
        """
        self._batch_count = 0
        self._sample_count = 0
        self._start_time = None
        self._fingerprint_cache.clear()
