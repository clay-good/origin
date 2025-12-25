"""
HuggingFace datasets instrumentation hook.

This module provides the DatasetHook class for instrumenting HuggingFace
datasets pipelines. It observes samples as they flow through the dataset
without modifying the data.

Requires: datasets>=2.0 (optional dependency)
"""

from typing import Any, Dict, Iterator, List, Optional, Tuple

from origin.hooks.base import BaseHook
from origin.storage.database import ProvenanceDatabase

# Check for HuggingFace datasets availability
try:
    import datasets
    DATASETS_AVAILABLE = True
except ImportError:
    datasets = None  # type: ignore
    DATASETS_AVAILABLE = False


class DatasetHook(BaseHook):
    """
    Instrumentation hook for HuggingFace datasets.

    This hook wraps dataset iteration to record provenance information
    for each sample or batch without modifying the data. It handles both
    regular datasets and streaming datasets.

    Requires the HuggingFace datasets library to be installed.

    Example:
        >>> from datasets import load_dataset
        >>> dataset = load_dataset("imdb", split="train")
        >>> hook = DatasetHook(db, session_id, source_id="imdb")
        >>> for batch in hook.wrap(dataset.iter(batch_size=32)):
        ...     # batch is unchanged, provenance recorded automatically
        ...     process_batch(batch)
    """

    def __init__(
        self,
        db: ProvenanceDatabase,
        session_id: str,
        source_id: str = "hf_dataset",
        license_id: Optional[str] = None,
    ) -> None:
        """
        Initialize the Dataset hook.

        Args:
            db: ProvenanceDatabase instance for recording observations.
            session_id: Session ID for this training run.
            source_id: Identifier for the data source (default: 'hf_dataset').
            license_id: Optional SPDX license ID for the source data.

        Raises:
            ImportError: If HuggingFace datasets is not installed.
        """
        if not DATASETS_AVAILABLE:
            raise ImportError(
                "HuggingFace datasets is required for DatasetHook. "
                "Install it with: pip install datasets>=2.0"
            )

        super().__init__(db, session_id, source_id, license_id)

    def _extract_samples(self, batch: Any) -> List[Tuple[Any, Dict[str, Any]]]:
        """
        Extract individual samples from a HuggingFace batch.

        HuggingFace batches are typically dictionaries where each value is
        a list of items (one per sample). This method transposes the batch
        from dict-of-lists to list-of-dicts.

        Args:
            batch: A batch from a HuggingFace dataset (dict of lists).

        Returns:
            List of (sample_dict, metadata) tuples.
        """
        samples: List[Tuple[Any, Dict[str, Any]]] = []

        if isinstance(batch, dict):
            # Standard HuggingFace batch format: dict of lists
            samples.extend(self._extract_from_batch_dict(batch))

        elif isinstance(batch, (list, tuple)):
            # List of samples (less common)
            for i, item in enumerate(batch):
                if isinstance(item, dict):
                    samples.append((item, {"index": i}))
                else:
                    samples.append(({"value": item}, {"index": i}))

        else:
            # Single sample or unknown format
            if isinstance(batch, dict):
                samples.append((batch, {"format": "single_dict"}))
            else:
                samples.append(({"value": batch}, {"format": "unknown"}))

        return samples

    def _extract_from_batch_dict(
        self, batch: Dict[str, Any]
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Extract samples from a dict-of-lists batch format.

        Args:
            batch: Dictionary where each value is a list of items.

        Returns:
            List of (sample_dict, metadata) tuples.
        """
        samples = []

        # Determine batch size from first list-like value
        batch_size = 1
        columns = list(batch.keys())

        for value in batch.values():
            if isinstance(value, (list, tuple)):
                batch_size = len(value)
                break

        # Check if this is actually a batched format
        is_batched = all(
            isinstance(v, (list, tuple)) and len(v) == batch_size
            for v in batch.values()
            if isinstance(v, (list, tuple))
        )

        if is_batched and batch_size > 0:
            # Transpose dict-of-lists to list-of-dicts
            for i in range(batch_size):
                sample_dict = {}
                for key, value in batch.items():
                    if isinstance(value, (list, tuple)) and len(value) == batch_size:
                        sample_dict[key] = value[i]
                    else:
                        # Non-list value or mismatched length - include as-is
                        sample_dict[key] = value

                metadata = {
                    "index": i,
                    "columns": columns,
                }
                samples.append((sample_dict, metadata))
        else:
            # Single sample in dict format
            samples.append((batch, {"columns": columns, "format": "single"}))

        return samples

    def wrap_dataset(self, dataset: Any) -> Iterator[Any]:
        """
        Wrap a HuggingFace Dataset to observe individual samples.

        This is a specialized wrapper for datasets.Dataset objects that
        yields individual samples rather than batches. Each sample is
        observed and recorded.

        Args:
            dataset: A HuggingFace Dataset object.

        Yields:
            Individual samples from the dataset, unchanged.

        Example:
            >>> from datasets import load_dataset
            >>> dataset = load_dataset("imdb", split="train")
            >>> hook = DatasetHook(db, session_id, source_id="imdb")
            >>> for sample in hook.wrap_dataset(dataset):
            ...     process(sample)  # sample is unchanged
        """
        import time
        self._start_time = time.time()

        for sample in dataset:
            # Wrap single sample as a batch of size 1
            if isinstance(sample, dict):
                # Convert to batch format for consistent handling
                batch = {key: [value] for key, value in sample.items()}
            else:
                batch = {"value": [sample]}

            self.observe(batch)
            yield sample

    def wrap_streaming(self, dataset: Any) -> Iterator[Any]:
        """
        Wrap a HuggingFace streaming dataset.

        This handles IterableDataset objects from HuggingFace datasets
        that support streaming (e.g., large datasets loaded with streaming=True).

        Args:
            dataset: A HuggingFace IterableDataset object.

        Yields:
            Individual samples from the streaming dataset, unchanged.

        Example:
            >>> from datasets import load_dataset
            >>> dataset = load_dataset("imdb", split="train", streaming=True)
            >>> hook = DatasetHook(db, session_id, source_id="imdb")
            >>> for sample in hook.wrap_streaming(dataset):
            ...     process(sample)
        """
        # Streaming datasets work the same way as regular iteration
        return self.wrap_dataset(dataset)
