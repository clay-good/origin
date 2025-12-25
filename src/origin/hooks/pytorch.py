"""
PyTorch DataLoader instrumentation hook.

This module provides the DataLoaderHook class for instrumenting PyTorch
DataLoader pipelines. It observes batches as they flow through the loader
without modifying the data.

Requires: torch>=2.0 (optional dependency)
"""

from typing import Any, Dict, List, Optional, Tuple

from origin.hooks.base import BaseHook
from origin.storage.database import ProvenanceDatabase

# Check for PyTorch availability
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False


class DataLoaderHook(BaseHook):
    """
    Instrumentation hook for PyTorch DataLoaders.

    This hook wraps DataLoader iteration to record provenance information
    for each batch without modifying the data. It handles common batch
    formats including single tensors, tuples of tensors, and dictionaries.

    Requires PyTorch to be installed.

    Attributes:
        record_gradients: Whether to include gradient information in metadata.

    Example:
        >>> from torch.utils.data import DataLoader
        >>> hook = DataLoaderHook(db, session_id, source_id="mnist")
        >>> for batch in hook.wrap(dataloader):
        ...     inputs, labels = batch
        ...     # batch is unchanged, provenance recorded automatically
        ...     train_step(inputs, labels)
    """

    def __init__(
        self,
        db: ProvenanceDatabase,
        session_id: str,
        source_id: str = "pytorch_dataset",
        license_id: Optional[str] = None,
        record_gradients: bool = False,
    ) -> None:
        """
        Initialize the DataLoader hook.

        Args:
            db: ProvenanceDatabase instance for recording observations.
            session_id: Session ID for this training run.
            source_id: Identifier for the data source (default: 'pytorch_dataset').
            license_id: Optional SPDX license ID for the source data.
            record_gradients: Whether to record gradient information in metadata.

        Raises:
            ImportError: If PyTorch is not installed.
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for DataLoaderHook. "
                "Install it with: pip install torch>=2.0"
            )

        super().__init__(db, session_id, source_id, license_id)
        self._record_gradients = record_gradients

    @property
    def record_gradients(self) -> bool:
        """Return whether gradients are being recorded."""
        return self._record_gradients

    def _tensor_to_bytes(self, tensor: "torch.Tensor") -> bytes:
        """
        Convert a PyTorch tensor to bytes for fingerprinting.

        The tensor is detached and moved to CPU before conversion to ensure
        consistent fingerprints regardless of device or gradient state.

        Args:
            tensor: PyTorch tensor to convert.

        Returns:
            Bytes representation of the tensor data.
        """
        # Detach from computation graph and move to CPU
        t = tensor.detach()
        if t.device.type != "cpu":
            t = t.cpu()

        # Convert to contiguous numpy array, then to bytes
        return t.numpy().tobytes()

    def _fingerprint_tensor(self, tensor: "torch.Tensor") -> Dict[str, Any]:
        """
        Create a fingerprint-ready representation of a tensor.

        Returns a dictionary that can be fingerprinted, including shape
        and dtype information for reproducibility.

        Args:
            tensor: PyTorch tensor to process.

        Returns:
            Dictionary with tensor metadata suitable for fingerprinting.
        """
        t = tensor.detach()
        if t.device.type != "cpu":
            t = t.cpu()

        # Return numpy array which fingerprint_sample can handle
        return t.numpy()

    def _extract_samples(self, batch: Any) -> List[Tuple[Any, Dict[str, Any]]]:
        """
        Extract individual samples from a PyTorch batch.

        Handles common DataLoader output formats:
        - Single tensor: Each row (dim 0) is a sample
        - Tuple/list of tensors: First element is data, rest are labels
        - Dictionary: Each value is processed independently

        Args:
            batch: A batch from a DataLoader.

        Returns:
            List of (sample_data, metadata) tuples.
        """
        samples: List[Tuple[Any, Dict[str, Any]]] = []

        if torch is None:
            return samples

        if isinstance(batch, torch.Tensor):
            # Single tensor - each row is a sample
            samples.extend(self._extract_from_tensor(batch))

        elif isinstance(batch, (tuple, list)):
            # Tuple/list of tensors - typically (data, labels, ...)
            # Extract samples from the first tensor (data)
            if len(batch) > 0 and isinstance(batch[0], torch.Tensor):
                data_tensor = batch[0]
                batch_size = data_tensor.shape[0] if data_tensor.dim() > 0 else 1

                for i in range(batch_size):
                    # Combine all tensor slices at index i
                    sample_parts = []
                    for j, item in enumerate(batch):
                        if isinstance(item, torch.Tensor) and item.dim() > 0:
                            part = item[i]
                            sample_parts.append(self._fingerprint_tensor(part))
                        elif isinstance(item, torch.Tensor):
                            sample_parts.append(self._fingerprint_tensor(item))

                    # Create a combined representation
                    if len(sample_parts) == 1:
                        sample_data = sample_parts[0]
                    else:
                        # Combine as dict for fingerprinting
                        sample_data = {
                            f"part_{k}": self._fingerprint_tensor(batch[k][i])
                            if isinstance(batch[k], torch.Tensor) and batch[k].dim() > 0
                            else (self._fingerprint_tensor(batch[k]) if isinstance(batch[k], torch.Tensor) else batch[k])
                            for k in range(len(batch))
                            if isinstance(batch[k], torch.Tensor)
                        }

                    metadata = {
                        "index": i,
                        "batch_size": batch_size,
                        "num_elements": len(batch),
                    }
                    samples.append((sample_data, metadata))

        elif isinstance(batch, dict):
            # Dictionary batch - transpose to list of dicts
            samples.extend(self._extract_from_dict(batch))

        else:
            # Unknown format - treat as single sample
            samples.append((batch, {"format": "unknown"}))

        return samples

    def _extract_from_tensor(
        self, tensor: "torch.Tensor"
    ) -> List[Tuple[Any, Dict[str, Any]]]:
        """
        Extract samples from a single tensor.

        Args:
            tensor: PyTorch tensor where dim 0 is the batch dimension.

        Returns:
            List of (sample_data, metadata) tuples.
        """
        samples = []

        if tensor.dim() == 0:
            # Scalar tensor
            samples.append((
                self._fingerprint_tensor(tensor),
                {"shape": list(tensor.shape), "dtype": str(tensor.dtype)},
            ))
        else:
            # Batch tensor - iterate along dim 0
            batch_size = tensor.shape[0]
            for i in range(batch_size):
                sample_tensor = tensor[i]
                samples.append((
                    self._fingerprint_tensor(sample_tensor),
                    {
                        "index": i,
                        "shape": list(sample_tensor.shape),
                        "dtype": str(tensor.dtype),
                    },
                ))

        return samples

    def _extract_from_dict(
        self, batch: Dict[str, Any]
    ) -> List[Tuple[Any, Dict[str, Any]]]:
        """
        Extract samples from a dictionary batch.

        Args:
            batch: Dictionary where values may be tensors or lists.

        Returns:
            List of (sample_data, metadata) tuples.
        """
        samples = []

        # Determine batch size from first tensor
        batch_size = 1
        for value in batch.values():
            if isinstance(value, torch.Tensor) and value.dim() > 0:
                batch_size = value.shape[0]
                break
            elif isinstance(value, (list, tuple)):
                batch_size = len(value)
                break

        # Transpose dict of lists to list of dicts
        for i in range(batch_size):
            sample_dict = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    if value.dim() > 0 and value.shape[0] == batch_size:
                        sample_dict[key] = self._fingerprint_tensor(value[i])
                    else:
                        sample_dict[key] = self._fingerprint_tensor(value)
                elif isinstance(value, (list, tuple)) and len(value) == batch_size:
                    sample_dict[key] = value[i]
                else:
                    sample_dict[key] = value

            metadata = {
                "index": i,
                "columns": list(batch.keys()),
            }
            samples.append((sample_dict, metadata))

        return samples
