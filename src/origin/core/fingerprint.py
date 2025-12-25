"""
Fingerprinting engine for Origin provenance tracking.

This module provides deterministic, content-addressable hashing for data
samples. All functions are pure and deterministic - the same input will
always produce the same output, enabling reliable provenance tracking.

The module implements:
- Individual sample fingerprinting (bytes, text, tensors, dicts)
- Merkle tree construction for batch fingerprinting
- Thread-safe LRU caching for performance optimization

All fingerprints are SHA-256 hashes represented as 64-character lowercase
hexadecimal strings.
"""

import hashlib
import json
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional


def fingerprint_bytes(data: bytes) -> str:
    """
    Compute SHA-256 fingerprint of raw bytes.

    This is the fundamental fingerprinting operation. All other fingerprint
    functions ultimately delegate to this one.

    Args:
        data: Raw bytes to fingerprint.

    Returns:
        A 64-character lowercase hexadecimal string representing the SHA-256
        hash of the input bytes.

    Raises:
        TypeError: If data is not bytes.

    Note:
        This function is deterministic: the same input bytes will always
        produce the same output hash.

    Example:
        >>> fingerprint_bytes(b"hello")
        '2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824'
    """
    if not isinstance(data, bytes):
        raise TypeError(f"Expected bytes, got {type(data).__name__}")
    return hashlib.sha256(data).hexdigest()


def fingerprint_text(text: str) -> str:
    """
    Compute SHA-256 fingerprint of a text string.

    The text is encoded as UTF-8 bytes before hashing, ensuring consistent
    fingerprints across different platforms and Python versions.

    Args:
        text: Unicode string to fingerprint.

    Returns:
        A 64-character lowercase hexadecimal string representing the SHA-256
        hash of the UTF-8 encoded text.

    Raises:
        TypeError: If text is not a string.

    Note:
        This function is deterministic: the same text will always produce
        the same fingerprint. The fingerprint of text equals the fingerprint
        of its UTF-8 byte encoding.

    Example:
        >>> fingerprint_text("hello")
        '2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824'
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")
    return fingerprint_bytes(text.encode("utf-8"))


def fingerprint_tensor(tensor: Any) -> str:
    """
    Compute SHA-256 fingerprint of a tensor-like object.

    Handles numpy arrays and PyTorch tensors by creating a canonical byte
    representation that includes shape and dtype information for full
    reproducibility.

    The canonical representation is:
        shape_bytes + b'|' + dtype_bytes + b'|' + data_bytes

    Where:
        - shape_bytes: JSON-encoded shape tuple as UTF-8 bytes
        - dtype_bytes: String name of the dtype as UTF-8 bytes
        - data_bytes: Raw tensor data from tobytes()

    Args:
        tensor: A numpy array or PyTorch tensor (any object with 'shape',
            'dtype', and 'tobytes' attributes).

    Returns:
        A 64-character lowercase hexadecimal string representing the SHA-256
        hash of the canonical tensor representation.

    Raises:
        TypeError: If the object lacks required attributes (shape, dtype,
            tobytes).

    Note:
        This function is deterministic: tensors with identical shape, dtype,
        and data will produce identical fingerprints. The inclusion of shape
        and dtype ensures that tensors with the same raw bytes but different
        interpretations produce different fingerprints.

    Example:
        >>> import numpy as np
        >>> arr = np.array([1, 2, 3], dtype=np.int32)
        >>> fingerprint_tensor(arr)  # Returns consistent 64-char hex string
    """
    # Check for required attributes
    if not hasattr(tensor, "tobytes"):
        raise TypeError(
            f"Object of type {type(tensor).__name__} lacks 'tobytes' method. "
            "Expected numpy array or torch tensor."
        )
    if not hasattr(tensor, "shape"):
        raise TypeError(
            f"Object of type {type(tensor).__name__} lacks 'shape' attribute. "
            "Expected numpy array or torch tensor."
        )
    if not hasattr(tensor, "dtype"):
        raise TypeError(
            f"Object of type {type(tensor).__name__} lacks 'dtype' attribute. "
            "Expected numpy array or torch tensor."
        )

    # Build canonical representation
    shape_bytes = json.dumps(list(tensor.shape), separators=(",", ":")).encode("utf-8")
    dtype_bytes = str(tensor.dtype).encode("utf-8")
    data_bytes = tensor.tobytes()

    canonical = shape_bytes + b"|" + dtype_bytes + b"|" + data_bytes
    return fingerprint_bytes(canonical)


def fingerprint_dict(data: Dict[str, Any]) -> str:
    """
    Compute SHA-256 fingerprint of a dictionary.

    The dictionary is serialized to JSON with sorted keys and compact
    separators to ensure consistent fingerprints regardless of insertion
    order.

    Args:
        data: Dictionary to fingerprint. All keys and values must be
            JSON-serializable.

    Returns:
        A 64-character lowercase hexadecimal string representing the SHA-256
        hash of the canonical JSON representation.

    Raises:
        TypeError: If data is not a dictionary or contains non-serializable
            values.

    Note:
        This function is deterministic: dictionaries with the same content
        will produce the same fingerprint regardless of key insertion order.
        The JSON serialization uses sort_keys=True and compact separators.

    Example:
        >>> fingerprint_dict({"b": 1, "a": 2})
        >>> fingerprint_dict({"a": 2, "b": 1})  # Same as above
    """
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict, got {type(data).__name__}")

    # Use sorted keys and compact separators for deterministic output
    json_str = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return fingerprint_bytes(json_str.encode("utf-8"))


def fingerprint_sample(sample: Any) -> str:
    """
    Compute SHA-256 fingerprint of a data sample.

    This is the main entry point for fingerprinting arbitrary data samples.
    It automatically detects the sample type and routes to the appropriate
    specialized fingerprinting function.

    Supported types:
        - bytes: Raw byte data
        - str: Text strings (UTF-8 encoded)
        - dict: Dictionaries (JSON serialized with sorted keys)
        - Tensor-like: Objects with shape, dtype, and tobytes attributes
          (numpy arrays, PyTorch tensors)

    Args:
        sample: The data sample to fingerprint.

    Returns:
        A 64-character lowercase hexadecimal string representing the SHA-256
        hash of the sample.

    Raises:
        TypeError: If the sample type is not supported.

    Note:
        This function is deterministic: the same sample will always produce
        the same fingerprint.

    Example:
        >>> fingerprint_sample(b"raw bytes")
        >>> fingerprint_sample("text string")
        >>> fingerprint_sample({"key": "value"})
    """
    if isinstance(sample, bytes):
        return fingerprint_bytes(sample)
    elif isinstance(sample, str):
        return fingerprint_text(sample)
    elif isinstance(sample, dict):
        return fingerprint_dict(sample)
    elif hasattr(sample, "tobytes") and hasattr(sample, "shape"):
        return fingerprint_tensor(sample)
    else:
        raise TypeError(
            f"Unsupported sample type: {type(sample).__name__}. "
            "Supported types: bytes, str, dict, numpy.ndarray, torch.Tensor"
        )


def merkle_root(fingerprints: List[str]) -> str:
    """
    Compute the Merkle root of a list of fingerprints.

    A Merkle tree is a binary tree where each leaf node contains a fingerprint
    and each non-leaf node contains the hash of its children. The root of this
    tree provides a single fingerprint that represents all input fingerprints.

    Algorithm:
        1. If the list is empty, raise ValueError
        2. If the list has one item, return that item
        3. If the list has an odd number of items, duplicate the last item
        4. Pair adjacent items, concatenate, and hash each pair
        5. Recurse until a single root remains

    Args:
        fingerprints: List of fingerprint strings (64-char hex each).

    Returns:
        A 64-character lowercase hexadecimal string representing the Merkle
        root of all input fingerprints.

    Raises:
        ValueError: If the fingerprints list is empty.

    Note:
        This function is deterministic: the same list of fingerprints in the
        same order will always produce the same Merkle root. Order matters -
        different orderings produce different roots.

    Example:
        >>> fp1 = fingerprint_text("sample1")
        >>> fp2 = fingerprint_text("sample2")
        >>> root = merkle_root([fp1, fp2])
    """
    if not fingerprints:
        raise ValueError("Cannot compute Merkle root of empty list")

    if len(fingerprints) == 1:
        return fingerprints[0]

    # Make a copy to avoid modifying the input
    current_level = list(fingerprints)

    while len(current_level) > 1:
        # If odd number of items, duplicate the last one
        if len(current_level) % 2 == 1:
            current_level.append(current_level[-1])

        # Compute next level by hashing pairs
        next_level = []
        for i in range(0, len(current_level), 2):
            left = current_level[i]
            right = current_level[i + 1]
            # Concatenate as bytes and hash
            combined = (left + right).encode("utf-8")
            pair_hash = fingerprint_bytes(combined)
            next_level.append(pair_hash)

        current_level = next_level

    return current_level[0]


class FingerprintCache:
    """
    Thread-safe LRU cache for fingerprint operations.

    This cache stores computed fingerprints to avoid redundant computation
    for frequently accessed data. It uses an LRU (Least Recently Used)
    eviction policy to bound memory usage.

    The cache is thread-safe, using a lock to protect all operations. This
    allows safe concurrent access from multiple threads.

    Attributes:
        max_size: Maximum number of entries the cache can hold.

    Example:
        >>> cache = FingerprintCache(max_size=1000)
        >>> cache.put("key1", "fingerprint1")
        >>> cache.get("key1")
        'fingerprint1'
        >>> cache.stats()
        {'size': 1, 'max_size': 1000, 'hits': 1, 'misses': 0}
    """

    def __init__(self, max_size: int = 10000) -> None:
        """
        Initialize the fingerprint cache.

        Args:
            max_size: Maximum number of entries to cache. When this limit is
                reached, the least recently used entry is evicted. Defaults
                to 10000.

        Raises:
            ValueError: If max_size is less than 1.
        """
        if max_size < 1:
            raise ValueError("max_size must be at least 1")

        self._max_size = max_size
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[str]:
        """
        Retrieve a fingerprint from the cache.

        If the key exists, it is moved to the end of the LRU order (most
        recently used position).

        Args:
            key: The cache key to look up.

        Returns:
            The cached fingerprint string if found, None otherwise.

        Note:
            This operation is thread-safe.
        """
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            else:
                self._misses += 1
                return None

    def put(self, key: str, value: str) -> None:
        """
        Store a fingerprint in the cache.

        If the key already exists, its value is updated and it is moved to
        the most recently used position. If the cache is at capacity, the
        least recently used entry is evicted.

        Args:
            key: The cache key.
            value: The fingerprint value to cache.

        Note:
            This operation is thread-safe.
        """
        with self._lock:
            if key in self._cache:
                # Update existing entry and move to end
                self._cache.move_to_end(key)
                self._cache[key] = value
            else:
                # Add new entry
                self._cache[key] = value
                # Evict oldest if over capacity
                while len(self._cache) > self._max_size:
                    self._cache.popitem(last=False)

    def clear(self) -> None:
        """
        Remove all entries from the cache.

        This also resets the hit and miss counters.

        Note:
            This operation is thread-safe.
        """
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> Dict[str, int]:
        """
        Get cache statistics.

        Returns:
            A dictionary with the following keys:
                - 'size': Current number of entries in the cache
                - 'max_size': Maximum capacity of the cache
                - 'hits': Number of successful cache lookups
                - 'misses': Number of failed cache lookups

        Note:
            This operation is thread-safe.
        """
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
            }
