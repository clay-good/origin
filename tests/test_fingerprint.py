"""
Tests for Origin fingerprinting functions.

Tests cover:
- Individual fingerprinting functions (bytes, text, dict, sample)
- Merkle root computation
- Fingerprint cache with LRU eviction
- Determinism verification
- Thread safety
"""

import threading
import unittest

from origin.core.fingerprint import (
    fingerprint_bytes,
    fingerprint_text,
    fingerprint_dict,
    fingerprint_sample,
    merkle_root,
    FingerprintCache,
)
from tests.fixtures.sample_data import (
    SAMPLE_TEXT,
    SAMPLE_DICT,
    SAMPLE_BYTES,
    create_mock_tensor,
)


class TestFingerprintBytes(unittest.TestCase):
    """Tests for fingerprint_bytes function."""

    def test_returns_64_char_hex(self):
        """Fingerprint should return a 64-character hex string."""
        result = fingerprint_bytes(b"test")
        self.assertEqual(len(result), 64)
        self.assertTrue(all(c in '0123456789abcdef' for c in result))

    def test_deterministic(self):
        """Same input should always produce same output."""
        result1 = fingerprint_bytes(b"test")
        result2 = fingerprint_bytes(b"test")
        self.assertEqual(result1, result2)

    def test_different_input_different_output(self):
        """Different inputs should produce different fingerprints."""
        result1 = fingerprint_bytes(b"test1")
        result2 = fingerprint_bytes(b"test2")
        self.assertNotEqual(result1, result2)

    def test_empty_bytes(self):
        """Empty bytes should produce a valid fingerprint."""
        result = fingerprint_bytes(b"")
        self.assertEqual(len(result), 64)

    def test_large_input(self):
        """Large input should produce a valid fingerprint."""
        large_data = b"x" * 1_000_000
        result = fingerprint_bytes(large_data)
        self.assertEqual(len(result), 64)

    def test_sample_bytes_fixture(self):
        """Test with SAMPLE_BYTES fixture."""
        result = fingerprint_bytes(SAMPLE_BYTES)
        self.assertEqual(len(result), 64)
        # Verify determinism with fixture
        self.assertEqual(result, fingerprint_bytes(SAMPLE_BYTES))


class TestFingerprintText(unittest.TestCase):
    """Tests for fingerprint_text function."""

    def test_matches_utf8_bytes(self):
        """Text fingerprint should match UTF-8 encoded bytes fingerprint."""
        text = "hello"
        self.assertEqual(
            fingerprint_text(text),
            fingerprint_bytes(text.encode('utf-8'))
        )

    def test_unicode(self):
        """Unicode text should produce a valid fingerprint."""
        result = fingerprint_text("unicode: \u00e9\u00e0\u00fc\u4e2d\u6587")
        self.assertEqual(len(result), 64)

    def test_empty_text(self):
        """Empty text should produce a valid fingerprint."""
        result = fingerprint_text("")
        self.assertEqual(len(result), 64)
        self.assertEqual(result, fingerprint_bytes(b""))

    def test_sample_text_fixture(self):
        """Test with SAMPLE_TEXT fixture."""
        result = fingerprint_text(SAMPLE_TEXT)
        self.assertEqual(len(result), 64)

    def test_deterministic(self):
        """Same text should always produce same fingerprint."""
        self.assertEqual(
            fingerprint_text("hello world"),
            fingerprint_text("hello world")
        )


class TestFingerprintDict(unittest.TestCase):
    """Tests for fingerprint_dict function."""

    def test_key_order_independent(self):
        """Dictionary fingerprint should not depend on key order."""
        self.assertEqual(
            fingerprint_dict({"b": 1, "a": 2}),
            fingerprint_dict({"a": 2, "b": 1})
        )

    def test_nested_dict(self):
        """Nested dictionaries should produce valid fingerprints."""
        result = fingerprint_dict({"outer": {"inner": "value"}})
        self.assertEqual(len(result), 64)

    def test_empty_dict(self):
        """Empty dictionary should produce a valid fingerprint."""
        result = fingerprint_dict({})
        self.assertEqual(len(result), 64)

    def test_sample_dict_fixture(self):
        """Test with SAMPLE_DICT fixture."""
        result = fingerprint_dict(SAMPLE_DICT)
        self.assertEqual(len(result), 64)

    def test_nested_key_order_independent(self):
        """Nested dictionaries should also be order-independent."""
        dict1 = {"outer": {"b": 1, "a": 2}}
        dict2 = {"outer": {"a": 2, "b": 1}}
        self.assertEqual(fingerprint_dict(dict1), fingerprint_dict(dict2))

    def test_different_values_different_fingerprint(self):
        """Different values should produce different fingerprints."""
        self.assertNotEqual(
            fingerprint_dict({"key": "value1"}),
            fingerprint_dict({"key": "value2"})
        )


class TestFingerprintSample(unittest.TestCase):
    """Tests for fingerprint_sample function."""

    def test_bytes_input(self):
        """Bytes input should produce valid fingerprint."""
        result = fingerprint_sample(b"test data")
        self.assertEqual(len(result), 64)

    def test_string_input(self):
        """String input should produce valid fingerprint."""
        result = fingerprint_sample("test string")
        self.assertEqual(len(result), 64)

    def test_dict_input(self):
        """Dict input should produce valid fingerprint."""
        result = fingerprint_sample({"key": "value"})
        self.assertEqual(len(result), 64)

    def test_mock_tensor_input(self):
        """Mock tensor with tobytes should produce valid fingerprint."""
        tensor = create_mock_tensor()
        result = fingerprint_sample(tensor)
        self.assertEqual(len(result), 64)

    def test_unsupported_type_raises(self):
        """Unsupported types should raise TypeError."""
        with self.assertRaises(TypeError):
            fingerprint_sample(12345)

        with self.assertRaises(TypeError):
            fingerprint_sample([1, 2, 3])


class TestMerkleRoot(unittest.TestCase):
    """Tests for merkle_root function."""

    def test_single_item(self):
        """Single item should return that item unchanged."""
        result = merkle_root(["abc123"])
        self.assertEqual(result, "abc123")

    def test_two_items(self):
        """Two items should produce a valid 64-char hash."""
        result = merkle_root(["a" * 64, "b" * 64])
        self.assertEqual(len(result), 64)

    def test_three_items(self):
        """Three items should work with padding."""
        result = merkle_root(["a" * 64, "b" * 64, "c" * 64])
        self.assertEqual(len(result), 64)

    def test_deterministic(self):
        """Same items should always produce same root."""
        items = ["a" * 64, "b" * 64, "c" * 64]
        self.assertEqual(merkle_root(items), merkle_root(items))

    def test_order_matters(self):
        """Different order should produce different root."""
        items1 = ["a" * 64, "b" * 64]
        items2 = ["b" * 64, "a" * 64]
        self.assertNotEqual(merkle_root(items1), merkle_root(items2))

    def test_empty_raises(self):
        """Empty list should raise ValueError."""
        with self.assertRaises(ValueError):
            merkle_root([])

    def test_power_of_two_items(self):
        """Power of two items should work correctly."""
        items = [str(i) * 64 for i in range(8)]
        result = merkle_root(items)
        self.assertEqual(len(result), 64)


class TestFingerprintCache(unittest.TestCase):
    """Tests for FingerprintCache class."""

    def test_put_and_get(self):
        """Put and get should work correctly."""
        cache = FingerprintCache(max_size=10)
        cache.put("key", "value")
        self.assertEqual(cache.get("key"), "value")

    def test_get_missing_returns_none(self):
        """Getting missing key should return None."""
        cache = FingerprintCache(max_size=10)
        self.assertIsNone(cache.get("missing"))

    def test_lru_eviction(self):
        """Least recently used items should be evicted when at capacity."""
        cache = FingerprintCache(max_size=2)
        cache.put("a", "1")
        cache.put("b", "2")
        cache.put("c", "3")  # Should evict "a"
        self.assertIsNone(cache.get("a"))
        self.assertEqual(cache.get("b"), "2")
        self.assertEqual(cache.get("c"), "3")

    def test_access_updates_lru(self):
        """Accessing an item should update its LRU position."""
        cache = FingerprintCache(max_size=2)
        cache.put("a", "1")
        cache.put("b", "2")
        cache.get("a")  # Access "a" to make it most recently used
        cache.put("c", "3")  # Should evict "b" instead of "a"
        self.assertEqual(cache.get("a"), "1")
        self.assertIsNone(cache.get("b"))
        self.assertEqual(cache.get("c"), "3")

    def test_stats(self):
        """Stats should track hits and misses."""
        cache = FingerprintCache(max_size=10)
        cache.put("key", "value")
        cache.get("key")  # hit
        cache.get("missing")  # miss
        stats = cache.stats()
        self.assertEqual(stats['hits'], 1)
        self.assertEqual(stats['misses'], 1)
        self.assertEqual(stats['size'], 1)

    def test_clear(self):
        """Clear should remove all entries."""
        cache = FingerprintCache(max_size=10)
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.clear()
        self.assertIsNone(cache.get("key1"))
        self.assertIsNone(cache.get("key2"))
        self.assertEqual(cache.stats()['size'], 0)

    def test_thread_safety(self):
        """Cache should be thread-safe."""
        cache = FingerprintCache(max_size=100)
        errors = []

        def worker(thread_id):
            try:
                for i in range(50):
                    key = f"thread_{thread_id}_key_{i}"
                    cache.put(key, f"value_{i}")
                    cache.get(key)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Thread safety errors: {errors}")


class TestDeterminism(unittest.TestCase):
    """Tests verifying deterministic behavior across functions."""

    def test_fingerprint_bytes_determinism(self):
        """Multiple calls with same input should always match."""
        data = b"determinism test data"
        results = [fingerprint_bytes(data) for _ in range(100)]
        self.assertEqual(len(set(results)), 1)

    def test_fingerprint_text_determinism(self):
        """Multiple calls with same text should always match."""
        text = "determinism test text"
        results = [fingerprint_text(text) for _ in range(100)]
        self.assertEqual(len(set(results)), 1)

    def test_fingerprint_dict_determinism(self):
        """Multiple calls with same dict should always match."""
        data = {"nested": {"key": "value"}, "list": [1, 2, 3]}
        results = [fingerprint_dict(data) for _ in range(100)]
        self.assertEqual(len(set(results)), 1)

    def test_merkle_root_determinism(self):
        """Multiple calls with same items should always match."""
        items = [fingerprint_bytes(f"item_{i}".encode()) for i in range(10)]
        results = [merkle_root(items) for _ in range(100)]
        self.assertEqual(len(set(results)), 1)


if __name__ == "__main__":
    unittest.main()
