"""
Origin hooks module.

Provides instrumentation hooks for observing data as it flows through
training pipelines. Hooks record provenance information without modifying
the data.

Available hooks:
    - BaseHook: Abstract base class for custom hooks (always available)
    - DataLoaderHook: PyTorch DataLoader instrumentation (requires torch)
    - DatasetHook: HuggingFace datasets instrumentation (requires datasets)

The PyTorch and HuggingFace hooks are optional - they are set to None if
their respective dependencies are not installed.
"""

from origin.hooks.base import BaseHook

# Attempt to import PyTorch hook
try:
    from origin.hooks.pytorch import DataLoaderHook, TORCH_AVAILABLE
except ImportError:
    DataLoaderHook = None  # type: ignore
    TORCH_AVAILABLE = False

# Attempt to import HuggingFace hook
try:
    from origin.hooks.huggingface import DatasetHook, DATASETS_AVAILABLE
except ImportError:
    DatasetHook = None  # type: ignore
    DATASETS_AVAILABLE = False

__all__ = [
    "BaseHook",
    "DataLoaderHook",
    "DatasetHook",
    "TORCH_AVAILABLE",
    "DATASETS_AVAILABLE",
]
