# Origin Integration Guide

This guide explains how to integrate Origin into machine learning pipelines for automatic provenance tracking.

## 1. Quick Start

### Installation

```bash
pip install origin-provenance
```

### Minimal Example

```python
from origin.storage.database import ProvenanceDatabase
from origin.hooks.base import BaseHook

# Initialize database
db = ProvenanceDatabase("./provenance.db")

# Start a training session
session = db.begin_session(config_hash="sha256_of_config")

# Your training loop...
# (See framework-specific examples below)

# End session
db.end_session(session.session_id, status="completed")
db.close()
```

## 2. PyTorch Integration

Origin provides `DataLoaderHook` for instrumenting PyTorch DataLoaders.

### Requirements

```bash
pip install torch>=2.0
```

### Basic Usage

```python
from torch.utils.data import DataLoader, TensorDataset
import torch

from origin.storage.database import ProvenanceDatabase
from origin.hooks.pytorch import DataLoaderHook

# Your dataset and dataloader
dataset = TensorDataset(torch.randn(1000, 784), torch.randint(0, 10, (1000,)))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize Origin
db = ProvenanceDatabase("./provenance.db")
session = db.begin_session(config_hash="experiment_v1")

# Create the hook
hook = DataLoaderHook(
    db=db,
    session_id=session.session_id,
    source_id="mnist_train",
    license_id="CC-BY-SA-4.0"
)

# Wrap your dataloader
for batch in hook.wrap(dataloader):
    inputs, labels = batch
    # batch is unchanged - train normally
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

# Check statistics
stats = hook.get_stats()
print(f"Recorded {stats['samples_observed']} samples in {stats['batches_observed']} batches")

# End session
db.end_session(session.session_id, status="completed")
```

### Handling Different Batch Formats

DataLoaderHook handles common DataLoader output formats:

```python
# Single tensor: Each row (dim 0) is a sample
dataloader = DataLoader(...)  # yields tensor of shape (batch, features)

# Tuple of tensors: (data, labels, ...)
dataloader = DataLoader(...)  # yields (inputs, labels)

# Dictionary batch: {"input_ids": tensor, "attention_mask": tensor, ...}
dataloader = DataLoader(...)  # yields dict of tensors
```

### Configuration Options

```python
hook = DataLoaderHook(
    db=db,
    session_id=session.session_id,
    source_id="dataset_name",      # Identifier for this data source
    license_id="MIT",               # SPDX license identifier
    record_gradients=False,         # Include gradient info in metadata
)
```

## 3. HuggingFace Integration

Origin provides `DatasetHook` for instrumenting HuggingFace datasets.

### Requirements

```bash
pip install datasets>=2.0
```

### Basic Usage

```python
from datasets import load_dataset

from origin.storage.database import ProvenanceDatabase
from origin.hooks.huggingface import DatasetHook

# Load a HuggingFace dataset
dataset = load_dataset("imdb", split="train")

# Initialize Origin
db = ProvenanceDatabase("./provenance.db")
session = db.begin_session(config_hash="imdb_training")

# Create the hook
hook = DatasetHook(
    db=db,
    session_id=session.session_id,
    source_id="imdb",
    license_id="unknown"  # Check dataset license
)

# Option 1: Wrap batch iteration
for batch in hook.wrap(dataset.iter(batch_size=32)):
    # batch is dict: {"text": [...], "label": [...]}
    process_batch(batch)

# Option 2: Wrap individual sample iteration
for sample in hook.wrap_dataset(dataset):
    # sample is dict: {"text": "...", "label": 0}
    process_sample(sample)
```

### Handling Streaming Datasets

For large datasets loaded with `streaming=True`:

```python
from datasets import load_dataset

# Load streaming dataset
dataset = load_dataset("imdb", split="train", streaming=True)

# Create hook
hook = DatasetHook(db, session_id, source_id="imdb")

# Use wrap_streaming for IterableDataset
for sample in hook.wrap_streaming(dataset):
    process_sample(sample)
```

### Multiple Data Sources

Track provenance from multiple datasets:

```python
# Load multiple datasets
train_data = load_dataset("squad", split="train")
val_data = load_dataset("squad", split="validation")

# Create separate hooks for each source
train_hook = DatasetHook(db, session_id, source_id="squad_train", license_id="CC-BY-SA-4.0")
val_hook = DatasetHook(db, session_id, source_id="squad_val", license_id="CC-BY-SA-4.0")

# Track training data
for batch in train_hook.wrap(train_data.iter(batch_size=16)):
    train_step(batch)

# Track validation data
for batch in val_hook.wrap(val_data.iter(batch_size=16)):
    evaluate(batch)
```

## 4. Custom Data Loaders

For custom data pipelines, extend `BaseHook`:

```python
from typing import Any, Dict, List, Tuple

from origin.hooks.base import BaseHook
from origin.storage.database import ProvenanceDatabase

class CustomHook(BaseHook):
    """Hook for custom data format."""

    def __init__(
        self,
        db: ProvenanceDatabase,
        session_id: str,
        source_id: str = "custom",
        license_id: str = None,
    ):
        super().__init__(db, session_id, source_id, license_id)

    def _extract_samples(self, batch: Any) -> List[Tuple[Any, Dict[str, Any]]]:
        """
        Extract individual samples from a batch.

        Must return a list of (sample_data, metadata) tuples.

        Args:
            batch: Your custom batch format

        Returns:
            List of tuples where:
            - sample_data: The data to fingerprint (bytes, str, dict, or tensor-like)
            - metadata: Dict of additional information about the sample
        """
        samples = []

        # Example: batch is a list of dicts
        for i, item in enumerate(batch):
            sample_data = item  # Will be fingerprinted
            metadata = {"index": i, "custom_field": "value"}
            samples.append((sample_data, metadata))

        return samples
```

### Required Methods

| Method | Description |
|--------|-------------|
| `_extract_samples(batch)` | Extract individual samples from a batch |

### Inherited Methods

| Method | Description |
|--------|-------------|
| `observe(batch)` | Record a batch (called by wrap) |
| `wrap(iterable)` | Wrap an iterable to observe all batches |
| `get_stats()` | Get observation statistics |
| `reset_stats()` | Reset counters |

### Fingerprint-Compatible Data Types

The following types can be fingerprinted:

| Type | Handling |
|------|----------|
| `bytes` | Used directly |
| `str` | UTF-8 encoded |
| `dict` | Canonical JSON (sorted keys) |
| Objects with `tobytes()` | Converted to bytes |

## 5. Session Management

### Manual Session Control

```python
from origin.storage.database import ProvenanceDatabase

db = ProvenanceDatabase("./provenance.db")

# Start session with config hash
session = db.begin_session(config_hash="sha256_of_training_config")

# Access session info
print(f"Session ID: {session.session_id}")
print(f"Created: {session.created_at}")

# Your training code...

# End session with status
db.end_session(session.session_id, status="completed")

# Or mark as failed
db.end_session(session.session_id, status="failed")
```

### Resuming Sessions

Sessions cannot be resumed after ending. For multi-stage training, either:

1. Use a single session for the entire pipeline
2. Create linked sessions with shared identifiers in metadata

```python
# Option 1: Single long-running session
session = db.begin_session(config_hash="full_pipeline")
# ... phase 1 ...
# ... phase 2 ...
# ... phase 3 ...
db.end_session(session.session_id, status="completed")

# Option 2: Linked sessions
phase1 = db.begin_session(config_hash="phase1")
# ... phase 1 ...
db.end_session(phase1.session_id, status="completed")

# Reference previous session in metadata
phase2 = db.begin_session(config_hash="phase2")
# Store phase1.session_id in your tracking system
```

### Handling Failures

```python
import sys

try:
    session = db.begin_session(config_hash="experiment")

    for batch in hook.wrap(dataloader):
        train_step(batch)

    db.end_session(session.session_id, status="completed")

except Exception as e:
    # Mark session as failed
    db.end_session(session.session_id, status="failed")
    raise

finally:
    db.close()
```

## 6. License Metadata

### Adding License Information to Sources

```python
from origin.core.record import SourceRecord

# Register a source with license
source = SourceRecord(
    source_id="my_dataset",
    source_type="file",
    source_path="/data/training_data.csv",
    license_id="MIT",  # SPDX identifier
    first_seen="2025-01-15T00:00:00Z"
)
db.record_source(source)

# Use this source_id in your hook
hook = DataLoaderHook(db, session_id, source_id="my_dataset")
```

### Custom License Definitions

```python
from origin.core.record import LicenseRecord

# Define a custom license
custom_license = LicenseRecord(
    license_id="CUSTOM-1.0",
    license_name="Custom Dataset License v1.0",
    license_url="https://example.com/license",
    permissions=("use", "modify"),
    restrictions=("no-redistribution", "no-commercial"),
    conditions=("attribution",),
    copyleft=False
)
db.record_license(custom_license)

# Use in source registration
source = SourceRecord(
    source_id="proprietary_data",
    source_type="database",
    source_path="internal://dataset/v2",
    license_id="CUSTOM-1.0",
    first_seen="2025-01-15T00:00:00Z"
)
db.record_source(source)
```

### Handling Unknown Licenses

```python
# When license is unknown, use "unknown"
hook = DataLoaderHook(
    db, session_id,
    source_id="scraped_data",
    license_id="unknown"  # Flags for review
)

# Check for unknown licenses later
from origin.query.engine import QueryEngine

engine = QueryEngine(db)
samples = engine.find_samples_by_license("unknown")
print(f"Found {len(samples)} samples with unknown license")
```

## 7. Querying During Training

### Real-time Monitoring

```python
from origin.query.engine import QueryEngine

engine = QueryEngine(db)

# Check current statistics
stats = hook.get_stats()
print(f"Batches: {stats['batches_observed']}")
print(f"Samples: {stats['samples_observed']}")
print(f"Unique: {stats['unique_samples']}")

# Query license breakdown
breakdown = engine.get_license_breakdown(session.session_id)
for license_id, count in breakdown.items():
    print(f"  {license_id}: {count} samples")
```

### Progress Tracking

```python
import time

start_time = time.time()
total_batches = len(dataloader)

for i, batch in enumerate(hook.wrap(dataloader)):
    train_step(batch)

    # Log progress every 100 batches
    if i % 100 == 0:
        stats = hook.get_stats()
        elapsed = time.time() - start_time
        print(f"Batch {i}/{total_batches}")
        print(f"  Samples recorded: {stats['samples_observed']}")
        print(f"  Unique samples: {stats['unique_samples']}")
        print(f"  Time elapsed: {elapsed:.1f}s")
```

## 8. Post-Training Export

### MLflow Integration

```python
from origin.export.formats import ProvenanceExporter
import mlflow
import json

# Export provenance data
exporter = ProvenanceExporter(db)
mlflow_data = exporter.to_mlflow(session.session_id)

# Log to MLflow
with mlflow.start_run():
    # Log tags
    for key, value in mlflow_data["tags"].items():
        mlflow.set_tag(key, value)

    # Log params
    for key, value in mlflow_data["params"].items():
        mlflow.log_param(key, value)

    # Log artifacts
    for name, content in mlflow_data["artifacts"].items():
        with open(name, "w") as f:
            f.write(content)
        mlflow.log_artifact(name)
```

### Weights and Biases Integration

```python
from origin.export.formats import ProvenanceExporter
import wandb

# Export provenance data
exporter = ProvenanceExporter(db)
wandb_data = exporter.to_wandb(session.session_id)

# Log to W&B
wandb.init(project="my_project")
wandb.config.update(wandb_data["metadata"])
wandb.log({"provenance_description": wandb_data["description"]})
```

### HuggingFace Hub Integration

```python
from origin.export.formats import ProvenanceExporter

# Export in HuggingFace format
exporter = ProvenanceExporter(db)
hf_data = exporter.to_huggingface(session.session_id)

# Add to model card
model_card = f"""
---
license: {hf_data["license"]}
datasets: {hf_data["datasets"]}
---

# Model Card

## Training Data Provenance

- Total samples: {hf_data["training_data"]["samples"]}
- Provenance fingerprint: {hf_data["training_data"]["provenance_fingerprint"]}

## License Information

{hf_data["license"]}
"""

# Save model card
with open("README.md", "w") as f:
    f.write(model_card)
```

### JSON Export for Compliance

```python
from origin.export.formats import ProvenanceExporter

exporter = ProvenanceExporter(db)

# Full JSON export
json_data = exporter.to_json(session.session_id, pretty=True)
with open("provenance_audit.json", "w") as f:
    f.write(json_data)

# Streaming JSONL for large exports
lines = exporter.to_jsonl(session.session_id, "provenance_full.jsonl")
print(f"Exported {lines} records")
```

### Provenance Card Generation

```python
from origin.cards.generator import ProvenanceCardGenerator

generator = ProvenanceCardGenerator(db)
card = generator.generate(session.session_id)

# Save as markdown
with open("PROVENANCE_CARD.md", "w") as f:
    f.write(card)
```
