# Origin Examples

This directory contains runnable examples demonstrating how to integrate Origin into your ML pipelines.

## Quick Start

All examples can be run directly:

```bash
# Install Origin first
pip install -e .

# Run any example
python examples/basic_usage.py
```

## Examples Overview

| Example | Description | Dependencies |
|---------|-------------|--------------|
| [basic_usage.py](basic_usage.py) | Core functionality without frameworks | None (stdlib only) |
| [pytorch_training.py](pytorch_training.py) | PyTorch DataLoader integration | `torch>=2.0` |
| [huggingface_nlp.py](huggingface_nlp.py) | HuggingFace datasets integration | `datasets>=2.0` |
| [custom_data_loader.py](custom_data_loader.py) | Building custom hooks | None (stdlib only) |
| [csv_data_pipeline.py](csv_data_pipeline.py) | Tabular data pipelines | None (stdlib only) |
| [compliance_audit.py](compliance_audit.py) | Regulatory compliance auditing | None (stdlib only) |
| [multi_source_training.py](multi_source_training.py) | Multi-source training scenarios | None (stdlib only) |
| [cli_workflow.py](cli_workflow.py) | CLI commands demonstration | None (stdlib only) |
| [export_formats.py](export_formats.py) | Export format examples | None (stdlib only) |
| [license_conflict_detection.py](license_conflict_detection.py) | License conflict scenarios | None (stdlib only) |

## Example Descriptions

### basic_usage.py

**Start here if you're new to Origin.**

Demonstrates core concepts:
- Initializing the provenance database
- Registering data sources with licenses
- Recording samples and computing fingerprints
- Querying provenance data
- Generating provenance cards

```bash
python examples/basic_usage.py
```

### pytorch_training.py

**For PyTorch users.**

Shows how to wrap your DataLoader:

```python
from origin.hooks.pytorch import DataLoaderHook

hook = DataLoaderHook(db, session_id, source_id="mnist", license_id="CC-BY-4.0")

for batch in hook.wrap(dataloader):
    # batch is unchanged - train normally
    loss = train_step(batch)
```

Requires: `pip install torch>=2.0`

```bash
python examples/pytorch_training.py
```

### huggingface_nlp.py

**For HuggingFace users.**

Shows how to wrap HuggingFace datasets:

```python
from origin.hooks.huggingface import DatasetHook

hook = DatasetHook(db, session_id, source_id="imdb", license_id="CC-BY-4.0")

# Batch iteration
for batch in hook.wrap(dataset.iter(batch_size=32)):
    process_batch(batch)

# Sample iteration
for sample in hook.wrap_dataset(dataset):
    process_sample(sample)
```

Requires: `pip install datasets>=2.0`

```bash
python examples/huggingface_nlp.py
```

### custom_data_loader.py

**For custom data pipelines.**

Shows how to create hooks for any data format:

```python
from origin.hooks.base import BaseHook

class MyHook(BaseHook):
    def _extract_samples(self, batch):
        # Return list of (sample_data, metadata) tuples
        return [(item, {"index": i}) for i, item in enumerate(batch)]
```

```bash
python examples/custom_data_loader.py
```

### csv_data_pipeline.py

**For traditional ML with tabular data.**

Demonstrates:
- Loading CSV files with provenance tracking
- Registering multiple data sources
- Handling different licenses for different data files
- Detecting non-commercial restrictions

```bash
python examples/csv_data_pipeline.py
```

### compliance_audit.py

**For regulatory compliance (EU AI Act, etc.).**

Shows how to audit training data:
- Session overview and statistics
- Data source analysis
- License breakdown and compatibility
- Sample tracing
- Exporting for regulators
- Generating compliance reports

```bash
python examples/compliance_audit.py
```

### multi_source_training.py

**For training on mixed data sources.**

Demonstrates:
- Combining data from multiple sources
- Tracking licenses per-source
- Analyzing combined license terms
- Detecting license conflicts
- Documenting data composition

```bash
python examples/multi_source_training.py
```

### cli_workflow.py

**For command-line users.**

Demonstrates all Origin CLI commands:
- `origin init` - Initialize database
- `origin status` - View statistics
- `origin sessions` - List training sessions
- `origin inspect` - View session details
- `origin query` - Search by license
- `origin conflicts` - Check license conflicts
- `origin card` - Generate provenance card
- `origin export` - Export to JSON/JSONL
- `origin trace` - Trace sample lineage

```bash
python examples/cli_workflow.py
```

### export_formats.py

**For integrating with ML tracking tools.**

Shows all available export formats:
- JSON - Machine-readable audit format
- JSONL - Streaming format for large exports
- MLflow - Tags, params, and artifacts
- Weights & Biases - Config and metadata
- HuggingFace - Model card format
- Markdown - Human-readable provenance cards

```bash
python examples/export_formats.py
```

### license_conflict_detection.py

**For understanding license compatibility.**

Demonstrates 7 license scenarios:
1. Permissive license mix (MIT + Apache + BSD)
2. Permissive + Copyleft (MIT + GPL-3.0)
3. Copyleft version conflict (GPL-2.0 + GPL-3.0)
4. Copyleft + Proprietary conflict
5. Creative Commons mix (CC-BY + CC-BY-NC + CC0)
6. CC ShareAlike + NonCommercial
7. Unknown license handling

Includes a comprehensive license compatibility matrix.

```bash
python examples/license_conflict_detection.py
```

## Integration Patterns

### Minimal Integration (5 lines)

```python
from origin.storage.database import ProvenanceDatabase
from origin.hooks.pytorch import DataLoaderHook

db = ProvenanceDatabase("./provenance.db")
session = db.begin_session(config_hash="v1")
hook = DataLoaderHook(db, session.session_id, source_id="data", license_id="MIT")

for batch in hook.wrap(dataloader):
    train(batch)

db.end_session(session.session_id, status="completed")
```

### Full Integration

```python
from origin.storage.database import ProvenanceDatabase
from origin.core.record import SourceRecord
from origin.hooks.pytorch import DataLoaderHook
from origin.query.engine import QueryEngine
from origin.cards.generator import ProvenanceCardGenerator

# Initialize
db = ProvenanceDatabase("./provenance.db")

# Register sources
db.record_source(SourceRecord(
    source_id="my_dataset",
    source_type="file",
    source_path="/data/train.csv",
    license_id="MIT",
    first_seen="2025-01-01T00:00:00Z"
))

# Start session
session = db.begin_session(config_hash="experiment_v1")

# Create hook
hook = DataLoaderHook(db, session.session_id, source_id="my_dataset", license_id="MIT")

# Training loop
for epoch in range(num_epochs):
    for batch in hook.wrap(dataloader):
        train(batch)

# End session
db.end_session(session.session_id, status="completed")

# Query provenance
engine = QueryEngine(db)
summary = engine.get_session_summary(session.session_id)
licenses = engine.get_license_breakdown(session.session_id)

# Generate provenance card
generator = ProvenanceCardGenerator(db)
card = generator.generate(session.session_id)
with open("PROVENANCE_CARD.md", "w") as f:
    f.write(card)

db.close()
```

## Common Use Cases

### Tracking Data from Multiple Sources

```python
# Create separate hooks for each source
train_hook = DataLoaderHook(db, session_id, source_id="train", license_id="MIT")
val_hook = DataLoaderHook(db, session_id, source_id="val", license_id="Apache-2.0")

# Track each dataset
for batch in train_hook.wrap(train_loader):
    train(batch)

for batch in val_hook.wrap(val_loader):
    evaluate(batch)
```

### Checking License Compatibility

```python
from origin.core.license import LicenseAnalyzer

analyzer = LicenseAnalyzer()
result = analyzer.analyze_session(db, session_id)

if result["conflicts"]:
    print("License conflicts detected!")
    for conflict in result["conflicts"]:
        print(f"  {conflict['license_a']} + {conflict['license_b']}: {conflict['reason']}")
```

### Exporting for Auditors

```python
from origin.export.formats import ProvenanceExporter

exporter = ProvenanceExporter(db)

# JSON export
with open("audit.json", "w") as f:
    f.write(exporter.to_json(session_id))

# JSONL for large exports
exporter.to_jsonl(session_id, "audit.jsonl")
```

## Notes

- All examples use temporary directories and clean up after themselves
- Examples with optional dependencies check for availability and exit gracefully
- The core library has zero external dependencies
- PyTorch and HuggingFace integrations are optional

## Getting Help

- See [../docs/integration-guide.md](../docs/integration-guide.md) for detailed integration documentation
- See [../docs/architecture.md](../docs/architecture.md) for system design
- See [../docs/license-taxonomy.md](../docs/license-taxonomy.md) for license compatibility rules
