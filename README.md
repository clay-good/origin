# Origin

Runtime data provenance for AI pipelines.

Origin is a lightweight library that tracks data as it flows through machine learning training pipelines, generating cryptographic fingerprints and maintaining license metadata for compliance and auditability.

## The Problem

Modern AI models are trained on massive datasets aggregated from diverse sources. This creates significant challenges:

- **Compliance requirements**: Regulations like the EU AI Act mandate transparency about training data
- **License conflicts**: Datasets may contain data under incompatible licenses (e.g., mixing GPL and proprietary content)
- **Audit trails**: Organizations need to demonstrate what data was used to train specific models
- **Reproducibility**: Without provenance records, reproducing training runs is unreliable

Studies have found that popular ML datasets frequently contain mislabeled or miscategorized license information, exposing model creators to legal and compliance risks.

## The Solution

Origin provides a zero-dependency observation layer that:

- **Records provenance automatically** without modifying your training pipeline
- **Generates cryptographic fingerprints** for individual samples and batches
- **Tracks license metadata** from data sources through to model training
- **Detects potential conflicts** between incompatible licenses
- **Exports audit-ready reports** in standard formats

Unlike experiment tracking tools, Origin focuses exclusively on data lineage. It observes and records, never transforms.

## Quick Start

```bash
pip install origin-provenance
```

```python
from origin.storage.database import ProvenanceDatabase
from origin.hooks.pytorch import DataLoaderHook

# Initialize database
db = ProvenanceDatabase("./provenance.db")
session = db.begin_session(config_hash="experiment_v1")

# Create hook for your DataLoader
hook = DataLoaderHook(
    db=db,
    session_id=session.session_id,
    source_id="training_data",
    license_id="MIT"
)

# Wrap your training loop - data flows through unchanged
for batch in hook.wrap(dataloader):
    loss = train_step(batch)  # batch is not modified

# End session
db.end_session(session.session_id, status="completed")
```

## How It Works

Origin operates as a passive observation layer alongside your data pipeline:

```
┌─────────────┐    ┌────────────────────┐    ┌──────────────┐
│ Data Source │───▶│ Origin Hook        │───▶│ Training     │
│             │    │ (observe only)     │    │ Loop         │
└─────────────┘    └────────────────────┘    └──────────────┘
                          │
                          ▼
                   ┌────────────────────┐
                   │ Provenance DB      │
                   │ - Fingerprints     │
                   │ - Licenses         │
                   │ - Batch records    │
                   └────────────────────┘
```

**Fingerprinting**: Each data sample is hashed using SHA-256. Batches are aggregated using Merkle trees, enabling efficient verification and tamper detection.

**Storage**: All provenance data is stored in a local SQLite database. No network connectivity required. No data leaves your system.

**Querying**: After training, query the database to trace samples, check license compatibility, or generate compliance reports.

## Key Features

| Feature | Description |
|---------|-------------|
| **Automatic instrumentation** | Wrap DataLoaders and datasets with minimal code changes |
| **Data fingerprinting** | SHA-256 hashes with Merkle tree batch aggregation |
| **License propagation** | Track licenses from sources through batches to sessions |
| **Conflict detection** | Identify incompatible license combinations |
| **Provenance cards** | Generate human-readable Markdown reports |
| **Export integrations** | MLflow, Weights & Biases, HuggingFace Hub formats |
| **Zero dependencies** | Core library uses only Python standard library |

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     ORIGIN PROVENANCE                            │
│                                                                  │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐  │
│  │ Fingerprint     │───▶│ Storage Layer   │◀───│ Query       │  │
│  │ Engine          │    │ (SQLite)        │    │ Engine      │  │
│  └─────────────────┘    └─────────────────┘    └─────────────┘  │
│                                │                      │          │
│                                ▼                      ▼          │
│                         ┌─────────────┐        ┌───────────┐    │
│                         │ Export      │        │ Card      │    │
│                         │ System      │        │ Generator │    │
│                         └─────────────┘        └───────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

See [docs/architecture.md](docs/architecture.md) for complete system design.

## Deterministic Logic

Origin uses **zero AI or LLM** in its core operations. All fingerprinting, hashing, and record-keeping is fully deterministic:

- Same input always produces same fingerprint
- Merkle roots are computed using standard cryptographic algorithms
- License compatibility uses predefined, auditable rules
- No machine learning, neural networks, or probabilistic operations

This guarantees reproducibility and auditability.

## CLI Reference

| Command | Description |
|---------|-------------|
| `origin init` | Initialize a new provenance database |
| `origin status` | Show database statistics |
| `origin sessions` | List all training sessions |
| `origin inspect SESSION` | Show session details |
| `origin query LICENSE` | Check if license is present |
| `origin card SESSION` | Generate provenance card |
| `origin export SESSION` | Export session data |
| `origin conflicts` | List license conflicts |
| `origin trace SAMPLE` | Trace sample to source |
| `origin version` | Show version information |

See [docs/cli-reference.md](docs/cli-reference.md) for complete documentation.

## Safety Guarantees

Origin is designed with safety as a core principle:

| Guarantee | Implementation |
|-----------|----------------|
| **Read-only by default** | Hooks observe data without modification |
| **Explicit writes** | Database commits require explicit calls |
| **No data modification** | Training data passes through unchanged |
| **SQL injection prevention** | All queries use parameterized statements |
| **Local-first** | No network connectivity required |
| **Deterministic** | Same input always produces same output |

## Limitations

Origin is intentionally focused and does not:

| Limitation | Reason |
|------------|--------|
| Store actual training data | Only metadata and fingerprints for privacy and scale |
| Make legal determinations | Flags conflicts for human review; not legal advice |
| Provide distributed storage | Local-first design; no built-in synchronization |
| Support non-Python pipelines | Python-only implementation |
| Track model versions | Data provenance only; model versioning out of scope |
| Provide a web interface | CLI and API only; no built-in dashboard |

## Examples

The [examples/](examples/) directory contains runnable examples for common use cases:

| Example | Description |
|---------|-------------|
| [basic_usage.py](examples/basic_usage.py) | Core functionality without frameworks |
| [pytorch_training.py](examples/pytorch_training.py) | PyTorch DataLoader integration |
| [huggingface_nlp.py](examples/huggingface_nlp.py) | HuggingFace datasets integration |
| [custom_data_loader.py](examples/custom_data_loader.py) | Building custom hooks for any data format |
| [csv_data_pipeline.py](examples/csv_data_pipeline.py) | Tabular data pipelines |
| [compliance_audit.py](examples/compliance_audit.py) | Regulatory compliance auditing |
| [multi_source_training.py](examples/multi_source_training.py) | Training on data from multiple sources |
| [cli_workflow.py](examples/cli_workflow.py) | CLI commands demonstration |
| [export_formats.py](examples/export_formats.py) | All export format options |
| [license_conflict_detection.py](examples/license_conflict_detection.py) | License conflict scenarios |

Run any example:

```bash
pip install -e .
python examples/basic_usage.py
```

## Documentation

- [Architecture](docs/architecture.md) - System design and database schema
- [CLI Reference](docs/cli-reference.md) - Complete command documentation
- [Integration Guide](docs/integration-guide.md) - PyTorch, HuggingFace, and custom integrations
- [License Taxonomy](docs/license-taxonomy.md) - License definitions and compatibility rules
- [Examples README](examples/README.md) - Detailed example descriptions and integration patterns
