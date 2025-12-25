# Origin Architecture

This document describes the system design and architecture of Origin, a runtime data provenance library for AI pipelines.

## 1. Overview

Origin is a lightweight observation layer that tracks data as it flows through machine learning training pipelines. It generates cryptographic fingerprints of data samples, maintains license metadata, and stores provenance records for later auditing.

### Design Philosophy

Origin is built around these core principles:

1. **Zero External Dependencies**: Uses only Python standard library modules
2. **Single Responsibility**: Tracks provenance only; does not modify data
3. **Read-Safe by Default**: Observations are read-only; writes are explicit
4. **Deterministic Core**: Same input always produces same output; no AI/LLM usage
5. **Audit-First Design**: Optimizes for later auditability over runtime convenience

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DATA PIPELINE                                  │
│  ┌─────────────┐    ┌──────────────────────┐    ┌──────────────────┐   │
│  │ Data Source │───▶│ Instrumentation Hook │───▶│ Training Loop    │   │
│  └─────────────┘    └──────────────────────┘    └──────────────────┘   │
│                              │                                          │
└──────────────────────────────┼──────────────────────────────────────────┘
                               │ (observe only)
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         ORIGIN PROVENANCE                                │
│                                                                          │
│  ┌───────────────────┐    ┌─────────────────┐    ┌─────────────────┐   │
│  │ Fingerprint Engine│───▶│  Storage Layer  │◀───│  Query Engine   │   │
│  │                   │    │    (SQLite)     │    │                 │   │
│  │  - SHA-256 hash   │    │                 │    │  - License      │   │
│  │  - Merkle trees   │    │  - Sessions     │    │    queries      │   │
│  │  - Content types  │    │  - Batches      │    │  - Sample       │   │
│  └───────────────────┘    │  - Samples      │    │    tracing      │   │
│                           │  - Licenses     │    │  - Conflicts    │   │
│                           └─────────────────┘    └─────────────────┘   │
│                                   │                      │              │
│                                   ▼                      ▼              │
│                           ┌─────────────────┐    ┌─────────────────┐   │
│                           │  Export System  │    │ Card Generator  │   │
│                           │                 │    │                 │   │
│                           │  - JSON/JSONL   │    │  - Markdown     │   │
│                           │  - MLflow       │    │  - Human-       │   │
│                           │  - W&B          │    │    readable     │   │
│                           │  - HuggingFace  │    │    summaries    │   │
│                           └─────────────────┘    └─────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility |
|-----------|----------------|
| Instrumentation Hooks | Intercept data at pipeline boundaries without modifying flow |
| Fingerprint Engine | Generate content-addressable identifiers for samples and batches |
| Storage Layer | SQLite-based persistence optimized for audit queries |
| Query Engine | SQL-based interface for provenance audits |
| Export System | Format conversion for external tool integration |
| Card Generator | Human-readable provenance documentation |

## 3. Data Flow

### Registration Flow

```
1. Pipeline starts training session
2. Origin creates SessionRecord with unique ID and config hash
3. Data sources registered with license metadata
4. Session marked as "active"
```

### Observation Flow

```
1. DataLoader yields batch to training loop
2. Instrumentation hook intercepts batch (non-blocking)
3. For each sample in batch:
   a. Compute SHA-256 fingerprint
   b. Create ProvenanceRecord with metadata
   c. Store sample (INSERT OR IGNORE for deduplication)
4. Compute batch Merkle root from sample fingerprints
5. Create BatchRecord linking to session
6. Training loop receives unchanged batch
```

### Query Flow

```
1. User or system initiates query (CLI, API, or direct)
2. Query Engine translates request to SQL
3. Database returns matching records
4. Results formatted as requested (JSON, table, etc.)
```

## 4. Database Schema

Origin uses SQLite with a normalized schema optimized for audit queries.

### Entity-Relationship Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  ┌──────────────┐         ┌──────────────┐         ┌────────────┐   │
│  │   sessions   │         │   batches    │         │  samples   │   │
│  ├──────────────┤ 1     n ├──────────────┤ n     n ├────────────┤   │
│  │ session_id   │◀────────│ session_id   │─────────│ sample_id  │   │
│  │ created_at   │         │ batch_id     │         │ source_id  │──┐│
│  │ config_hash  │         │ batch_index  │    ┌────│ byte_size  │  ││
│  │ metadata     │         │ sample_count │    │    │ timestamp  │  ││
│  │ status       │         │ created_at   │    │    └────────────┘  ││
│  └──────────────┘         └──────────────┘    │                    ││
│         │                        │            │                    ││
│         │                        ▼            │                    ││
│         │                 ┌──────────────┐    │                    ││
│         │                 │batch_samples │    │                    ││
│         │                 ├──────────────┤    │                    ││
│         │                 │ batch_id     │    │                    ││
│         │                 │ sample_id    │◀───┘                    ││
│         │                 │ position     │                         ││
│         │                 └──────────────┘                         ││
│         │                                                          ││
│         │         ┌──────────────┐         ┌──────────────┐        ││
│         │         │license_      │         │   sources    │        ││
│         │         │conflicts     │         ├──────────────┤        ││
│         └────────▶├──────────────┤         │ source_id    │◀───────┘│
│                   │ conflict_id  │         │ source_type  │         │
│                   │ session_id   │         │ source_path  │         │
│                   │ license_a    │         │ license_id   │───┐     │
│                   │ license_b    │         │ first_seen   │   │     │
│                   │ conflict_type│         └──────────────┘   │     │
│                   │ detected_at  │                            │     │
│                   └──────────────┘         ┌──────────────┐   │     │
│                                            │   licenses   │   │     │
│                                            ├──────────────┤   │     │
│                                            │ license_id   │◀──┘     │
│                                            │ license_name │         │
│                                            │ permissions  │         │
│                                            │ restrictions │         │
│                                            │ conditions   │         │
│                                            │ copyleft     │         │
│                                            └──────────────┘         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Table Descriptions

#### sessions
Tracks individual training runs.

| Column | Type | Description |
|--------|------|-------------|
| session_id | TEXT | Primary key, UUID |
| created_at | TEXT | ISO 8601 timestamp |
| config_hash | TEXT | SHA-256 of training configuration |
| metadata | TEXT | JSON-encoded additional data |
| status | TEXT | active, completed, failed |

#### batches
Records batch-level aggregations.

| Column | Type | Description |
|--------|------|-------------|
| batch_id | TEXT | Primary key, Merkle root of sample fingerprints |
| session_id | TEXT | Foreign key to sessions |
| batch_index | INTEGER | Sequential batch number within session |
| sample_count | INTEGER | Number of samples in batch |
| created_at | TEXT | ISO 8601 timestamp |

#### samples
Individual data sample records (content-addressed).

| Column | Type | Description |
|--------|------|-------------|
| sample_id | TEXT | Primary key, SHA-256 fingerprint of content |
| source_id | TEXT | Foreign key to sources |
| content_type | TEXT | bytes, text, dict, tensor, unknown |
| byte_size | INTEGER | Size in bytes |
| first_seen | TEXT | ISO 8601 timestamp of first observation |

#### batch_samples
Junction table linking batches to samples (many-to-many).

| Column | Type | Description |
|--------|------|-------------|
| batch_id | TEXT | Foreign key to batches |
| sample_id | TEXT | Foreign key to samples |
| position | INTEGER | Position within batch |

#### sources
Data source information.

| Column | Type | Description |
|--------|------|-------------|
| source_id | TEXT | Primary key, fingerprint of source path |
| source_type | TEXT | file, url, dataset, api |
| source_path | TEXT | Path or URL to source |
| license_id | TEXT | SPDX identifier or custom |
| first_seen | TEXT | ISO 8601 timestamp |
| metadata | TEXT | JSON-encoded additional data |

#### licenses
License metadata definitions.

| Column | Type | Description |
|--------|------|-------------|
| license_id | TEXT | Primary key, SPDX identifier |
| license_name | TEXT | Human-readable name |
| license_url | TEXT | URL to license text |
| permissions | TEXT | JSON array of permissions |
| restrictions | TEXT | JSON array of restrictions |
| conditions | TEXT | JSON array of conditions |
| copyleft | INTEGER | 1 if copyleft, 0 otherwise |

#### license_conflicts
Records detected license incompatibilities.

| Column | Type | Description |
|--------|------|-------------|
| conflict_id | TEXT | Primary key, UUID |
| session_id | TEXT | Foreign key to sessions |
| license_a | TEXT | First license ID |
| license_b | TEXT | Second license ID |
| conflict_type | TEXT | Type of conflict detected |
| detected_at | TEXT | ISO 8601 timestamp |

## 5. Fingerprinting Algorithm

### Individual Sample Fingerprinting

Origin uses SHA-256 to generate 64-character hexadecimal fingerprints:

```
fingerprint(sample) = SHA-256(normalize(sample))
```

Normalization by content type:
- **bytes**: Used directly
- **text**: UTF-8 encoded
- **dict**: Recursively sorted keys, canonical JSON
- **tensor**: Raw bytes via tobytes() method

### Merkle Tree Batch Fingerprinting

Batches are fingerprinted using a binary Merkle tree:

```
                    ┌─────────────┐
                    │ Batch Root  │
                    │   (H1234)   │
                    └──────┬──────┘
               ┌───────────┴───────────┐
               ▼                       ▼
        ┌──────────┐            ┌──────────┐
        │   H12    │            │   H34    │
        └────┬─────┘            └────┬─────┘
         ┌───┴───┐               ┌───┴───┐
         ▼       ▼               ▼       ▼
       ┌───┐   ┌───┐           ┌───┐   ┌───┐
       │ S1│   │ S2│           │ S3│   │ S4│
       └───┘   └───┘           └───┘   └───┘
```

Algorithm:
1. Compute fingerprint for each sample: `[fp1, fp2, fp3, fp4]`
2. If odd number, duplicate last: `[fp1, fp2, fp3, fp3]`
3. Pairwise hash: `H12 = SHA-256(fp1 + fp2)`, `H34 = SHA-256(fp3 + fp4)`
4. Continue until single root: `H1234 = SHA-256(H12 + H34)`

### Reproducibility Guarantees

- Identical content always produces identical fingerprints
- Fingerprints are computed client-side with no external dependencies
- Merkle roots allow efficient batch verification
- Content-addressed storage enables automatic deduplication

## 6. License Propagation Model

### License Tracking

Each data source is associated with a license identifier (SPDX format preferred). As samples flow through batches, license information propagates.

### Propagation Rules

When multiple licenses are combined in a session:

| Property | Rule | Rationale |
|----------|------|-----------|
| Permissions | Intersection | Only permissions granted by all |
| Restrictions | Union | All restrictions apply |
| Conditions | Union | All conditions must be satisfied |

### Conflict Detection

Conflicts are detected in these scenarios:

1. **Copyleft + Proprietary**: Copyleft licenses (GPL) conflict with proprietary restrictions
2. **Incompatible Copyleft**: Different copyleft licenses (GPL-3.0 vs GPL-2.0) may be incompatible
3. **Commercial Restrictions**: Non-commercial licenses (CC-BY-NC) conflict with commercial use

Example conflict scenarios:

```
GPL-3.0-only + GPL-2.0-only = CONFLICT (incompatible copyleft)
GPL-3.0-only + proprietary  = CONFLICT (copyleft-proprietary)
MIT + Apache-2.0            = COMPATIBLE (permissive licenses)
MIT + GPL-3.0-only          = COMPATIBLE (permissive can be incorporated)
```

## 7. Design Decisions

### Why SQLite?

- **Portability**: Single file, no server required
- **Zero Configuration**: Works out of the box
- **Reliability**: ACID transactions, WAL mode
- **Performance**: Fast for read-heavy audit workloads
- **Ubiquity**: Available on all platforms

### Why Merkle Trees?

- **Efficient Verification**: Verify batch integrity without all samples
- **Deduplication**: Identical batches produce identical roots
- **Incremental Updates**: New samples don't require rehashing everything
- **Audit Trail**: Tamper-evident structure

### Why Content-Addressable Storage?

- **Deduplication**: Same sample stored once regardless of batches
- **Integrity**: Fingerprint verifies content hasn't changed
- **Efficient Queries**: Lookup by fingerprint is O(1)
- **Cross-Session Sharing**: Same sample in different sessions linked automatically

### Why Read-Safe by Default?

- **Audit Integrity**: Observers cannot modify what they observe
- **Pipeline Safety**: No risk of data corruption
- **Explicit Writes**: Clear separation between observation and persistence
- **Compliance**: Immutable records for regulatory requirements

## 8. Limitations

Origin explicitly does not support:

| Limitation | Reason |
|------------|--------|
| Actual data storage | Only stores metadata and fingerprints for privacy and scale |
| Legal determinations | Flags conflicts for human review; does not provide legal advice |
| Distributed storage | Local-first design; no built-in synchronization |
| Real-time streaming | Batch-oriented; not designed for streaming pipelines |
| Non-Python pipelines | Python-only implementation |
| Model provenance | Tracks data only; model versioning is out of scope |
| Web interface | CLI and API only; no built-in dashboard |

These limitations are intentional design choices that keep Origin focused, portable, and reliable.
