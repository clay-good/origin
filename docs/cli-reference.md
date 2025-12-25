# Origin CLI Reference

Complete command-line interface reference for Origin.

## Overview

Origin provides a command-line interface for managing provenance databases, querying data lineage, and exporting provenance information.

### Basic Usage

```
origin [OPTIONS] COMMAND [ARGS]
```

### Global Options

| Option | Description |
|--------|-------------|
| `--path PATH` | Path to provenance database (default: `./origin.db`) |
| `--quiet`, `-q` | Suppress non-essential output |
| `--json` | Output in JSON format |
| `--version` | Show version and exit |
| `--help` | Show help message and exit |

## Commands

### init

Initialize a new provenance database.

**Syntax:**
```
origin init [OPTIONS]
```

**Description:**
Creates a new SQLite database with the Origin schema. If the database already exists, reports success without modification.

**Example:**
```bash
# Initialize database in current directory
origin init

# Initialize at specific path
origin --path /data/project.db init
```

**Output:**
```
Initialized provenance database at ./origin.db
```

---

### status

Show database status and statistics.

**Syntax:**
```
origin status [OPTIONS]
```

**Description:**
Displays summary statistics about the provenance database including session count, sample count, and conflict count.

**Example:**
```bash
origin status
```

**Output:**
```
Database: ./origin.db
Schema Version: 1

Statistics:
  Sessions: 5
  Batches: 127
  Samples: 12543
  Sources: 3
  Licenses: 2
  Conflicts: 0
```

**JSON Output:**
```bash
origin --json status
```
```json
{
  "batch_count": 127,
  "conflict_count": 0,
  "license_count": 2,
  "sample_count": 12543,
  "schema_version": 1,
  "session_count": 5,
  "source_count": 3
}
```

---

### sessions

List all training sessions.

**Syntax:**
```
origin sessions [OPTIONS]
```

**Options:**
| Option | Description |
|--------|-------------|
| `--limit`, `-n` | Maximum sessions to show (default: 20) |

**Example:**
```bash
origin sessions
```

**Output:**
```
Session ID        Status     Created              Batches
----------------  ---------  -------------------  -------
a1b2c3d4e5f6...   completed  2025-01-15T10:30:00  45
f6e5d4c3b2a1...   completed  2025-01-14T14:22:00  32
1234567890ab...   active     2025-01-16T08:00:00  12
```

---

### inspect

Show details for a specific session.

**Syntax:**
```
origin inspect SESSION_ID [OPTIONS]
```

**Arguments:**
| Argument | Description |
|----------|-------------|
| `SESSION_ID` | The session ID to inspect |

**Options:**
| Option | Description |
|--------|-------------|
| `--batches`, `-b` | Show batch details |
| `--samples`, `-s` | Show sample statistics |

**Example:**
```bash
origin inspect a1b2c3d4e5f67890
```

**Output:**
```
Session: a1b2c3d4e5f67890abcdef1234567890abcdef1234567890abcdef12345678
Status: completed
Created: 2025-01-15T10:30:00+00:00
Config Hash: 1234567890abcdef...

Batches: 45
Total Samples: 4500

License Breakdown:
  MIT: 3200
  Apache-2.0: 1300
```

---

### query

Check if a license is present in sessions.

**Syntax:**
```
origin query LICENSE_ID [OPTIONS]
```

**Arguments:**
| Argument | Description |
|----------|-------------|
| `LICENSE_ID` | SPDX license identifier to search for |

**Options:**
| Option | Description |
|--------|-------------|
| `--session SESSION` | Limit search to specific session |

**Example:**
```bash
# Search all sessions
origin query MIT

# Search specific session
origin query GPL-3.0-only --session a1b2c3d4
```

**Output:**
```
License 'MIT' found in 3200 samples
```

**Exit Codes:**
- `0`: License found
- `2`: License not found

---

### card

Generate a provenance card for a session.

**Syntax:**
```
origin card SESSION_ID [OPTIONS]
```

**Arguments:**
| Argument | Description |
|----------|-------------|
| `SESSION_ID` | The session ID to generate card for |

**Options:**
| Option | Description |
|--------|-------------|
| `--output`, `-o` | Write to file instead of stdout |

**Example:**
```bash
# Print to stdout
origin card a1b2c3d4e5f67890

# Write to file
origin card a1b2c3d4e5f67890 --output provenance_card.md
```

**Output:**
A Markdown-formatted provenance card containing:
- Session information
- Data source summary
- License analysis
- Sample statistics
- Batch summary
- Audit trail

---

### export

Export session provenance data.

**Syntax:**
```
origin export SESSION_ID [OPTIONS]
```

**Arguments:**
| Argument | Description |
|----------|-------------|
| `SESSION_ID` | The session ID to export |

**Options:**
| Option | Description |
|--------|-------------|
| `--format`, `-f` | Export format: json, jsonl, mlflow, wandb, hf (default: json) |
| `--output`, `-o` | Output file path (required for jsonl) |

**Formats:**

| Format | Description |
|--------|-------------|
| `json` | Complete session data as JSON object |
| `jsonl` | Streaming format, one record per line |
| `mlflow` | MLflow-compatible tags and params |
| `wandb` | Weights & Biases metadata format |
| `hf` | HuggingFace Hub model card format |

**Examples:**
```bash
# Export as JSON (stdout)
origin export a1b2c3d4 --format json

# Export as JSONL (requires output file)
origin export a1b2c3d4 --format jsonl --output session.jsonl

# Export for MLflow
origin export a1b2c3d4 --format mlflow --output mlflow_data.json
```

---

### conflicts

List license conflicts.

**Syntax:**
```
origin conflicts [OPTIONS]
```

**Options:**
| Option | Description |
|--------|-------------|
| `--session SESSION` | Filter by session ID |

**Example:**
```bash
origin conflicts
```

**Output:**
```
Found 2 license conflict(s):

  Session: a1b2c3d4e5f6...
  Licenses: GPL-3.0-only vs GPL-2.0-only
  Type: incompatible_copyleft
  Detected: 2025-01-15T10:35:00

  Session: f6e5d4c3b2a1...
  Licenses: GPL-3.0-only vs proprietary
  Type: copyleft-proprietary
  Detected: 2025-01-14T14:30:00
```

---

### trace

Show full provenance trace for a sample.

**Syntax:**
```
origin trace SAMPLE_ID
```

**Arguments:**
| Argument | Description |
|----------|-------------|
| `SAMPLE_ID` | The sample fingerprint to trace |

**Example:**
```bash
origin trace abc123def456...
```

**Output:**
```
Sample: abc123def456789012345678901234567890123456789012345678901234
Content Type: text
Size: 1024 bytes
First Seen: 2025-01-15T10:30:00

Source: /data/training/dataset.csv
Source ID: 1234567890abcdef...
License: MIT

Appears in 3 batch(es):
  - Batch 12 in session a1b2c3d4e5f6...
  - Batch 45 in session a1b2c3d4e5f6...
  - Batch 7 in session f6e5d4c3b2a1...

Sessions: a1b2c3d4e5f6..., f6e5d4c3b2a1...
```

---

### version

Show version information.

**Syntax:**
```
origin version
```

**Example:**
```bash
origin version
```

**Output:**
```
origin 0.1.0
```

**JSON Output:**
```bash
origin --json version
```
```json
{
  "name": "origin",
  "version": "0.1.0"
}
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error (database not found, invalid input, etc.) |
| 2 | Query not found (for query command) |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ORIGIN_DB_PATH` | Default database path (overridden by `--path`) |

## Common Workflows

### Initial Setup

```bash
# Create a new project database
origin --path ./my_project.db init

# Verify creation
origin --path ./my_project.db status
```

### After Training

```bash
# List all sessions
origin sessions

# Inspect the latest session
origin inspect <session_id>

# Check for license conflicts
origin conflicts --session <session_id>

# Generate provenance card
origin card <session_id> --output MODEL_CARD.md
```

### Audit Query

```bash
# Check if GPL-licensed data was used
origin query GPL-3.0-only

# Trace a specific sample back to its source
origin trace <sample_fingerprint>

# Export for compliance review
origin export <session_id> --format json --output audit.json
```

### Integration with MLflow

```bash
# Export provenance for MLflow run
origin export <session_id> --format mlflow --output provenance.json

# In your MLflow code:
# mlflow.log_dict(json.load(open("provenance.json")), "provenance.json")
```
