"""
Database schema definitions for Origin provenance storage.

This module defines the SQLite schema used to store provenance data.
The schema is normalized and optimized for audit queries.

Tables:
    - sessions: Training run metadata
    - sources: Data source information
    - licenses: License metadata
    - samples: Individual data sample records
    - batches: Batch-level aggregations
    - batch_samples: Junction table linking batches to samples
    - license_conflicts: Detected license conflicts
    - schema_version: Schema versioning for migrations
"""

import hashlib

# Schema version for migration tracking
SCHEMA_VERSION = 1

# Table creation SQL statements
CREATE_TABLES_SQL = """
-- Training session metadata
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    metadata TEXT,
    status TEXT NOT NULL
);

-- Data source information
CREATE TABLE IF NOT EXISTS sources (
    source_id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,
    source_path TEXT NOT NULL,
    license_id TEXT,
    first_seen TEXT NOT NULL,
    metadata TEXT
);

-- License metadata
CREATE TABLE IF NOT EXISTS licenses (
    license_id TEXT PRIMARY KEY,
    license_name TEXT NOT NULL,
    license_url TEXT,
    permissions TEXT,
    restrictions TEXT,
    conditions TEXT,
    copyleft INTEGER NOT NULL
);

-- Individual data sample records
CREATE TABLE IF NOT EXISTS samples (
    sample_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    content_type TEXT NOT NULL,
    byte_size INTEGER NOT NULL,
    first_seen TEXT NOT NULL,
    FOREIGN KEY (source_id) REFERENCES sources (source_id)
);

-- Batch-level aggregations
CREATE TABLE IF NOT EXISTS batches (
    batch_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    batch_index INTEGER NOT NULL,
    sample_count INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
);

-- Junction table linking batches to samples
CREATE TABLE IF NOT EXISTS batch_samples (
    batch_id TEXT NOT NULL,
    sample_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    PRIMARY KEY (batch_id, sample_id, position),
    FOREIGN KEY (batch_id) REFERENCES batches (batch_id),
    FOREIGN KEY (sample_id) REFERENCES samples (sample_id)
);

-- Detected license conflicts
CREATE TABLE IF NOT EXISTS license_conflicts (
    conflict_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    license_a TEXT NOT NULL,
    license_b TEXT NOT NULL,
    conflict_type TEXT NOT NULL,
    detected_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
);

-- Schema versioning for migrations
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);
"""

# Index creation SQL for common query patterns
INDEXES_SQL = """
-- Index for querying batches by session
CREATE INDEX IF NOT EXISTS idx_batches_session_id ON batches (session_id);

-- Index for querying samples by source
CREATE INDEX IF NOT EXISTS idx_samples_source_id ON samples (source_id);

-- Index for querying batch_samples by batch
CREATE INDEX IF NOT EXISTS idx_batch_samples_batch_id ON batch_samples (batch_id);

-- Index for querying batch_samples by sample
CREATE INDEX IF NOT EXISTS idx_batch_samples_sample_id ON batch_samples (sample_id);

-- Index for querying conflicts by session
CREATE INDEX IF NOT EXISTS idx_conflicts_session_id ON license_conflicts (session_id);

-- Index for querying sources by license
CREATE INDEX IF NOT EXISTS idx_sources_license_id ON sources (license_id);

-- Index for querying sessions by status
CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions (status);

-- Index for querying sessions by creation time
CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON sessions (created_at);
"""


def get_schema_hash() -> str:
    """
    Compute a SHA-256 hash of the schema definition.

    This hash can be used to detect schema changes and trigger migrations.
    The hash is computed from the CREATE_TABLES_SQL string.

    Returns:
        A 64-character lowercase hexadecimal string representing the
        SHA-256 hash of the schema definition.
    """
    return hashlib.sha256(CREATE_TABLES_SQL.encode("utf-8")).hexdigest()
