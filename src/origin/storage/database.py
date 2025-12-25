"""
SQLite-based provenance database for Origin.

This module provides the ProvenanceDatabase class, which handles all
persistence operations for provenance tracking. The database uses SQLite
with WAL mode for crash safety and supports both read-write and read-only
access modes.

All queries use parameterized statements to prevent SQL injection.
Write operations are wrapped in transactions for data integrity.
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from origin.core.record import (
    ProvenanceRecord,
    SessionRecord,
    BatchRecord,
    SourceRecord,
    LicenseRecord,
)
from origin.storage.schema import (
    SCHEMA_VERSION,
    CREATE_TABLES_SQL,
    INDEXES_SQL,
)


def _safe_json_loads(data: Optional[str], default: Any = None) -> Any:
    """
    Safely parse JSON, returning default on failure.

    Args:
        data: JSON string to parse, or None.
        default: Default value to return if parsing fails or data is None.

    Returns:
        Parsed JSON data, or default if parsing fails.
    """
    if data is None:
        return default if default is not None else {}
    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError):
        return default if default is not None else {}


class ProvenanceDatabase:
    """
    SQLite-based storage for provenance data.

    This class provides all database operations for Origin provenance tracking,
    including session management, sample recording, and audit queries.

    The database uses:
        - WAL mode for crash safety and concurrent reads
        - Parameterized queries to prevent SQL injection
        - Transactions for write operations
        - INSERT OR IGNORE for idempotent inserts

    Attributes:
        path: Path to the database file.
        read_only: Whether the database is opened in read-only mode.

    Example:
        >>> with ProvenanceDatabase("./origin.db") as db:
        ...     session = db.begin_session("config_hash_123")
        ...     db.record_sample(sample_record)
        ...     db.end_session(session.session_id, "completed")
    """

    def __init__(
        self,
        path: Union[str, Path],
        read_only: bool = False,
    ) -> None:
        """
        Initialize the provenance database.

        Args:
            path: Path to the SQLite database file. Will be created if it
                does not exist (unless read_only is True).
            read_only: If True, open the database in read-only mode. The
                database file must already exist.

        Raises:
            FileNotFoundError: If read_only is True and the database file
                does not exist.
            sqlite3.Error: If there is an error connecting to the database.
        """
        self._path = Path(path)
        self._read_only = read_only
        self._conn: Optional[sqlite3.Connection] = None

        if read_only and not self._path.exists():
            raise FileNotFoundError(
                f"Database file not found: {self._path}. "
                "Cannot open non-existent database in read-only mode."
            )

        self._connect()
        self._initialize_schema()

    def _connect(self) -> None:
        """Establish connection to the database."""
        if self._read_only:
            # Open in read-only mode using URI
            uri = f"file:{self._path}?mode=ro"
            self._conn = sqlite3.connect(uri, uri=True)
        else:
            self._conn = sqlite3.connect(str(self._path))
            # Enable WAL mode for crash safety
            self._conn.execute("PRAGMA journal_mode=WAL")

        # Enable foreign keys
        self._conn.execute("PRAGMA foreign_keys=ON")
        # Return rows as Row objects for named access
        self._conn.row_factory = sqlite3.Row

    def _initialize_schema(self) -> None:
        """Initialize or verify the database schema."""
        if self._read_only:
            # Just verify schema version exists
            cursor = self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
            )
            if cursor.fetchone() is None:
                raise ValueError(
                    "Database schema not initialized. Cannot use read-only mode."
                )
            return

        # Create tables
        self._conn.executescript(CREATE_TABLES_SQL)

        # Create indexes
        self._conn.executescript(INDEXES_SQL)

        # Check/set schema version
        cursor = self._conn.execute("SELECT version FROM schema_version LIMIT 1")
        row = cursor.fetchone()

        if row is None:
            # First time initialization
            self._conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)",
                (SCHEMA_VERSION,),
            )
            self._conn.commit()
        elif row[0] != SCHEMA_VERSION:
            # Schema version mismatch - would need migration
            raise ValueError(
                f"Schema version mismatch. Database has version {row[0]}, "
                f"but code expects version {SCHEMA_VERSION}. "
                "Migration required."
            )

    @property
    def path(self) -> Path:
        """Return the database file path."""
        return self._path

    @property
    def read_only(self) -> bool:
        """Return whether the database is in read-only mode."""
        return self._read_only

    def __enter__(self) -> "ProvenanceDatabase":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager and close connection."""
        self.close()

    def __del__(self) -> None:
        """Destructor to ensure connection is closed."""
        self.close()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _check_write_allowed(self) -> None:
        """Raise an error if database is read-only."""
        if self._read_only:
            raise PermissionError(
                "Cannot write to database opened in read-only mode."
            )

    @staticmethod
    def _now_iso() -> str:
        """Return current UTC time as ISO 8601 string."""
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _generate_uuid() -> str:
        """Generate a new UUID string."""
        return str(uuid.uuid4())

    @staticmethod
    def _validate_string_id(value: str, name: str, max_length: int = 256) -> None:
        """
        Validate a string identifier.

        Args:
            value: The string to validate.
            name: Name of the parameter (for error messages).
            max_length: Maximum allowed length.

        Raises:
            ValueError: If validation fails.
        """
        if not isinstance(value, str):
            raise ValueError(f"{name} must be a string")
        if not value or not value.strip():
            raise ValueError(f"{name} cannot be empty")
        if len(value) > max_length:
            raise ValueError(f"{name} exceeds maximum length of {max_length}")

    # =========================================================================
    # Session Management
    # =========================================================================

    def begin_session(
        self,
        config_hash: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SessionRecord:
        """
        Begin a new provenance tracking session.

        Args:
            config_hash: SHA-256 hash of the training configuration.
            metadata: Optional metadata dictionary.

        Returns:
            A SessionRecord for the newly created session.

        Raises:
            PermissionError: If database is in read-only mode.
            ValueError: If config_hash is invalid.
        """
        self._check_write_allowed()
        self._validate_string_id(config_hash, "config_hash")

        session_id = self._generate_uuid()
        created_at = self._now_iso()
        metadata_json = json.dumps(metadata) if metadata else None
        status = "running"

        self._conn.execute(
            """
            INSERT INTO sessions (session_id, created_at, config_hash, metadata, status)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, created_at, config_hash, metadata_json, status),
        )
        self._conn.commit()

        return SessionRecord(
            session_id=session_id,
            created_at=created_at,
            config_hash=config_hash,
            status=status,
            metadata=metadata or {},
        )

    def end_session(self, session_id: str, status: str) -> None:
        """
        End a provenance tracking session.

        Args:
            session_id: The session ID to update.
            status: The final status ('completed' or 'failed').

        Raises:
            PermissionError: If database is in read-only mode.
            ValueError: If status is not valid or session_id is invalid.
        """
        self._check_write_allowed()
        self._validate_string_id(session_id, "session_id")

        if status not in ("completed", "failed"):
            raise ValueError("status must be 'completed' or 'failed'")

        self._conn.execute(
            "UPDATE sessions SET status = ? WHERE session_id = ?",
            (status, session_id),
        )
        self._conn.commit()

    # =========================================================================
    # Recording Methods
    # =========================================================================

    def record_source(self, source: SourceRecord) -> None:
        """
        Record a data source.

        Uses INSERT OR IGNORE to handle duplicate sources gracefully.

        Args:
            source: The SourceRecord to store.

        Raises:
            PermissionError: If database is in read-only mode.
        """
        self._check_write_allowed()

        metadata_json = json.dumps(source.metadata) if source.metadata else None

        self._conn.execute(
            """
            INSERT OR IGNORE INTO sources
            (source_id, source_type, source_path, license_id, first_seen, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                source.source_id,
                source.source_type,
                source.source_path,
                source.license_id,
                source.first_seen,
                metadata_json,
            ),
        )
        self._conn.commit()

    def record_sample(self, record: ProvenanceRecord) -> None:
        """
        Record a data sample observation.

        Uses INSERT OR IGNORE to handle duplicate samples gracefully.
        Note: The timestamp from the record is stored as first_seen.

        Args:
            record: The ProvenanceRecord to store.

        Raises:
            PermissionError: If database is in read-only mode.
        """
        self._check_write_allowed()

        self._conn.execute(
            """
            INSERT OR IGNORE INTO samples
            (sample_id, source_id, content_type, byte_size, first_seen)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                record.sample_id,
                record.source_id,
                record.content_type,
                record.byte_size,
                record.timestamp,
            ),
        )
        self._conn.commit()

    def record_batch(self, batch: BatchRecord) -> None:
        """
        Record a batch and its sample associations.

        Uses INSERT OR IGNORE to handle duplicate batches gracefully.
        Also inserts into batch_samples junction table.

        Args:
            batch: The BatchRecord to store.

        Raises:
            PermissionError: If database is in read-only mode.
        """
        self._check_write_allowed()

        # Insert batch record
        self._conn.execute(
            """
            INSERT OR IGNORE INTO batches
            (batch_id, session_id, batch_index, sample_count, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                batch.batch_id,
                batch.session_id,
                batch.batch_index,
                batch.sample_count,
                batch.created_at,
            ),
        )

        # Insert batch_samples associations
        for position, sample_id in enumerate(batch.sample_ids):
            self._conn.execute(
                """
                INSERT OR IGNORE INTO batch_samples
                (batch_id, sample_id, position)
                VALUES (?, ?, ?)
                """,
                (batch.batch_id, sample_id, position),
            )

        self._conn.commit()

    def record_license(self, license: LicenseRecord) -> None:
        """
        Record a license definition.

        Uses INSERT OR IGNORE to handle duplicate licenses gracefully.

        Args:
            license: The LicenseRecord to store.

        Raises:
            PermissionError: If database is in read-only mode.
        """
        self._check_write_allowed()

        self._conn.execute(
            """
            INSERT OR IGNORE INTO licenses
            (license_id, license_name, license_url, permissions, restrictions,
             conditions, copyleft)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                license.license_id,
                license.license_name,
                license.license_url,
                json.dumps(list(license.permissions)),
                json.dumps(list(license.restrictions)),
                json.dumps(list(license.conditions)),
                1 if license.copyleft else 0,
            ),
        )
        self._conn.commit()

    def record_conflict(
        self,
        session_id: str,
        license_a: str,
        license_b: str,
        conflict_type: str,
    ) -> None:
        """
        Record a license conflict.

        Args:
            session_id: The session where the conflict was detected.
            license_a: First license in the conflict.
            license_b: Second license in the conflict.
            conflict_type: Type of conflict (e.g., 'copyleft_mix').

        Raises:
            PermissionError: If database is in read-only mode.
        """
        self._check_write_allowed()

        conflict_id = self._generate_uuid()
        detected_at = self._now_iso()

        self._conn.execute(
            """
            INSERT INTO license_conflicts
            (conflict_id, session_id, license_a, license_b, conflict_type, detected_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (conflict_id, session_id, license_a, license_b, conflict_type, detected_at),
        )
        self._conn.commit()

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_session(self, session_id: str) -> Optional[SessionRecord]:
        """
        Retrieve a session by ID.

        Args:
            session_id: The session ID to look up.

        Returns:
            The SessionRecord if found, None otherwise.
        """
        cursor = self._conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?",
            (session_id,),
        )
        row = cursor.fetchone()

        if row is None:
            return None

        metadata = _safe_json_loads(row["metadata"], {})

        return SessionRecord(
            session_id=row["session_id"],
            created_at=row["created_at"],
            config_hash=row["config_hash"],
            status=row["status"],
            metadata=metadata,
        )

    def get_sample(self, sample_id: str) -> Optional[ProvenanceRecord]:
        """
        Retrieve a sample by ID.

        Args:
            sample_id: The sample ID (fingerprint) to look up.

        Returns:
            The ProvenanceRecord if found, None otherwise.
        """
        cursor = self._conn.execute(
            "SELECT * FROM samples WHERE sample_id = ?",
            (sample_id,),
        )
        row = cursor.fetchone()

        if row is None:
            return None

        return ProvenanceRecord(
            sample_id=row["sample_id"],
            source_id=row["source_id"],
            content_type=row["content_type"],
            byte_size=row["byte_size"],
            timestamp=row["first_seen"],
            metadata={},
        )

    def get_batch(self, batch_id: str) -> Optional[BatchRecord]:
        """
        Retrieve a batch by ID.

        Args:
            batch_id: The batch ID (Merkle root) to look up.

        Returns:
            The BatchRecord if found, None otherwise.
        """
        cursor = self._conn.execute(
            "SELECT * FROM batches WHERE batch_id = ?",
            (batch_id,),
        )
        row = cursor.fetchone()

        if row is None:
            return None

        # Get sample IDs from junction table
        sample_cursor = self._conn.execute(
            """
            SELECT sample_id FROM batch_samples
            WHERE batch_id = ?
            ORDER BY position
            """,
            (batch_id,),
        )
        sample_ids = tuple(r["sample_id"] for r in sample_cursor.fetchall())

        return BatchRecord(
            batch_id=row["batch_id"],
            session_id=row["session_id"],
            batch_index=row["batch_index"],
            sample_count=row["sample_count"],
            created_at=row["created_at"],
            sample_ids=sample_ids,
        )

    def get_source(self, source_id: str) -> Optional[SourceRecord]:
        """
        Retrieve a source by ID.

        Args:
            source_id: The source ID to look up.

        Returns:
            The SourceRecord if found, None otherwise.
        """
        cursor = self._conn.execute(
            "SELECT * FROM sources WHERE source_id = ?",
            (source_id,),
        )
        row = cursor.fetchone()

        if row is None:
            return None

        metadata = json.loads(row["metadata"]) if row["metadata"] else {}

        return SourceRecord(
            source_id=row["source_id"],
            source_type=row["source_type"],
            source_path=row["source_path"],
            license_id=row["license_id"],
            first_seen=row["first_seen"],
            metadata=metadata,
        )

    def get_license(self, license_id: str) -> Optional[LicenseRecord]:
        """
        Retrieve a license by ID.

        Args:
            license_id: The license ID (SPDX or custom) to look up.

        Returns:
            The LicenseRecord if found, None otherwise.
        """
        cursor = self._conn.execute(
            "SELECT * FROM licenses WHERE license_id = ?",
            (license_id,),
        )
        row = cursor.fetchone()

        if row is None:
            return None

        permissions = tuple(_safe_json_loads(row["permissions"], []))
        restrictions = tuple(_safe_json_loads(row["restrictions"], []))
        conditions = tuple(_safe_json_loads(row["conditions"], []))

        return LicenseRecord(
            license_id=row["license_id"],
            license_name=row["license_name"],
            license_url=row["license_url"],
            permissions=permissions,
            restrictions=restrictions,
            conditions=conditions,
            copyleft=bool(row["copyleft"]),
        )

    def list_sessions(self, limit: int = 100) -> List[SessionRecord]:
        """
        List sessions, ordered by creation time (newest first).

        Args:
            limit: Maximum number of sessions to return.

        Returns:
            List of SessionRecord objects.
        """
        cursor = self._conn.execute(
            "SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )

        sessions = []
        for row in cursor.fetchall():
            metadata = _safe_json_loads(row["metadata"], {})
            sessions.append(
                SessionRecord(
                    session_id=row["session_id"],
                    created_at=row["created_at"],
                    config_hash=row["config_hash"],
                    status=row["status"],
                    metadata=metadata,
                )
            )
        return sessions

    def list_batches(self, session_id: str) -> List[BatchRecord]:
        """
        List all batches for a session, ordered by batch index.

        Args:
            session_id: The session ID to query.

        Returns:
            List of BatchRecord objects.
        """
        cursor = self._conn.execute(
            "SELECT * FROM batches WHERE session_id = ? ORDER BY batch_index",
            (session_id,),
        )

        batches = []
        for row in cursor.fetchall():
            # Get sample IDs for this batch
            sample_cursor = self._conn.execute(
                """
                SELECT sample_id FROM batch_samples
                WHERE batch_id = ?
                ORDER BY position
                """,
                (row["batch_id"],),
            )
            sample_ids = tuple(r["sample_id"] for r in sample_cursor.fetchall())

            batches.append(
                BatchRecord(
                    batch_id=row["batch_id"],
                    session_id=row["session_id"],
                    batch_index=row["batch_index"],
                    sample_count=row["sample_count"],
                    created_at=row["created_at"],
                    sample_ids=sample_ids,
                )
            )
        return batches

    def list_samples(self, batch_id: str) -> List[ProvenanceRecord]:
        """
        List all samples in a batch, ordered by position.

        Args:
            batch_id: The batch ID to query.

        Returns:
            List of ProvenanceRecord objects.
        """
        cursor = self._conn.execute(
            """
            SELECT s.* FROM samples s
            JOIN batch_samples bs ON s.sample_id = bs.sample_id
            WHERE bs.batch_id = ?
            ORDER BY bs.position
            """,
            (batch_id,),
        )

        samples = []
        for row in cursor.fetchall():
            samples.append(
                ProvenanceRecord(
                    sample_id=row["sample_id"],
                    source_id=row["source_id"],
                    content_type=row["content_type"],
                    byte_size=row["byte_size"],
                    timestamp=row["first_seen"],
                    metadata={},
                )
            )
        return samples

    def list_conflicts(self, session_id: str) -> List[Dict[str, Any]]:
        """
        List all license conflicts for a session.

        Args:
            session_id: The session ID to query.

        Returns:
            List of conflict dictionaries with keys: conflict_id, license_a,
            license_b, conflict_type, detected_at.
        """
        cursor = self._conn.execute(
            """
            SELECT * FROM license_conflicts
            WHERE session_id = ?
            ORDER BY detected_at
            """,
            (session_id,),
        )

        conflicts = []
        for row in cursor.fetchall():
            conflicts.append(
                {
                    "conflict_id": row["conflict_id"],
                    "session_id": row["session_id"],
                    "license_a": row["license_a"],
                    "license_b": row["license_b"],
                    "conflict_type": row["conflict_type"],
                    "detected_at": row["detected_at"],
                }
            )
        return conflicts

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics summary.

        Returns:
            Dictionary with counts for sessions, batches, samples,
            sources, licenses, conflicts, and schema version.
        """
        stats = {}

        # Get schema version
        cursor = self._conn.execute("SELECT version FROM schema_version LIMIT 1")
        row = cursor.fetchone()
        stats["schema_version"] = row[0] if row else None

        # Count each table
        tables = [
            ("session_count", "sessions"),
            ("batch_count", "batches"),
            ("sample_count", "samples"),
            ("source_count", "sources"),
            ("license_count", "licenses"),
            ("conflict_count", "license_conflicts"),
        ]

        for stat_name, table_name in tables:
            cursor = self._conn.execute(f"SELECT COUNT(*) FROM {table_name}")
            stats[stat_name] = cursor.fetchone()[0]

        return stats
