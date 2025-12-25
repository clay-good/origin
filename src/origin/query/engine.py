"""
Query engine for Origin provenance auditing.

This module provides the QueryEngine class for executing audit queries
against the provenance database. All queries are read-only and use
parameterized SQL to prevent injection attacks.
"""

import fnmatch
import json
from typing import Any, Dict, List, Optional, Set

from origin.core.record import (
    ProvenanceRecord,
    SessionRecord,
    BatchRecord,
    SourceRecord,
    LicenseRecord,
)
from origin.storage.database import ProvenanceDatabase


class QueryEngine:
    """
    Audit query interface for provenance data.

    The QueryEngine provides methods for querying provenance records,
    tracing data lineage, and analyzing license usage across training
    sessions. All queries are read-only.

    Attributes:
        db: The ProvenanceDatabase instance.

    Example:
        >>> engine = QueryEngine(db)
        >>> samples = engine.find_samples_by_license("MIT")
        >>> breakdown = engine.get_license_breakdown(session_id)
    """

    def __init__(self, db: ProvenanceDatabase) -> None:
        """
        Initialize the query engine.

        Args:
            db: ProvenanceDatabase instance for executing queries.
        """
        self._db = db

    @property
    def db(self) -> ProvenanceDatabase:
        """Return the database instance."""
        return self._db

    def find_samples_by_license(
        self,
        license_id: str,
        session_id: Optional[str] = None,
    ) -> List[ProvenanceRecord]:
        """
        Find all samples associated with a specific license.

        Args:
            license_id: The license ID (SPDX identifier) to search for.
            session_id: Optional session ID to limit the search.

        Returns:
            List of ProvenanceRecord objects matching the criteria.
        """
        if session_id:
            # Query samples in specific session via batches
            query = """
                SELECT DISTINCT s.*
                FROM samples s
                JOIN sources src ON s.source_id = src.source_id
                JOIN batch_samples bs ON s.sample_id = bs.sample_id
                JOIN batches b ON bs.batch_id = b.batch_id
                WHERE src.license_id = ? AND b.session_id = ?
            """
            cursor = self._db._conn.execute(query, (license_id, session_id))
        else:
            # Query all samples with this license
            query = """
                SELECT DISTINCT s.*
                FROM samples s
                JOIN sources src ON s.source_id = src.source_id
                WHERE src.license_id = ?
            """
            cursor = self._db._conn.execute(query, (license_id,))

        results = []
        for row in cursor.fetchall():
            results.append(ProvenanceRecord(
                sample_id=row["sample_id"],
                source_id=row["source_id"],
                content_type=row["content_type"],
                byte_size=row["byte_size"],
                timestamp=row["first_seen"],
                metadata={},
            ))

        return results

    def find_samples_by_source(
        self,
        source_pattern: str,
        session_id: Optional[str] = None,
    ) -> List[ProvenanceRecord]:
        """
        Find samples from sources matching a glob pattern.

        Args:
            source_pattern: Glob pattern to match against source paths.
                Examples: "*.csv", "/data/train/*", "**/*.json"
            session_id: Optional session ID to limit the search.

        Returns:
            List of ProvenanceRecord objects from matching sources.
        """
        # Get all sources and filter by pattern
        if session_id:
            query = """
                SELECT DISTINCT src.source_id, src.source_path
                FROM sources src
                JOIN samples s ON src.source_id = s.source_id
                JOIN batch_samples bs ON s.sample_id = bs.sample_id
                JOIN batches b ON bs.batch_id = b.batch_id
                WHERE b.session_id = ?
            """
            cursor = self._db._conn.execute(query, (session_id,))
        else:
            query = "SELECT source_id, source_path FROM sources"
            cursor = self._db._conn.execute(query)

        matching_source_ids = []
        for row in cursor.fetchall():
            if fnmatch.fnmatch(row["source_path"], source_pattern):
                matching_source_ids.append(row["source_id"])

        if not matching_source_ids:
            return []

        # Get samples from matching sources
        results = []
        for source_id in matching_source_ids:
            if session_id:
                query = """
                    SELECT DISTINCT s.*
                    FROM samples s
                    JOIN batch_samples bs ON s.sample_id = bs.sample_id
                    JOIN batches b ON bs.batch_id = b.batch_id
                    WHERE s.source_id = ? AND b.session_id = ?
                """
                cursor = self._db._conn.execute(query, (source_id, session_id))
            else:
                query = "SELECT * FROM samples WHERE source_id = ?"
                cursor = self._db._conn.execute(query, (source_id,))

            for row in cursor.fetchall():
                results.append(ProvenanceRecord(
                    sample_id=row["sample_id"],
                    source_id=row["source_id"],
                    content_type=row["content_type"],
                    byte_size=row["byte_size"],
                    timestamp=row["first_seen"],
                    metadata={},
                ))

        return results

    def find_batches_with_license(
        self,
        session_id: str,
        license_id: str,
    ) -> List[BatchRecord]:
        """
        Find all batches containing samples with a specific license.

        Args:
            session_id: The session ID to search within.
            license_id: The license ID to search for.

        Returns:
            List of BatchRecord objects containing matching samples.
        """
        query = """
            SELECT DISTINCT b.*
            FROM batches b
            JOIN batch_samples bs ON b.batch_id = bs.batch_id
            JOIN samples s ON bs.sample_id = s.sample_id
            JOIN sources src ON s.source_id = src.source_id
            WHERE b.session_id = ? AND src.license_id = ?
            ORDER BY b.batch_index
        """
        cursor = self._db._conn.execute(query, (session_id, license_id))

        results = []
        for row in cursor.fetchall():
            # Get sample IDs for this batch
            sample_query = """
                SELECT sample_id FROM batch_samples
                WHERE batch_id = ? ORDER BY position
            """
            sample_cursor = self._db._conn.execute(sample_query, (row["batch_id"],))
            sample_ids = tuple(r["sample_id"] for r in sample_cursor.fetchall())

            results.append(BatchRecord(
                batch_id=row["batch_id"],
                session_id=row["session_id"],
                batch_index=row["batch_index"],
                sample_count=row["sample_count"],
                created_at=row["created_at"],
                sample_ids=sample_ids,
            ))

        return results

    def check_license_presence(
        self,
        session_id: str,
        license_id: str,
    ) -> bool:
        """
        Check if any sample with a specific license exists in a session.

        This is an efficient existence check using SQL EXISTS.

        Args:
            session_id: The session ID to check.
            license_id: The license ID to look for.

        Returns:
            True if any sample with the license exists, False otherwise.
        """
        query = """
            SELECT EXISTS(
                SELECT 1
                FROM batches b
                JOIN batch_samples bs ON b.batch_id = bs.batch_id
                JOIN samples s ON bs.sample_id = s.sample_id
                JOIN sources src ON s.source_id = src.source_id
                WHERE b.session_id = ? AND src.license_id = ?
            )
        """
        cursor = self._db._conn.execute(query, (session_id, license_id))
        result = cursor.fetchone()[0]
        return bool(result)

    def get_license_breakdown(self, session_id: str) -> Dict[str, int]:
        """
        Get count of samples per license in a session.

        Args:
            session_id: The session ID to analyze.

        Returns:
            Dictionary mapping license IDs to sample counts,
            sorted by count descending. Samples without a license
            are counted under 'unknown'.
        """
        query = """
            SELECT COALESCE(src.license_id, 'unknown') as license_id,
                   COUNT(DISTINCT s.sample_id) as count
            FROM batches b
            JOIN batch_samples bs ON b.batch_id = bs.batch_id
            JOIN samples s ON bs.sample_id = s.sample_id
            JOIN sources src ON s.source_id = src.source_id
            WHERE b.session_id = ?
            GROUP BY license_id
            ORDER BY count DESC
        """
        cursor = self._db._conn.execute(query, (session_id,))

        breakdown = {}
        for row in cursor.fetchall():
            breakdown[row["license_id"]] = row["count"]

        return breakdown

    def get_source_breakdown(self, session_id: str) -> Dict[str, int]:
        """
        Get count of samples per source in a session.

        Args:
            session_id: The session ID to analyze.

        Returns:
            Dictionary mapping source IDs to sample counts.
        """
        query = """
            SELECT s.source_id, COUNT(DISTINCT s.sample_id) as count
            FROM batches b
            JOIN batch_samples bs ON b.batch_id = bs.batch_id
            JOIN samples s ON bs.sample_id = s.sample_id
            WHERE b.session_id = ?
            GROUP BY s.source_id
            ORDER BY count DESC
        """
        cursor = self._db._conn.execute(query, (session_id,))

        breakdown = {}
        for row in cursor.fetchall():
            breakdown[row["source_id"]] = row["count"]

        return breakdown

    def find_conflicts(
        self,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find all license conflicts, optionally filtered by session.

        Args:
            session_id: Optional session ID to filter by.

        Returns:
            List of conflict dictionaries with keys: conflict_id,
            session_id, license_a, license_b, conflict_type, detected_at.
        """
        if session_id:
            return self._db.list_conflicts(session_id)
        else:
            # Query all conflicts
            query = "SELECT * FROM license_conflicts ORDER BY detected_at"
            cursor = self._db._conn.execute(query)

            conflicts = []
            for row in cursor.fetchall():
                conflicts.append({
                    "conflict_id": row["conflict_id"],
                    "session_id": row["session_id"],
                    "license_a": row["license_a"],
                    "license_b": row["license_b"],
                    "conflict_type": row["conflict_type"],
                    "detected_at": row["detected_at"],
                })

            return conflicts

    def trace_sample(self, sample_id: str) -> Dict[str, Any]:
        """
        Get full provenance trace for a sample.

        This traces a sample back to its source and license, and finds
        all batches and sessions where it appears.

        Args:
            sample_id: The sample ID (fingerprint) to trace.

        Returns:
            Dictionary with:
            - sample: ProvenanceRecord as dict, or None
            - source: SourceRecord as dict, or None
            - license: LicenseRecord as dict, or None
            - batches: List of BatchRecord dicts
            - sessions: List of session IDs where sample appears
        """
        result: Dict[str, Any] = {
            "sample": None,
            "source": None,
            "license": None,
            "batches": [],
            "sessions": [],
        }

        # Get sample
        sample = self._db.get_sample(sample_id)
        if sample is None:
            return result

        result["sample"] = sample.to_dict()

        # Get source
        source = self._db.get_source(sample.source_id)
        if source is not None:
            result["source"] = source.to_dict()

            # Get license
            if source.license_id:
                license_rec = self._db.get_license(source.license_id)
                if license_rec is not None:
                    result["license"] = license_rec.to_dict()

        # Find all batches containing this sample
        query = """
            SELECT DISTINCT b.*
            FROM batches b
            JOIN batch_samples bs ON b.batch_id = bs.batch_id
            WHERE bs.sample_id = ?
            ORDER BY b.created_at
        """
        cursor = self._db._conn.execute(query, (sample_id,))

        sessions: Set[str] = set()
        for row in cursor.fetchall():
            # Get sample IDs for this batch
            sample_query = """
                SELECT sample_id FROM batch_samples
                WHERE batch_id = ? ORDER BY position
            """
            sample_cursor = self._db._conn.execute(sample_query, (row["batch_id"],))
            sample_ids = tuple(r["sample_id"] for r in sample_cursor.fetchall())

            batch = BatchRecord(
                batch_id=row["batch_id"],
                session_id=row["session_id"],
                batch_index=row["batch_index"],
                sample_count=row["sample_count"],
                created_at=row["created_at"],
                sample_ids=sample_ids,
            )
            result["batches"].append(batch.to_dict())
            sessions.add(row["session_id"])

        result["sessions"] = sorted(sessions)

        return result

    def compare_sessions(
        self,
        session_a: str,
        session_b: str,
    ) -> Dict[str, Any]:
        """
        Compare two sessions for sample overlap and license differences.

        Args:
            session_a: First session ID.
            session_b: Second session ID.

        Returns:
            Dictionary with:
            - shared_samples: Count of samples in both sessions
            - unique_to_a: Count of samples only in session A
            - unique_to_b: Count of samples only in session B
            - license_diff: Dict of license count differences
        """
        # Get sample IDs for each session
        query_a = """
            SELECT DISTINCT s.sample_id
            FROM batches b
            JOIN batch_samples bs ON b.batch_id = bs.batch_id
            JOIN samples s ON bs.sample_id = s.sample_id
            WHERE b.session_id = ?
        """
        cursor_a = self._db._conn.execute(query_a, (session_a,))
        samples_a = set(row["sample_id"] for row in cursor_a.fetchall())

        cursor_b = self._db._conn.execute(query_a, (session_b,))
        samples_b = set(row["sample_id"] for row in cursor_b.fetchall())

        shared = samples_a & samples_b
        unique_a = samples_a - samples_b
        unique_b = samples_b - samples_a

        # Get license breakdowns
        breakdown_a = self.get_license_breakdown(session_a)
        breakdown_b = self.get_license_breakdown(session_b)

        # Compute differences
        all_licenses = set(breakdown_a.keys()) | set(breakdown_b.keys())
        license_diff = {}
        for lic in all_licenses:
            count_a = breakdown_a.get(lic, 0)
            count_b = breakdown_b.get(lic, 0)
            if count_a != count_b:
                license_diff[lic] = {
                    "session_a": count_a,
                    "session_b": count_b,
                    "difference": count_a - count_b,
                }

        return {
            "shared_samples": len(shared),
            "unique_to_a": len(unique_a),
            "unique_to_b": len(unique_b),
            "license_diff": license_diff,
        }

    def export_result(
        self,
        result: Any,
        format: str = "json",
    ) -> str:
        """
        Serialize query results to JSON or JSONL format.

        Args:
            result: Query result to serialize. Can be a list of dataclass
                records, a dictionary, or other JSON-serializable data.
            format: Output format - 'json' for pretty-printed JSON,
                'jsonl' for one JSON object per line.

        Returns:
            Serialized string in the specified format.

        Raises:
            ValueError: If format is not 'json' or 'jsonl'.
        """
        if format not in ("json", "jsonl"):
            raise ValueError(f"Unknown format: {format}. Use 'json' or 'jsonl'.")

        # Convert dataclass records to dicts
        def convert(obj: Any) -> Any:
            if hasattr(obj, "to_dict"):
                return obj.to_dict()
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            else:
                return obj

        converted = convert(result)

        if format == "json":
            return json.dumps(converted, indent=2, sort_keys=True)
        else:  # jsonl
            if isinstance(converted, list):
                lines = [json.dumps(item, sort_keys=True) for item in converted]
                return "\n".join(lines)
            else:
                return json.dumps(converted, sort_keys=True)
