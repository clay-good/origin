"""
Export system for Origin provenance data.

This module provides the ProvenanceExporter class for serializing provenance
data to portable formats compatible with external tools and platforms.

Supported formats:
    - JSON: Complete session export as single document
    - JSONL: Streaming line-by-line export for large datasets
    - MLflow: Run parameters, tags, and artifacts
    - W&B: Weights & Biases artifact metadata
    - HuggingFace: Model card metadata section
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from origin import __version__
from origin.core.fingerprint import merkle_root
from origin.core.license import LicenseAnalyzer
from origin.core.record import (
    SessionRecord,
    BatchRecord,
    SourceRecord,
    ProvenanceRecord,
    LicenseRecord,
)
from origin.storage.database import ProvenanceDatabase
from origin.cards.generator import ProvenanceCardGenerator


class ProvenanceExporter:
    """
    Export provenance data to various portable formats.

    The exporter serializes provenance data for integration with external
    systems like MLflow, Weights & Biases, and HuggingFace. All exports
    are read-only and deterministic.

    Attributes:
        db: The ProvenanceDatabase instance.

    Example:
        >>> exporter = ProvenanceExporter(db)
        >>> json_str = exporter.to_json(session_id)
        >>> mlflow_data = exporter.to_mlflow(session_id)
    """

    def __init__(self, db: ProvenanceDatabase) -> None:
        """
        Initialize the provenance exporter.

        Args:
            db: ProvenanceDatabase instance for reading provenance data.
        """
        self._db = db
        self._analyzer = LicenseAnalyzer()

    @property
    def db(self) -> ProvenanceDatabase:
        """Return the database instance."""
        return self._db

    def _get_session_data(self, session_id: str) -> Dict[str, Any]:
        """
        Collect all data for a session.

        Args:
            session_id: The session ID to export.

        Returns:
            Dictionary with session, batches, sources, samples, etc.

        Raises:
            ValueError: If session not found.
        """
        session = self._db.get_session(session_id)
        if session is None:
            raise ValueError(f"Session not found: {session_id}")

        batches = self._db.list_batches(session_id)
        conflicts = self._db.list_conflicts(session_id)

        # Collect unique sources, samples, and licenses
        source_ids: Set[str] = set()
        sample_ids: Set[str] = set()
        license_ids: Set[str] = set()
        samples: List[ProvenanceRecord] = []

        for batch in batches:
            batch_samples = self._db.list_samples(batch.batch_id)
            for sample in batch_samples:
                if sample.sample_id not in sample_ids:
                    samples.append(sample)
                    sample_ids.add(sample.sample_id)
                source_ids.add(sample.source_id)

        sources: List[SourceRecord] = []
        for source_id in source_ids:
            source = self._db.get_source(source_id)
            if source:
                sources.append(source)
                if source.license_id:
                    license_ids.add(source.license_id)

        licenses: List[LicenseRecord] = []
        for license_id in license_ids:
            lic = self._db.get_license(license_id)
            if lic:
                licenses.append(lic)

        return {
            "session": session,
            "batches": batches,
            "sources": sources,
            "samples": samples,
            "licenses": licenses,
            "conflicts": conflicts,
        }

    def _compute_session_fingerprint(self, session_id: str) -> str:
        """
        Compute overall fingerprint for a session.

        The session fingerprint is the Merkle root of all batch fingerprints,
        providing a single hash representing all data in the session.

        Args:
            session_id: The session ID.

        Returns:
            64-character hex string fingerprint.
        """
        batches = self._db.list_batches(session_id)

        if not batches:
            # No batches - return fingerprint of empty marker
            from origin.core.fingerprint import fingerprint_bytes
            return fingerprint_bytes(b"empty_session")

        batch_fingerprints = [batch.batch_id for batch in batches]
        return merkle_root(batch_fingerprints)

    @staticmethod
    def _now_iso() -> str:
        """Return current UTC time as ISO 8601 string."""
        return datetime.now(timezone.utc).isoformat()

    def to_json(self, session_id: str, pretty: bool = True) -> str:
        """
        Export complete session as JSON document.

        Args:
            session_id: The session ID to export.
            pretty: If True, format with indentation (default True).

        Returns:
            JSON string containing complete session data.

        Raises:
            ValueError: If session not found.
        """
        data = self._get_session_data(session_id)
        session = data["session"]
        batches = data["batches"]
        sources = data["sources"]
        samples = data["samples"]
        licenses = data["licenses"]
        conflicts = data["conflicts"]

        # Compute statistics
        unique_samples = len(set(s.sample_id for s in samples))
        licenses_used = sorted(set(
            src.license_id for src in sources if src.license_id
        ))

        export_data = {
            "origin_version": __version__,
            "export_timestamp": self._now_iso(),
            "session": session.to_dict(),
            "sources": [src.to_dict() for src in sources],
            "licenses": [lic.to_dict() for lic in licenses],
            "batches": [batch.to_dict() for batch in batches],
            "samples": [sample.to_dict() for sample in samples],
            "conflicts": conflicts,
            "statistics": {
                "total_samples": len(samples),
                "unique_samples": unique_samples,
                "total_batches": len(batches),
                "licenses_used": licenses_used,
            },
        }

        if pretty:
            return json.dumps(export_data, indent=2, sort_keys=True)
        else:
            return json.dumps(export_data, sort_keys=True)

    def to_jsonl(
        self,
        session_id: str,
        output_path: Union[str, Path],
    ) -> int:
        """
        Export session as JSONL (one JSON object per line).

        This format is suitable for large datasets as it streams data
        incrementally rather than building a large in-memory structure.

        Args:
            session_id: The session ID to export.
            output_path: Path to write the JSONL file.

        Returns:
            Number of lines written.

        Raises:
            ValueError: If session not found.
        """
        data = self._get_session_data(session_id)
        session = data["session"]
        batches = data["batches"]
        sources = data["sources"]
        samples = data["samples"]
        conflicts = data["conflicts"]

        output_path = Path(output_path)
        lines_written = 0

        with open(output_path, "w", encoding="utf-8") as f:
            # Header line
            header = {
                "type": "header",
                "origin_version": __version__,
                "export_timestamp": self._now_iso(),
                "session": session.to_dict(),
            }
            f.write(json.dumps(header, sort_keys=True) + "\n")
            lines_written += 1

            # Source lines
            for source in sources:
                line = {"type": "source", "data": source.to_dict()}
                f.write(json.dumps(line, sort_keys=True) + "\n")
                lines_written += 1

            # Batch lines (with their samples)
            for batch in batches:
                batch_samples = self._db.list_samples(batch.batch_id)
                line = {
                    "type": "batch",
                    "data": batch.to_dict(),
                    "samples": [s.to_dict() for s in batch_samples],
                }
                f.write(json.dumps(line, sort_keys=True) + "\n")
                lines_written += 1

            # Conflict lines
            for conflict in conflicts:
                line = {"type": "conflict", "data": conflict}
                f.write(json.dumps(line, sort_keys=True) + "\n")
                lines_written += 1

            # Footer line
            unique_samples = len(set(s.sample_id for s in samples))
            footer = {
                "type": "footer",
                "statistics": {
                    "total_samples": len(samples),
                    "unique_samples": unique_samples,
                    "total_batches": len(batches),
                    "total_sources": len(sources),
                    "total_conflicts": len(conflicts),
                },
            }
            f.write(json.dumps(footer, sort_keys=True) + "\n")
            lines_written += 1

        return lines_written

    def to_mlflow(self, session_id: str) -> Dict[str, Any]:
        """
        Export session in MLflow-compatible format.

        The returned dict contains tags, params, and artifacts suitable
        for logging to an MLflow run.

        Args:
            session_id: The session ID to export.

        Returns:
            Dictionary with 'tags', 'params', and 'artifacts' keys.

        Raises:
            ValueError: If session not found.
        """
        data = self._get_session_data(session_id)
        batches = data["batches"]
        sources = data["sources"]
        samples = data["samples"]
        conflicts = data["conflicts"]

        # Compute session fingerprint
        session_fingerprint = self._compute_session_fingerprint(session_id)

        # Determine primary license
        license_analysis = self._analyzer.analyze_session(self._db, session_id)
        primary_license = license_analysis.get("result_license", "unknown")

        # Generate provenance card
        card_generator = ProvenanceCardGenerator(self._db)
        provenance_card = card_generator.generate(session_id)

        return {
            "tags": {
                "origin.session_id": session_id,
                "origin.data_fingerprint": session_fingerprint,
                "origin.primary_license": primary_license,
                "origin.has_conflicts": "true" if conflicts else "false",
            },
            "params": {
                "origin.sample_count": str(len(samples)),
                "origin.batch_count": str(len(batches)),
                "origin.source_count": str(len(sources)),
            },
            "artifacts": {
                "provenance_card.md": provenance_card,
            },
        }

    def to_wandb(self, session_id: str) -> Dict[str, Any]:
        """
        Export session in Weights & Biases artifact format.

        The returned dict contains metadata and description suitable
        for attaching to a W&B artifact.

        Args:
            session_id: The session ID to export.

        Returns:
            Dictionary with 'metadata' and 'description' keys.

        Raises:
            ValueError: If session not found.
        """
        data = self._get_session_data(session_id)
        session = data["session"]
        batches = data["batches"]
        sources = data["sources"]
        samples = data["samples"]
        conflicts = data["conflicts"]

        # Compute session fingerprint
        session_fingerprint = self._compute_session_fingerprint(session_id)

        # Collect license IDs
        licenses = sorted(set(
            src.license_id for src in sources if src.license_id
        ))

        # Collect source paths
        source_paths = sorted(src.source_path for src in sources)

        # Generate description
        description_lines = [
            f"Provenance data for training session {session_id}",
            f"",
            f"Session Status: {session.status}",
            f"Total Samples: {len(samples)}",
            f"Total Batches: {len(batches)}",
            f"Data Sources: {len(sources)}",
            f"Licenses: {', '.join(licenses) if licenses else 'Unknown'}",
        ]

        if conflicts:
            description_lines.append(f"License Conflicts: {len(conflicts)}")

        return {
            "metadata": {
                "origin_session_id": session_id,
                "origin_version": __version__,
                "data_fingerprint": session_fingerprint,
                "sample_count": len(samples),
                "batch_count": len(batches),
                "licenses": licenses,
                "has_conflicts": len(conflicts) > 0,
                "sources": source_paths,
            },
            "description": "\n".join(description_lines),
        }

    def to_huggingface(self, session_id: str) -> Dict[str, Any]:
        """
        Export session in HuggingFace model card metadata format.

        The returned dict is suitable for inclusion in the metadata
        section of a HuggingFace model card YAML frontmatter.

        Args:
            session_id: The session ID to export.

        Returns:
            Dictionary with 'datasets', 'license', and 'training_data' keys.

        Raises:
            ValueError: If session not found.
        """
        data = self._get_session_data(session_id)
        sources = data["sources"]
        samples = data["samples"]
        conflicts = data["conflicts"]

        # Compute session fingerprint
        session_fingerprint = self._compute_session_fingerprint(session_id)

        # Determine license
        license_analysis = self._analyzer.analyze_session(self._db, session_id)
        primary_license = license_analysis.get("result_license", "unknown")

        # Dataset identifiers (source paths or names)
        datasets = sorted(set(src.source_path for src in sources))

        # Format conflict summaries
        conflict_summaries = []
        for conflict in conflicts:
            summary = f"{conflict['license_a']} vs {conflict['license_b']}: {conflict['conflict_type']}"
            conflict_summaries.append(summary)

        return {
            "datasets": datasets,
            "license": primary_license,
            "training_data": {
                "samples": len(samples),
                "sources": len(sources),
                "provenance_session": session_id,
                "provenance_fingerprint": session_fingerprint,
                "license_conflicts": conflict_summaries,
            },
        }
