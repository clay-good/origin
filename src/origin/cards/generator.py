"""
Provenance card generator for Origin.

This module provides the ProvenanceCardGenerator class for creating
human-readable provenance documentation in Markdown format. Generated
cards are suitable for compliance review and audit purposes.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from origin import __version__
from origin.core.license import LicenseAnalyzer
from origin.core.record import (
    SessionRecord,
    BatchRecord,
    SourceRecord,
    ProvenanceRecord,
)
from origin.storage.database import ProvenanceDatabase
from origin.cards.templates import (
    CARD_TEMPLATE,
    SECTION_TEMPLATES,
    format_timestamp,
    format_license_summary,
    format_table,
    format_bytes,
    truncate_hash,
)


class ProvenanceCardGenerator:
    """
    Generator for human-readable provenance documentation.

    This class creates Markdown-formatted provenance cards that document
    the data lineage, license terms, and audit trail for training sessions.
    Generated cards are suitable for compliance review.

    Attributes:
        db: The ProvenanceDatabase instance.
        analyzer: LicenseAnalyzer for license analysis.

    Example:
        >>> generator = ProvenanceCardGenerator(db)
        >>> card = generator.generate(session_id)
        >>> print(card)  # Markdown content
        >>> generator.generate_to_file(session_id, "provenance.md")
    """

    def __init__(self, db: ProvenanceDatabase) -> None:
        """
        Initialize the provenance card generator.

        Args:
            db: ProvenanceDatabase instance for querying provenance data.
        """
        self._db = db
        self._analyzer = LicenseAnalyzer()

    @property
    def db(self) -> ProvenanceDatabase:
        """Return the database instance."""
        return self._db

    @property
    def analyzer(self) -> LicenseAnalyzer:
        """Return the license analyzer instance."""
        return self._analyzer

    def generate(self, session_id: str) -> str:
        """
        Generate a provenance card for a session.

        Args:
            session_id: The session ID to generate a card for.

        Returns:
            Markdown-formatted provenance card string.

        Raises:
            ValueError: If the session is not found.
        """
        # Get session record
        session = self._db.get_session(session_id)
        if session is None:
            raise ValueError(f"Session not found: {session_id}")

        # Get all batches for session
        batches = self._db.list_batches(session_id)

        # Collect unique sources
        sources = self._collect_sources(batches)

        # Run license analysis
        license_analysis = self._analyzer.analyze_session(self._db, session_id)

        # Generate each section
        session_info = self._generate_session_info(session)
        sources_section = self._generate_sources_section(sources)
        license_section = self._generate_license_section(license_analysis)
        stats_section = self._generate_stats_section(session_id, batches)
        batch_section = self._generate_batch_section(batches)
        audit_section = self._generate_audit_section(session_id, session, batches)

        # Combine into final card
        generated_at = format_timestamp(datetime.now(timezone.utc).isoformat())

        card = CARD_TEMPLATE.format(
            session_info=session_info,
            sources_section=sources_section,
            license_section=license_section,
            stats_section=stats_section,
            batch_section=batch_section,
            audit_section=audit_section,
            version=__version__,
            generated_at=generated_at,
        )

        return card

    def generate_to_file(
        self,
        session_id: str,
        path: Union[str, Path],
    ) -> Path:
        """
        Generate a provenance card and write it to a file.

        Args:
            session_id: The session ID to generate a card for.
            path: Path to write the card file.

        Returns:
            Path to the created file.

        Raises:
            ValueError: If the session is not found.
        """
        path = Path(path)
        card = self.generate(session_id)

        path.write_text(card, encoding="utf-8")

        return path

    def generate_batch_card(self, batch_id: str) -> str:
        """
        Generate a provenance card for a single batch.

        Args:
            batch_id: The batch ID to generate a card for.

        Returns:
            Markdown-formatted batch provenance card string.

        Raises:
            ValueError: If the batch is not found.
        """
        batch = self._db.get_batch(batch_id)
        if batch is None:
            raise ValueError(f"Batch not found: {batch_id}")

        # Format sample fingerprints as a list
        if batch.sample_ids:
            fingerprints = "\n".join(
                f"- `{truncate_hash(sid, 16)}`" for sid in batch.sample_ids
            )
        else:
            fingerprints = "No samples recorded."

        generated_at = format_timestamp(datetime.now(timezone.utc).isoformat())

        card = SECTION_TEMPLATES["batch_card"].format(
            batch_id=batch.batch_id,
            session_id=batch.session_id,
            batch_index=batch.batch_index,
            sample_count=batch.sample_count,
            created_at=format_timestamp(batch.created_at),
            sample_fingerprints=fingerprints,
            version=__version__,
            generated_at=generated_at,
        )

        return card

    def _collect_sources(self, batches: List[BatchRecord]) -> List[SourceRecord]:
        """
        Collect unique sources from all batches.

        Args:
            batches: List of BatchRecord objects.

        Returns:
            List of unique SourceRecord objects.
        """
        source_ids: Set[str] = set()
        sources: List[SourceRecord] = []

        for batch in batches:
            samples = self._db.list_samples(batch.batch_id)
            for sample in samples:
                if sample.source_id not in source_ids:
                    source = self._db.get_source(sample.source_id)
                    if source is not None:
                        sources.append(source)
                        source_ids.add(sample.source_id)

        return sources

    def _generate_session_info(self, session: SessionRecord) -> str:
        """
        Generate the session information section.

        Args:
            session: The SessionRecord to format.

        Returns:
            Markdown-formatted session information.
        """
        return SECTION_TEMPLATES["session_info"].format(
            session_id=session.session_id,
            created_at=format_timestamp(session.created_at),
            status=session.status.capitalize(),
            config_hash=truncate_hash(session.config_hash, 16),
        )

    def _generate_sources_section(self, sources: List[SourceRecord]) -> str:
        """
        Generate the data sources section.

        Args:
            sources: List of SourceRecord objects.

        Returns:
            Markdown-formatted sources table.
        """
        if not sources:
            return SECTION_TEMPLATES["sources_empty"]

        headers = ["Source ID", "Type", "Path", "License"]
        rows = []

        for source in sources:
            rows.append([
                f"`{truncate_hash(source.source_id)}`",
                source.source_type,
                source.source_path,
                source.license_id or "Unknown",
            ])

        return format_table(headers, rows)

    def _generate_license_section(self, analysis: Dict[str, Any]) -> str:
        """
        Generate the license analysis section.

        Args:
            analysis: Output from LicenseAnalyzer.analyze_session().

        Returns:
            Markdown-formatted license analysis.
        """
        return format_license_summary(analysis)

    def _generate_stats_section(
        self,
        session_id: str,
        batches: List[BatchRecord],
    ) -> str:
        """
        Generate the sample statistics section.

        Args:
            session_id: The session ID.
            batches: List of BatchRecord objects.

        Returns:
            Markdown-formatted statistics.
        """
        total_samples = 0
        unique_sample_ids: Set[str] = set()
        total_bytes = 0
        content_types: Dict[str, int] = {}

        for batch in batches:
            samples = self._db.list_samples(batch.batch_id)
            for sample in samples:
                total_samples += 1
                unique_sample_ids.add(sample.sample_id)
                total_bytes += sample.byte_size

                ct = sample.content_type
                content_types[ct] = content_types.get(ct, 0) + 1

        # Build stats table
        stats = SECTION_TEMPLATES["stats"].format(
            batch_count=len(batches),
            sample_count=total_samples,
            unique_samples=len(unique_sample_ids),
            total_bytes=format_bytes(total_bytes),
        )

        # Add content type breakdown if there are samples
        if content_types:
            stats += "\n### Content Types\n\n"
            headers = ["Type", "Count"]
            rows = [[ct, str(count)] for ct, count in sorted(content_types.items())]
            stats += format_table(headers, rows)

        return stats

    def _generate_batch_section(self, batches: List[BatchRecord]) -> str:
        """
        Generate the batch summary section.

        Args:
            batches: List of BatchRecord objects.

        Returns:
            Markdown-formatted batch summary.
        """
        if not batches:
            return "No batches recorded for this session."

        headers = ["Batch Index", "Fingerprint", "Sample Count", "Created"]
        rows = []

        for batch in batches:
            rows.append([
                str(batch.batch_index),
                f"`{truncate_hash(batch.batch_id)}`",
                str(batch.sample_count),
                format_timestamp(batch.created_at),
            ])

        return format_table(headers, rows)

    def _generate_audit_section(
        self,
        session_id: str,
        session: SessionRecord,
        batches: List[BatchRecord],
    ) -> str:
        """
        Generate the audit trail section.

        Args:
            session_id: The session ID.
            session: The SessionRecord.
            batches: List of BatchRecord objects.

        Returns:
            Markdown-formatted audit trail.
        """
        events = []

        # Session start event
        events.append((
            session.created_at,
            f"Session started (config: `{truncate_hash(session.config_hash, 8)}`)",
        ))

        # Batch events
        for batch in batches:
            events.append((
                batch.created_at,
                f"Batch {batch.batch_index} recorded ({batch.sample_count} samples)",
            ))

        # License conflicts
        conflicts = self._db.list_conflicts(session_id)
        for conflict in conflicts:
            events.append((
                conflict["detected_at"],
                f"License conflict detected: {conflict['license_a']} vs {conflict['license_b']}",
            ))

        # Session end (if completed/failed)
        if session.status in ("completed", "failed"):
            # Use the last batch timestamp or session creation as proxy
            end_time = batches[-1].created_at if batches else session.created_at
            events.append((
                end_time,
                f"Session {session.status}",
            ))

        # Sort by timestamp
        events.sort(key=lambda x: x[0])

        # Format events
        lines = []
        for timestamp, event in events:
            formatted_time = format_timestamp(timestamp)
            lines.append(SECTION_TEMPLATES["audit_entry"].format(
                timestamp=formatted_time,
                event=event,
            ))

        return "\n".join(lines) if lines else "No events recorded."
