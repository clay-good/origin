"""
Core data record types for Origin provenance tracking.

This module defines immutable dataclasses representing the fundamental
data structures used throughout Origin for tracking data provenance.

All dataclasses are:
- Immutable (frozen=True) to ensure data integrity
- Serializable via to_dict() and from_dict() methods
- Validated in __post_init__ for required fields

Note: For Python 3.9 compatibility, we use frozen=True without explicit
__slots__. The dataclass decorator handles memory layout appropriately.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, Tuple


@dataclass(frozen=True)
class ProvenanceRecord:
    """
    Represents a single observation of a data sample.

    A ProvenanceRecord captures metadata about an individual data sample
    as it passes through a training pipeline, including its fingerprint,
    source, content type, and size.

    Attributes:
        sample_id: SHA-256 fingerprint of the sample content (64 hex chars).
        source_id: Identifier of the data source this sample came from.
        content_type: Type of content (e.g., 'text', 'image', 'audio', 'tabular').
        byte_size: Size of the sample in bytes.
        timestamp: ISO 8601 timestamp when the sample was observed.
        metadata: Additional metadata as a dictionary.
    """

    sample_id: str
    source_id: str
    content_type: str
    byte_size: int
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate required fields are non-empty."""
        if not self.sample_id or not isinstance(self.sample_id, str):
            raise ValueError("sample_id must be a non-empty string")
        if not self.source_id or not isinstance(self.source_id, str):
            raise ValueError("source_id must be a non-empty string")
        if not self.content_type or not isinstance(self.content_type, str):
            raise ValueError("content_type must be a non-empty string")
        if not isinstance(self.byte_size, int) or self.byte_size < 0:
            raise ValueError("byte_size must be a non-negative integer")
        if not self.timestamp or not isinstance(self.timestamp, str):
            raise ValueError("timestamp must be a non-empty string")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the record to a plain dictionary.

        Returns:
            A dictionary representation of all fields.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProvenanceRecord":
        """
        Create a ProvenanceRecord from a dictionary.

        Args:
            data: Dictionary containing record fields.

        Returns:
            A new ProvenanceRecord instance.

        Raises:
            TypeError: If required fields are missing.
            ValueError: If field validation fails.
        """
        return cls(
            sample_id=data["sample_id"],
            source_id=data["source_id"],
            content_type=data["content_type"],
            byte_size=data["byte_size"],
            timestamp=data["timestamp"],
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class SessionRecord:
    """
    Represents a training session for provenance tracking.

    A SessionRecord captures metadata about a training run, including
    when it started, its configuration, and current status.

    Attributes:
        session_id: Unique identifier for the session (UUID).
        created_at: ISO 8601 timestamp when the session was created.
        config_hash: SHA-256 hash of the training configuration.
        status: Current status ('running', 'completed', 'failed').
        metadata: Additional metadata as a dictionary.
    """

    session_id: str
    created_at: str
    config_hash: str
    status: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate required fields are non-empty."""
        if not self.session_id or not isinstance(self.session_id, str):
            raise ValueError("session_id must be a non-empty string")
        if not self.created_at or not isinstance(self.created_at, str):
            raise ValueError("created_at must be a non-empty string")
        if not self.config_hash or not isinstance(self.config_hash, str):
            raise ValueError("config_hash must be a non-empty string")
        if not self.status or not isinstance(self.status, str):
            raise ValueError("status must be a non-empty string")
        if self.status not in ("running", "completed", "failed"):
            raise ValueError("status must be 'running', 'completed', or 'failed'")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the record to a plain dictionary.

        Returns:
            A dictionary representation of all fields.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionRecord":
        """
        Create a SessionRecord from a dictionary.

        Args:
            data: Dictionary containing record fields.

        Returns:
            A new SessionRecord instance.

        Raises:
            TypeError: If required fields are missing.
            ValueError: If field validation fails.
        """
        return cls(
            session_id=data["session_id"],
            created_at=data["created_at"],
            config_hash=data["config_hash"],
            status=data["status"],
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class BatchRecord:
    """
    Represents a batch of samples processed together.

    A BatchRecord aggregates multiple samples into a single unit with
    a Merkle root fingerprint for efficient verification.

    Attributes:
        batch_id: Merkle root of all sample fingerprints in the batch.
        session_id: ID of the session this batch belongs to.
        batch_index: Sequential index of this batch within the session.
        sample_count: Number of samples in this batch.
        created_at: ISO 8601 timestamp when the batch was recorded.
        sample_ids: Tuple of sample IDs in batch order.
    """

    batch_id: str
    session_id: str
    batch_index: int
    sample_count: int
    created_at: str
    sample_ids: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate required fields."""
        if not self.batch_id or not isinstance(self.batch_id, str):
            raise ValueError("batch_id must be a non-empty string")
        if not self.session_id or not isinstance(self.session_id, str):
            raise ValueError("session_id must be a non-empty string")
        if not isinstance(self.batch_index, int) or self.batch_index < 0:
            raise ValueError("batch_index must be a non-negative integer")
        if not isinstance(self.sample_count, int) or self.sample_count < 0:
            raise ValueError("sample_count must be a non-negative integer")
        if not self.created_at or not isinstance(self.created_at, str):
            raise ValueError("created_at must be a non-empty string")
        if not isinstance(self.sample_ids, tuple):
            raise ValueError("sample_ids must be a tuple")
        for sid in self.sample_ids:
            if not isinstance(sid, str):
                raise ValueError("all sample_ids must be strings")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the record to a plain dictionary.

        Returns:
            A dictionary representation of all fields.
            Note: sample_ids is converted to a list for JSON compatibility.
        """
        result = asdict(self)
        result["sample_ids"] = list(self.sample_ids)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatchRecord":
        """
        Create a BatchRecord from a dictionary.

        Args:
            data: Dictionary containing record fields.

        Returns:
            A new BatchRecord instance.

        Raises:
            TypeError: If required fields are missing.
            ValueError: If field validation fails.
        """
        sample_ids = data.get("sample_ids", [])
        if isinstance(sample_ids, list):
            sample_ids = tuple(sample_ids)
        return cls(
            batch_id=data["batch_id"],
            session_id=data["session_id"],
            batch_index=data["batch_index"],
            sample_count=data["sample_count"],
            created_at=data["created_at"],
            sample_ids=sample_ids,
        )


@dataclass(frozen=True)
class SourceRecord:
    """
    Represents a data source from which samples originate.

    A SourceRecord tracks metadata about where training data comes from,
    including its location and associated license.

    Attributes:
        source_id: Content-addressable hash identifying this source.
        source_type: Type of source ('file', 'url', 'dataset_name').
        source_path: Original location or identifier of the source.
        license_id: SPDX identifier or custom ID of the license (optional).
        first_seen: ISO 8601 timestamp when source was first observed.
        metadata: Additional metadata as a dictionary.
    """

    source_id: str
    source_type: str
    source_path: str
    license_id: Optional[str]
    first_seen: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate required fields are non-empty."""
        if not self.source_id or not isinstance(self.source_id, str):
            raise ValueError("source_id must be a non-empty string")
        if not self.source_type or not isinstance(self.source_type, str):
            raise ValueError("source_type must be a non-empty string")
        if not self.source_path or not isinstance(self.source_path, str):
            raise ValueError("source_path must be a non-empty string")
        if self.license_id is not None and not isinstance(self.license_id, str):
            raise ValueError("license_id must be a string or None")
        if not self.first_seen or not isinstance(self.first_seen, str):
            raise ValueError("first_seen must be a non-empty string")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the record to a plain dictionary.

        Returns:
            A dictionary representation of all fields.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SourceRecord":
        """
        Create a SourceRecord from a dictionary.

        Args:
            data: Dictionary containing record fields.

        Returns:
            A new SourceRecord instance.

        Raises:
            TypeError: If required fields are missing.
            ValueError: If field validation fails.
        """
        return cls(
            source_id=data["source_id"],
            source_type=data["source_type"],
            source_path=data["source_path"],
            license_id=data.get("license_id"),
            first_seen=data["first_seen"],
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class LicenseRecord:
    """
    Represents license information for data sources.

    A LicenseRecord captures the terms and conditions of a license,
    including what is permitted, restricted, and required.

    Attributes:
        license_id: SPDX identifier or custom identifier for the license.
        license_name: Human-readable name of the license.
        license_url: URL to the license text (optional).
        permissions: Tuple of permitted actions (e.g., 'commercial', 'modify').
        restrictions: Tuple of restrictions (e.g., 'no-warranty').
        conditions: Tuple of conditions (e.g., 'attribution', 'share-alike').
        copyleft: Whether the license has copyleft provisions.
    """

    license_id: str
    license_name: str
    license_url: Optional[str]
    permissions: Tuple[str, ...]
    restrictions: Tuple[str, ...]
    conditions: Tuple[str, ...]
    copyleft: bool

    def __post_init__(self) -> None:
        """Validate required fields."""
        if not self.license_id or not isinstance(self.license_id, str):
            raise ValueError("license_id must be a non-empty string")
        if not self.license_name or not isinstance(self.license_name, str):
            raise ValueError("license_name must be a non-empty string")
        if self.license_url is not None and not isinstance(self.license_url, str):
            raise ValueError("license_url must be a string or None")
        if not isinstance(self.permissions, tuple):
            raise ValueError("permissions must be a tuple")
        if not isinstance(self.restrictions, tuple):
            raise ValueError("restrictions must be a tuple")
        if not isinstance(self.conditions, tuple):
            raise ValueError("conditions must be a tuple")
        if not isinstance(self.copyleft, bool):
            raise ValueError("copyleft must be a boolean")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the record to a plain dictionary.

        Returns:
            A dictionary representation of all fields.
            Note: tuples are converted to lists for JSON compatibility.
        """
        result = asdict(self)
        result["permissions"] = list(self.permissions)
        result["restrictions"] = list(self.restrictions)
        result["conditions"] = list(self.conditions)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LicenseRecord":
        """
        Create a LicenseRecord from a dictionary.

        Args:
            data: Dictionary containing record fields.

        Returns:
            A new LicenseRecord instance.

        Raises:
            TypeError: If required fields are missing.
            ValueError: If field validation fails.
        """
        permissions = data.get("permissions", [])
        restrictions = data.get("restrictions", [])
        conditions = data.get("conditions", [])

        if isinstance(permissions, list):
            permissions = tuple(permissions)
        if isinstance(restrictions, list):
            restrictions = tuple(restrictions)
        if isinstance(conditions, list):
            conditions = tuple(conditions)

        return cls(
            license_id=data["license_id"],
            license_name=data["license_name"],
            license_url=data.get("license_url"),
            permissions=permissions,
            restrictions=restrictions,
            conditions=conditions,
            copyleft=data["copyleft"],
        )
