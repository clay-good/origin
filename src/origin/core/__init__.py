"""
Origin core module.

Contains fundamental data types and algorithms for provenance tracking.
"""

from origin.core.record import (
    ProvenanceRecord,
    SessionRecord,
    BatchRecord,
    SourceRecord,
    LicenseRecord,
)

from origin.core.fingerprint import (
    fingerprint_bytes,
    fingerprint_text,
    fingerprint_tensor,
    fingerprint_dict,
    fingerprint_sample,
    merkle_root,
    FingerprintCache,
)

from origin.core.license import (
    SPDX_LICENSES,
    LicenseRegistry,
    LicenseAnalyzer,
    parse_license_string,
)

__all__ = [
    # Record types
    "ProvenanceRecord",
    "SessionRecord",
    "BatchRecord",
    "SourceRecord",
    "LicenseRecord",
    # Fingerprinting
    "fingerprint_bytes",
    "fingerprint_text",
    "fingerprint_tensor",
    "fingerprint_dict",
    "fingerprint_sample",
    "merkle_root",
    "FingerprintCache",
    # License tracking
    "SPDX_LICENSES",
    "LicenseRegistry",
    "LicenseAnalyzer",
    "parse_license_string",
]
