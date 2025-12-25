"""
Origin - Runtime data provenance for AI pipelines.

Origin tracks data samples as they flow through AI training pipelines,
generating cryptographic fingerprints and propagating license metadata
for compliance and audit purposes.
"""

__version__ = "0.1.0"
__all__ = [
    "__version__",
    "ProvenanceRecord",
    "SessionRecord",
    "BatchRecord",
    "SourceRecord",
    "LicenseRecord",
]


def __getattr__(name: str):
    """Lazy import of main classes to avoid circular imports."""
    if name in (
        "ProvenanceRecord",
        "SessionRecord",
        "BatchRecord",
        "SourceRecord",
        "LicenseRecord",
    ):
        from origin.core.record import (
            ProvenanceRecord,
            SessionRecord,
            BatchRecord,
            SourceRecord,
            LicenseRecord,
        )
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
