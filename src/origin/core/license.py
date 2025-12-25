"""
License tracking system for Origin provenance.

This module manages license metadata and detects potential conflicts when
data from multiple sources is combined in training. It provides:

- SPDX_LICENSES: Built-in definitions for common open source licenses
- LicenseRegistry: Registry for looking up and managing license definitions
- LicenseAnalyzer: Analysis tools for compatibility checking and propagation
- parse_license_string: Normalizes license strings to SPDX identifiers

The license tracking follows a conservative model - conflicts are flagged
for human review rather than making legal determinations.
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from origin.core.record import LicenseRecord

if TYPE_CHECKING:
    from origin.storage.database import ProvenanceDatabase


# =============================================================================
# SPDX License Definitions
# =============================================================================

SPDX_LICENSES: Dict[str, Dict[str, Any]] = {
    "MIT": {
        "license_name": "MIT License",
        "license_url": "https://opensource.org/licenses/MIT",
        "permissions": ("commercial", "modify", "distribute", "private-use"),
        "restrictions": ("no-warranty", "no-liability"),
        "conditions": ("attribution", "license-notice"),
        "copyleft": False,
    },
    "Apache-2.0": {
        "license_name": "Apache License 2.0",
        "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
        "permissions": ("commercial", "modify", "distribute", "private-use", "patent-grant"),
        "restrictions": ("no-warranty", "no-liability", "no-trademark"),
        "conditions": ("attribution", "license-notice", "state-changes"),
        "copyleft": False,
    },
    "GPL-3.0-only": {
        "license_name": "GNU General Public License v3.0 only",
        "license_url": "https://www.gnu.org/licenses/gpl-3.0.html",
        "permissions": ("commercial", "modify", "distribute", "private-use", "patent-grant"),
        "restrictions": ("no-warranty", "no-liability"),
        "conditions": ("attribution", "license-notice", "state-changes", "disclose-source", "same-license"),
        "copyleft": True,
    },
    "GPL-2.0-only": {
        "license_name": "GNU General Public License v2.0 only",
        "license_url": "https://www.gnu.org/licenses/gpl-2.0.html",
        "permissions": ("commercial", "modify", "distribute", "private-use"),
        "restrictions": ("no-warranty", "no-liability"),
        "conditions": ("attribution", "license-notice", "state-changes", "disclose-source", "same-license"),
        "copyleft": True,
    },
    "BSD-3-Clause": {
        "license_name": "BSD 3-Clause License",
        "license_url": "https://opensource.org/licenses/BSD-3-Clause",
        "permissions": ("commercial", "modify", "distribute", "private-use"),
        "restrictions": ("no-warranty", "no-liability", "no-endorsement"),
        "conditions": ("attribution", "license-notice"),
        "copyleft": False,
    },
    "BSD-2-Clause": {
        "license_name": "BSD 2-Clause License",
        "license_url": "https://opensource.org/licenses/BSD-2-Clause",
        "permissions": ("commercial", "modify", "distribute", "private-use"),
        "restrictions": ("no-warranty", "no-liability"),
        "conditions": ("attribution", "license-notice"),
        "copyleft": False,
    },
    "CC-BY-4.0": {
        "license_name": "Creative Commons Attribution 4.0 International",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "permissions": ("commercial", "modify", "distribute", "private-use"),
        "restrictions": ("no-warranty", "no-liability"),
        "conditions": ("attribution", "license-notice"),
        "copyleft": False,
    },
    "CC-BY-SA-4.0": {
        "license_name": "Creative Commons Attribution-ShareAlike 4.0 International",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "permissions": ("commercial", "modify", "distribute", "private-use"),
        "restrictions": ("no-warranty", "no-liability"),
        "conditions": ("attribution", "license-notice", "same-license"),
        "copyleft": True,
    },
    "CC-BY-NC-4.0": {
        "license_name": "Creative Commons Attribution-NonCommercial 4.0 International",
        "license_url": "https://creativecommons.org/licenses/by-nc/4.0/",
        "permissions": ("modify", "distribute", "private-use"),
        "restrictions": ("no-warranty", "no-liability", "non-commercial"),
        "conditions": ("attribution", "license-notice"),
        "copyleft": False,
    },
    "CC0-1.0": {
        "license_name": "Creative Commons Zero v1.0 Universal",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "permissions": ("commercial", "modify", "distribute", "private-use"),
        "restrictions": ("no-warranty", "no-liability"),
        "conditions": (),
        "copyleft": False,
    },
    "proprietary": {
        "license_name": "Proprietary License",
        "license_url": None,
        "permissions": (),
        "restrictions": ("all-rights-reserved",),
        "conditions": (),
        "copyleft": False,
    },
    "unknown": {
        "license_name": "Unknown License",
        "license_url": None,
        "permissions": (),
        "restrictions": ("unknown-terms",),
        "conditions": (),
        "copyleft": False,
    },
}


# =============================================================================
# License Registry
# =============================================================================

class LicenseRegistry:
    """
    Registry for license definitions.

    The registry is pre-populated with common SPDX license definitions and
    supports registration of custom licenses. Lookups are case-insensitive.

    Example:
        >>> registry = LicenseRegistry()
        >>> mit = registry.get("MIT")
        >>> mit.license_name
        'MIT License'
    """

    def __init__(self) -> None:
        """Initialize the registry with SPDX license definitions."""
        self._licenses: Dict[str, LicenseRecord] = {}

        # Populate from SPDX_LICENSES
        for license_id, data in SPDX_LICENSES.items():
            record = LicenseRecord(
                license_id=license_id,
                license_name=data["license_name"],
                license_url=data["license_url"],
                permissions=tuple(data["permissions"]),
                restrictions=tuple(data["restrictions"]),
                conditions=tuple(data["conditions"]),
                copyleft=data["copyleft"],
            )
            self._licenses[license_id.lower()] = record

    def register(self, license: LicenseRecord) -> None:
        """
        Register a license definition.

        If a license with the same ID already exists, it will be overwritten.

        Args:
            license: The LicenseRecord to register.
        """
        self._licenses[license.license_id.lower()] = license

    def get(self, license_id: str) -> Optional[LicenseRecord]:
        """
        Look up a license by ID.

        The lookup is case-insensitive.

        Args:
            license_id: The SPDX identifier or custom license ID.

        Returns:
            The LicenseRecord if found, None otherwise.
        """
        return self._licenses.get(license_id.lower())

    def resolve(self, identifier: str) -> LicenseRecord:
        """
        Resolve a license identifier to a LicenseRecord.

        If the license is not found, returns the 'unknown' license.

        Args:
            identifier: The license identifier to resolve.

        Returns:
            The matching LicenseRecord, or the 'unknown' license if not found.
        """
        result = self.get(identifier)
        if result is not None:
            return result
        return self._licenses["unknown"]

    def list_all(self) -> List[str]:
        """
        List all registered license IDs.

        Returns:
            List of license IDs in the registry.
        """
        return [lic.license_id for lic in self._licenses.values()]


# =============================================================================
# License Analyzer
# =============================================================================

class LicenseAnalyzer:
    """
    Analyzer for license compatibility and propagation.

    This class provides tools for checking license compatibility and
    computing the effective license terms when multiple licenses are combined.

    The analyzer follows a conservative model:
    - Conflicts are flagged for human review
    - Permissions use intersection (most restrictive)
    - Restrictions and conditions use union (all apply)

    Example:
        >>> analyzer = LicenseAnalyzer()
        >>> compatible, reason = analyzer.check_compatibility("MIT", "Apache-2.0")
        >>> compatible
        True
    """

    def __init__(self, registry: Optional[LicenseRegistry] = None) -> None:
        """
        Initialize the analyzer.

        Args:
            registry: LicenseRegistry to use for lookups. If None, a default
                registry with SPDX licenses is created.
        """
        self._registry = registry if registry is not None else LicenseRegistry()

    @property
    def registry(self) -> LicenseRegistry:
        """Return the license registry."""
        return self._registry

    def check_compatibility(
        self,
        license_a_id: str,
        license_b_id: str,
    ) -> Tuple[bool, str]:
        """
        Check if two licenses are compatible.

        This performs a conservative compatibility check. Licenses are
        considered incompatible if:
        - One is copyleft and the other is proprietary
        - Both are copyleft but different (potential conflict)

        Unknown licenses generate warnings but are not treated as conflicts.

        Args:
            license_a_id: First license identifier.
            license_b_id: Second license identifier.

        Returns:
            A tuple of (is_compatible, reason) where:
            - is_compatible: True if licenses can be combined, False if conflict
            - reason: Human-readable explanation
        """
        license_a = self._registry.resolve(license_a_id)
        license_b = self._registry.resolve(license_b_id)

        # Same license is always compatible
        if license_a.license_id.lower() == license_b.license_id.lower():
            return (True, "same license")

        # Unknown license - warn but don't conflict
        if license_a.license_id.lower() == "unknown" or license_b.license_id.lower() == "unknown":
            return (True, "unknown license present")

        # Check for proprietary
        a_is_proprietary = license_a.license_id.lower() == "proprietary"
        b_is_proprietary = license_b.license_id.lower() == "proprietary"

        # Copyleft + proprietary = conflict
        if license_a.copyleft and b_is_proprietary:
            return (False, "copyleft-proprietary conflict")
        if license_b.copyleft and a_is_proprietary:
            return (False, "copyleft-proprietary conflict")

        # Different copyleft licenses = potential conflict
        if license_a.copyleft and license_b.copyleft:
            return (False, "incompatible copyleft licenses")

        # Check for non-commercial restrictions
        a_has_nc = "non-commercial" in license_a.restrictions
        b_has_nc = "non-commercial" in license_b.restrictions
        a_allows_commercial = "commercial" in license_a.permissions
        b_allows_commercial = "commercial" in license_b.permissions

        if (a_has_nc and b_allows_commercial) or (b_has_nc and a_allows_commercial):
            return (True, "commercial use restricted")

        return (True, "compatible")

    def propagate(self, license_ids: List[str]) -> Dict[str, Any]:
        """
        Analyze a collection of licenses and compute propagated terms.

        When multiple licenses are combined:
        - Permissions: intersection (only permissions granted by all)
        - Restrictions: union (all restrictions apply)
        - Conditions: union (all conditions apply)

        Args:
            license_ids: List of license identifiers to analyze.

        Returns:
            Dictionary with:
            - result_license: The effective license ID ('mixed' if multiple)
            - permissions: Tuple of permissions granted by all licenses
            - restrictions: Tuple of all restrictions that apply
            - conditions: Tuple of all conditions that apply
            - conflicts: List of conflict dicts with license_a, license_b, reason
            - warnings: List of warning strings
        """
        if not license_ids:
            return {
                "result_license": "unknown",
                "permissions": (),
                "restrictions": ("unknown-terms",),
                "conditions": (),
                "conflicts": [],
                "warnings": ["no licenses specified"],
            }

        # Resolve all licenses
        licenses = [self._registry.resolve(lid) for lid in license_ids]
        unique_ids = set(lic.license_id.lower() for lic in licenses)

        # Determine result license
        if len(unique_ids) == 1:
            result_license = licenses[0].license_id
        else:
            result_license = "mixed"

        # Compute permissions intersection
        if licenses:
            permissions_sets = [set(lic.permissions) for lic in licenses]
            permissions_intersection = permissions_sets[0]
            for pset in permissions_sets[1:]:
                permissions_intersection = permissions_intersection.intersection(pset)
            permissions = tuple(sorted(permissions_intersection))
        else:
            permissions = ()

        # Compute restrictions union
        restrictions_union: Set[str] = set()
        for lic in licenses:
            restrictions_union.update(lic.restrictions)
        restrictions = tuple(sorted(restrictions_union))

        # Compute conditions union
        conditions_union: Set[str] = set()
        for lic in licenses:
            conditions_union.update(lic.conditions)
        conditions = tuple(sorted(conditions_union))

        # Check for conflicts between all pairs
        conflicts: List[Dict[str, str]] = []
        warnings: List[str] = []
        checked_pairs: Set[Tuple[str, str]] = set()

        for i, lic_a in enumerate(licenses):
            for lic_b in licenses[i + 1:]:
                # Normalize pair order for deduplication
                pair = tuple(sorted([lic_a.license_id.lower(), lic_b.license_id.lower()]))
                if pair in checked_pairs:
                    continue
                checked_pairs.add(pair)

                compatible, reason = self.check_compatibility(
                    lic_a.license_id,
                    lic_b.license_id,
                )

                if not compatible:
                    conflicts.append({
                        "license_a": lic_a.license_id,
                        "license_b": lic_b.license_id,
                        "reason": reason,
                    })
                elif reason not in ("same license", "compatible"):
                    warnings.append(f"{lic_a.license_id} + {lic_b.license_id}: {reason}")

        return {
            "result_license": result_license,
            "permissions": permissions,
            "restrictions": restrictions,
            "conditions": conditions,
            "conflicts": conflicts,
            "warnings": warnings,
        }

    def analyze_session(
        self,
        db: "ProvenanceDatabase",
        session_id: str,
    ) -> Dict[str, Any]:
        """
        Analyze all licenses used in a training session.

        This aggregates license information across all batches in the session
        and computes the propagated license terms.

        Args:
            db: ProvenanceDatabase to query.
            session_id: Session ID to analyze.

        Returns:
            Dictionary with propagate() results plus:
            - batch_count: Number of batches in session
            - sample_count: Total number of samples
        """
        # Get all batches for session
        batches = db.list_batches(session_id)
        batch_count = len(batches)
        sample_count = 0

        # Collect all unique license IDs
        license_ids: List[str] = []

        for batch in batches:
            samples = db.list_samples(batch.batch_id)
            sample_count += len(samples)

            for sample in samples:
                source = db.get_source(sample.source_id)
                if source and source.license_id:
                    license_ids.append(source.license_id)

        # Run propagation analysis
        result = self.propagate(license_ids)
        result["batch_count"] = batch_count
        result["sample_count"] = sample_count

        return result


# =============================================================================
# License String Parser
# =============================================================================

# Common license string patterns and their SPDX mappings
_LICENSE_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # MIT variations
    (re.compile(r"^mit\s*(license)?$", re.IGNORECASE), "MIT"),
    (re.compile(r"^the\s+mit\s+license$", re.IGNORECASE), "MIT"),

    # Apache variations
    (re.compile(r"^apache(\s+license)?[\s\-]*2(\.0)?$", re.IGNORECASE), "Apache-2.0"),
    (re.compile(r"^apache[\s\-]*2\.0$", re.IGNORECASE), "Apache-2.0"),
    (re.compile(r"^asl[\s\-]*2(\.0)?$", re.IGNORECASE), "Apache-2.0"),

    # GPL variations
    (re.compile(r"^gpl[\s\-]*v?3(\.0)?(\s*only)?$", re.IGNORECASE), "GPL-3.0-only"),
    (re.compile(r"^gnu\s+(general\s+)?public\s+license\s+v?3(\.0)?$", re.IGNORECASE), "GPL-3.0-only"),
    (re.compile(r"^gpl[\s\-]*v?2(\.0)?(\s*only)?$", re.IGNORECASE), "GPL-2.0-only"),
    (re.compile(r"^gnu\s+(general\s+)?public\s+license\s+v?2(\.0)?$", re.IGNORECASE), "GPL-2.0-only"),

    # BSD variations
    (re.compile(r"^bsd[\s\-]*3[\s\-]*clause$", re.IGNORECASE), "BSD-3-Clause"),
    (re.compile(r"^bsd[\s\-]*2[\s\-]*clause$", re.IGNORECASE), "BSD-2-Clause"),
    (re.compile(r"^bsd$", re.IGNORECASE), "BSD-3-Clause"),  # Default to 3-clause
    (re.compile(r"^bsd[\s\-]*(license)?$", re.IGNORECASE), "BSD-3-Clause"),

    # Creative Commons variations
    (re.compile(r"^cc[\s\-]*by[\s\-]*4(\.0)?$", re.IGNORECASE), "CC-BY-4.0"),
    (re.compile(r"^cc[\s\-]*by[\s\-]*sa[\s\-]*4(\.0)?$", re.IGNORECASE), "CC-BY-SA-4.0"),
    (re.compile(r"^cc[\s\-]*by[\s\-]*nc[\s\-]*4(\.0)?$", re.IGNORECASE), "CC-BY-NC-4.0"),
    (re.compile(r"^cc[\s\-]*0$", re.IGNORECASE), "CC0-1.0"),
    (re.compile(r"^cc0[\s\-]*1(\.0)?$", re.IGNORECASE), "CC0-1.0"),
    (re.compile(r"^creative\s+commons\s+zero$", re.IGNORECASE), "CC0-1.0"),
    (re.compile(r"^public\s+domain$", re.IGNORECASE), "CC0-1.0"),

    # Proprietary
    (re.compile(r"^proprietary$", re.IGNORECASE), "proprietary"),
    (re.compile(r"^all\s+rights\s+reserved$", re.IGNORECASE), "proprietary"),
    (re.compile(r"^commercial$", re.IGNORECASE), "proprietary"),

    # Unknown
    (re.compile(r"^unknown$", re.IGNORECASE), "unknown"),
    (re.compile(r"^unlicensed$", re.IGNORECASE), "unknown"),
    (re.compile(r"^no\s+license$", re.IGNORECASE), "unknown"),
]


def parse_license_string(text: str) -> str:
    """
    Normalize a license string to an SPDX identifier.

    This function attempts to match common license string variations to their
    canonical SPDX identifiers. If no match is found, the input is returned
    stripped and lowercased.

    Args:
        text: The license string to parse.

    Returns:
        The SPDX identifier if a match is found, otherwise the input string
        stripped and lowercased.

    Examples:
        >>> parse_license_string("MIT License")
        'MIT'
        >>> parse_license_string("Apache 2.0")
        'Apache-2.0'
        >>> parse_license_string("GPLv3")
        'GPL-3.0-only'
        >>> parse_license_string("BSD")
        'BSD-3-Clause'
        >>> parse_license_string("CC BY 4.0")
        'CC-BY-4.0'
    """
    text = text.strip()

    # Try exact match first (case-insensitive)
    text_lower = text.lower()
    for spdx_id in SPDX_LICENSES:
        if text_lower == spdx_id.lower():
            return spdx_id

    # Try pattern matching
    for pattern, spdx_id in _LICENSE_PATTERNS:
        if pattern.match(text):
            return spdx_id

    # No match found - return normalized input
    return text_lower
