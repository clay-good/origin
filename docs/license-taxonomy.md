# Origin License Taxonomy

This document describes the license tracking system in Origin, including supported licenses, compatibility rules, and propagation logic.

## 1. Overview

### Purpose of License Tracking

Machine learning models are trained on data from diverse sources, each with its own licensing terms. Origin tracks license metadata throughout the training pipeline to:

- **Identify data sources**: Know which licenses apply to your training data
- **Detect conflicts**: Flag potential incompatibilities between licenses
- **Enable compliance**: Generate audit trails for regulatory requirements
- **Document provenance**: Create accurate model cards with license information

### Importance for Compliance

Regulations like the EU AI Act require transparency about training data. Origin provides:

- Traceable records of all data used in training
- License information for each data source
- Conflict detection for incompatible licenses
- Exportable provenance reports for auditors

## 2. Supported Licenses

Origin includes built-in definitions for common open source and data licenses.

### SPDX License Table

| SPDX ID | Name | Copyleft | Key Restrictions |
|---------|------|----------|------------------|
| `MIT` | MIT License | No | None |
| `Apache-2.0` | Apache License 2.0 | No | No trademark use |
| `GPL-3.0-only` | GNU GPL v3.0 | Yes | Disclose source, same license |
| `GPL-2.0-only` | GNU GPL v2.0 | Yes | Disclose source, same license |
| `BSD-3-Clause` | BSD 3-Clause | No | No endorsement |
| `BSD-2-Clause` | BSD 2-Clause | No | None |
| `CC-BY-4.0` | CC Attribution 4.0 | No | Attribution required |
| `CC-BY-SA-4.0` | CC Attribution-ShareAlike 4.0 | Yes | Same license required |
| `CC-BY-NC-4.0` | CC Attribution-NonCommercial 4.0 | No | Non-commercial only |
| `CC0-1.0` | CC Zero (Public Domain) | No | None |
| `proprietary` | Proprietary License | No | All rights reserved |
| `unknown` | Unknown License | No | Unknown terms |

## 3. License Properties

Each license in Origin is defined by four key properties:

### Permissions

What the license allows:

| Permission | Description |
|------------|-------------|
| `commercial` | Use for commercial purposes |
| `modify` | Create derivative works |
| `distribute` | Distribute copies |
| `private-use` | Use privately |
| `patent-grant` | Grants patent rights |

### Restrictions

What the license prohibits or limits:

| Restriction | Description |
|-------------|-------------|
| `no-warranty` | No warranty provided |
| `no-liability` | No liability assumed |
| `no-trademark` | Cannot use trademarks |
| `no-endorsement` | Cannot claim endorsement |
| `non-commercial` | Commercial use prohibited |
| `all-rights-reserved` | No rights granted |
| `unknown-terms` | Terms not known |

### Conditions

Requirements that must be met:

| Condition | Description |
|-----------|-------------|
| `attribution` | Credit the original author |
| `license-notice` | Include license text |
| `state-changes` | Document modifications |
| `disclose-source` | Make source available |
| `same-license` | Use same license for derivatives |

### Copyleft

Whether the license is copyleft:

- **Copyleft licenses** (e.g., GPL, CC-BY-SA) require derivative works to use the same license
- **Permissive licenses** (e.g., MIT, Apache) allow relicensing under different terms

## 4. Compatibility Rules

Origin uses conservative rules to detect potential license conflicts.

### Compatibility Matrix

| License A | License B | Compatible | Notes |
|-----------|-----------|------------|-------|
| MIT | Apache-2.0 | Yes | Permissive licenses combine freely |
| MIT | GPL-3.0-only | Yes | Permissive can be incorporated into copyleft |
| MIT | proprietary | Yes | Permissive allows proprietary use |
| GPL-3.0-only | GPL-2.0-only | **No** | Different copyleft versions incompatible |
| GPL-3.0-only | proprietary | **No** | Copyleft conflicts with proprietary |
| CC-BY-4.0 | CC-BY-NC-4.0 | Yes | NC restriction noted but not a conflict |
| unknown | any | Yes | Unknown flagged but not blocked |

### Conflict Types

| Type | Description | Example |
|------|-------------|---------|
| `copyleft-proprietary` | Copyleft cannot mix with proprietary | GPL + proprietary |
| `incompatible_copyleft` | Different copyleft licenses conflict | GPL-3.0 + GPL-2.0 |

### What Triggers a Conflict

Conflicts are detected when:

1. **Both licenses are copyleft** and they are different
2. **One license is copyleft** and the other is proprietary

### What Does NOT Trigger a Conflict

The following are compatible despite different terms:

- Permissive licenses (MIT, Apache, BSD) combine freely
- Permissive licenses can be incorporated into copyleft projects
- Unknown licenses generate warnings but not conflicts
- Non-commercial restrictions are noted but not conflicts

## 5. Propagation Logic

When multiple licenses are combined in a training session, Origin computes the effective terms.

### Propagation Rules

| Property | Rule | Rationale |
|----------|------|-----------|
| Permissions | Intersection | Only permissions granted by ALL licenses |
| Restrictions | Union | ALL restrictions apply |
| Conditions | Union | ALL conditions must be satisfied |

### Example: MIT + Apache-2.0

```
MIT:
  permissions: [commercial, modify, distribute, private-use]
  restrictions: [no-warranty, no-liability]
  conditions: [attribution, license-notice]

Apache-2.0:
  permissions: [commercial, modify, distribute, private-use, patent-grant]
  restrictions: [no-warranty, no-liability, no-trademark]
  conditions: [attribution, license-notice, state-changes]

Combined:
  permissions: [commercial, distribute, modify, private-use]  # Intersection
  restrictions: [no-liability, no-trademark, no-warranty]     # Union
  conditions: [attribution, license-notice, state-changes]    # Union
  result_license: "mixed"
```

### Result License

The `result_license` field indicates:

| Value | Meaning |
|-------|---------|
| `MIT` | All data uses MIT license |
| `Apache-2.0` | All data uses Apache-2.0 license |
| `mixed` | Multiple different licenses present |
| `unknown` | No licenses specified |

## 6. Custom Licenses

### Defining a Custom License

```python
from origin.core.record import LicenseRecord

custom = LicenseRecord(
    license_id="COMPANY-INTERNAL-1.0",
    license_name="Company Internal Data License v1.0",
    license_url="https://internal.company.com/data-license",
    permissions=("modify", "private-use"),
    restrictions=("no-redistribution", "no-commercial", "no-warranty"),
    conditions=("attribution",),
    copyleft=False
)

db.record_license(custom)
```

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `license_id` | str | Unique identifier (required) |
| `license_name` | str | Human-readable name (required) |
| `license_url` | str | URL to license text (optional) |
| `permissions` | tuple | Permissions granted |
| `restrictions` | tuple | Restrictions imposed |
| `conditions` | tuple | Conditions required |
| `copyleft` | bool | Whether copyleft applies |

### Registering Custom Licenses

```python
from origin.core.license import LicenseRegistry

registry = LicenseRegistry()

# Register custom license
registry.register(custom_license)

# Retrieve custom license
license_info = registry.get("COMPANY-INTERNAL-1.0")
```

## 7. Common Scenarios

### Scenario 1: Mixing MIT and Apache

Two common permissive licenses used together.

```
Sources:
  - dataset_a: MIT
  - dataset_b: Apache-2.0

Result:
  - result_license: "mixed"
  - conflicts: []
  - permissions: [commercial, distribute, modify, private-use]
  - restrictions: [no-liability, no-trademark, no-warranty]
  - conditions: [attribution, license-notice, state-changes]

Conclusion: COMPATIBLE
  Both are permissive licenses. The combined work must satisfy
  both sets of conditions (attribution, license notice, and
  state changes for Apache content).
```

### Scenario 2: Mixing GPL and Proprietary

A copyleft license combined with proprietary data.

```
Sources:
  - open_dataset: GPL-3.0-only
  - licensed_dataset: proprietary

Result:
  - result_license: "mixed"
  - conflicts: [
      {
        "license_a": "GPL-3.0-only",
        "license_b": "proprietary",
        "reason": "copyleft-proprietary conflict"
      }
    ]

Conclusion: CONFLICT
  GPL requires derivative works to be open source.
  Proprietary data cannot be combined with GPL.
  This combination requires legal review.
```

### Scenario 3: Creative Commons Variations

Different CC licenses with varying restrictions.

```
Sources:
  - images_a: CC-BY-4.0
  - images_b: CC-BY-NC-4.0

Result:
  - result_license: "mixed"
  - conflicts: []
  - warnings: ["CC-BY-4.0 + CC-BY-NC-4.0: commercial use restricted"]
  - restrictions: [no-liability, no-warranty, non-commercial]

Conclusion: COMPATIBLE (with restrictions)
  No conflict, but the combined work inherits the NC restriction.
  The model cannot be used commercially if trained on NC data.
```

### Scenario 4: Different GPL Versions

Two versions of the same copyleft license family.

```
Sources:
  - code_v2: GPL-2.0-only
  - code_v3: GPL-3.0-only

Result:
  - result_license: "mixed"
  - conflicts: [
      {
        "license_a": "GPL-3.0-only",
        "license_b": "GPL-2.0-only",
        "reason": "incompatible copyleft licenses"
      }
    ]

Conclusion: CONFLICT
  GPL-2.0-only and GPL-3.0-only are not compatible.
  The "-only" suffix prevents automatic upgrading.
  This combination requires legal review.
```

### Scenario 5: Unknown License

Data with unspecified licensing.

```
Sources:
  - scraped_data: unknown
  - licensed_data: MIT

Result:
  - result_license: "mixed"
  - conflicts: []
  - warnings: ["unknown license present"]
  - restrictions: [no-liability, no-warranty, unknown-terms]

Conclusion: COMPATIBLE (with warnings)
  Unknown licenses are flagged but not treated as conflicts.
  The unknown-terms restriction is added.
  Manual review recommended before release.
```

## 8. Limitations

### What Origin Does NOT Determine

Origin is a technical tool, not a legal advisor. It does not:

| Limitation | Explanation |
|------------|-------------|
| Provide legal advice | Conflict detection is conservative, not definitive |
| Interpret license terms | Uses predefined rules, may miss nuances |
| Verify license accuracy | Trusts the license information provided |
| Handle license versions | "GPL-3.0-or-later" not distinguished from "GPL-3.0-only" |
| Track jurisdiction | Does not consider regional legal variations |
| Evaluate fair use | Cannot determine if use qualifies as fair use |

### When to Seek Legal Review

Consult legal counsel when:

- Origin detects license conflicts
- Unknown licenses are present in significant quantities
- Training data includes proprietary or restricted content
- Model will be used commercially
- Model will be distributed or open-sourced
- Regulatory compliance is required (e.g., EU AI Act)

### Conservative Approach

Origin's conflict detection is intentionally conservative:

- Flags potential issues for human review
- Does not automatically clear combinations
- Errs on the side of caution
- Treats unknown licenses as requiring review

This approach ensures that potential issues are surfaced rather than missed, but final determinations should involve appropriate legal expertise.
