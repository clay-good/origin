"""
Test fixtures for license-related tests.

Provides pre-configured LicenseRecord instances for common licenses.
"""

from origin.core.record import LicenseRecord


# MIT License record for testing
TEST_MIT_LICENSE = LicenseRecord(
    license_id="MIT",
    license_name="MIT License",
    license_url="https://opensource.org/licenses/MIT",
    permissions=("commercial", "distribute", "modify", "private-use"),
    restrictions=("no-liability", "no-warranty"),
    conditions=("attribution", "license-notice"),
    copyleft=False,
)

# GPL-3.0 License record for testing
TEST_GPL_LICENSE = LicenseRecord(
    license_id="GPL-3.0",
    license_name="GNU General Public License v3.0",
    license_url="https://www.gnu.org/licenses/gpl-3.0.html",
    permissions=("commercial", "distribute", "modify", "patent-use", "private-use"),
    restrictions=("no-liability", "no-warranty"),
    conditions=("attribution", "disclose-source", "license-notice", "same-license", "state-changes"),
    copyleft=True,
)

# Apache-2.0 License record for testing
TEST_APACHE_LICENSE = LicenseRecord(
    license_id="Apache-2.0",
    license_name="Apache License 2.0",
    license_url="https://www.apache.org/licenses/LICENSE-2.0",
    permissions=("commercial", "distribute", "modify", "patent-use", "private-use"),
    restrictions=("no-liability", "no-warranty", "no-trademark"),
    conditions=("attribution", "license-notice", "state-changes"),
    copyleft=False,
)

# Unknown license record for testing edge cases
TEST_UNKNOWN_LICENSE = LicenseRecord(
    license_id="LicenseRef-Unknown",
    license_name="Unknown License",
    license_url=None,
    permissions=(),
    restrictions=(),
    conditions=(),
    copyleft=False,
)

# CC-BY-4.0 License for testing creative commons
TEST_CC_BY_LICENSE = LicenseRecord(
    license_id="CC-BY-4.0",
    license_name="Creative Commons Attribution 4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    permissions=("commercial", "distribute", "modify", "private-use"),
    restrictions=("no-warranty",),
    conditions=("attribution", "license-notice"),
    copyleft=False,
)

# CC-BY-NC-4.0 License (non-commercial) for conflict testing
TEST_CC_BY_NC_LICENSE = LicenseRecord(
    license_id="CC-BY-NC-4.0",
    license_name="Creative Commons Attribution-NonCommercial 4.0",
    license_url="https://creativecommons.org/licenses/by-nc/4.0/",
    permissions=("distribute", "modify", "private-use"),
    restrictions=("no-commercial", "no-warranty"),
    conditions=("attribution", "license-notice"),
    copyleft=False,
)
