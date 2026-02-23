"""Zip code to California county mapping using the zipcodes library."""

from typing import Optional

import zipcodes


def lookup_county(zip_code: str) -> Optional[str]:
    """Look up the California county name for a US ZIP code.

    Returns the county name (without " County" suffix) if the zip is in
    California, otherwise None.
    """
    normalized = zip_code.strip()[:5]
    if not normalized or not normalized.isdigit():
        return None

    results = zipcodes.matching(normalized)
    if not results:
        return None

    entry = results[0]
    if entry.get("state") != "CA":
        return None

    county = entry.get("county", "")
    if county.endswith(" County"):
        county = county[:-7]
    return county or None
