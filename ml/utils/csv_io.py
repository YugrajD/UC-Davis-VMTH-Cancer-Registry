"""CSV reading helpers shared across pipelines.

The report CSVs are exported as latin-1 with a UTF-8 BOM on the first column
header (``﻿`` or its mojibake form ``ï»¿``). Strip that prefix before any
column lookup so callers can use the documented column names verbatim.
"""

from __future__ import annotations

from typing import Iterable


def strip_bom_from_columns(columns: Iterable[str]) -> list[str]:
    return [c.lstrip("﻿").lstrip("ï»¿") for c in columns]
