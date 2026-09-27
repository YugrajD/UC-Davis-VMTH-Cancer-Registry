"""The one shared CSV reader/writer.

Raw inputs (report.csv, diagnoses.csv) are latin-1; everything we write is
utf-8, so reading one of our own outputs passes ``encoding="utf-8"``.

Reports are exported as latin-1 with a UTF-8 BOM on the first column header
(``﻿`` or its mojibake form ``ï»¿``). Strip that prefix before any
column lookup so callers can use the documented column names verbatim. Our
own outputs are always utf-8.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def strip_bom_from_columns(columns: Iterable[str]) -> list[str]:
    return [c.lstrip("﻿").lstrip("ï»¿") for c in columns]


def read_csv(path: str | Path, *, encoding: str = "latin-1", **kwargs) -> pd.DataFrame:
    """Read a CSV with a possibly BOM-prefixed header. Default latin-1 (report/diagnoses);
    pass ``encoding="utf-8"`` for our own outputs."""
    df = pd.read_csv(path, encoding=encoding, **kwargs)
    df.columns = strip_bom_from_columns(df.columns)
    return df


def write_csv(df: pd.DataFrame, path: str | Path, **kwargs) -> None:
    """Write a pipeline output CSV: utf-8, no index, LF line endings on every OS."""
    df.to_csv(path, index=False, encoding="utf-8", lineterminator="\n", **kwargs)
