"""Demographics feature encoder for Stage 1 (CasePresenceClassifier) and Stage 2 (GroupClassifier).

Produces a fixed-width numeric block from demographics.csv keyed by case_id.
Bypasses PetBERT entirely — no re-embed, no backbone change.

Column order (deterministic):
  [0]   age_years_zscore          — (DtOfRq - DateOfBirth) in years, z-scored on train
  [1]   age_missing               — 1.0 if DOB or DtOfRq was unparseable
  [2..] sex_<cat>                 — one-hot over train-observed Sex values + "Missing"
  [..]  species_<cat>             — one-hot over train-observed Species values + "Missing"
  [..]  breed_<top30>             — one-hot for top-30 train breeds + "Other" + "Missing"
  [-2]  zipcode_present           — 1.0 if Zipcode is non-empty
  [-1]  rfrrVtrnZipcode_present   — 1.0 if RfrrVtrnZipcode is non-empty

Cases absent from demographics.csv → all-missing row (age_missing=1, one-hot → Missing/Other bucket).
"""

from __future__ import annotations

import datetime
import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

# Excel epoch: serial day 1 = 1900-01-01 in Excel's (buggy) system;
# day 0 = 1899-12-30 in Python terms.
_EXCEL_EPOCH = datetime.date(1899, 12, 30)
_YEAR_RE = re.compile(r"^\d{4}$")
_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")
_SERIAL_RE = re.compile(r"^\d{5}$")

TOP_N_BREEDS = 30  # number of breed categories (plus "Other" + "Missing")


def _parse_dob(dob: str) -> datetime.date | None:
    """Parse DateOfBirth to a date.  Returns None on failure.

    Supported formats:
      - Bare 4-digit year ("1984") → January 1 of that year
      - ISO date ("2021-08-01")
      - 5-digit Excel serial ("32843") → decoded via Excel epoch
    """
    dob = dob.strip()
    if not dob:
        return None
    if _YEAR_RE.match(dob):
        try:
            return datetime.date(int(dob), 1, 1)
        except ValueError:
            return None
    if _ISO_RE.match(dob):
        try:
            return datetime.date.fromisoformat(dob[:10])
        except ValueError:
            return None
    if _SERIAL_RE.match(dob):
        try:
            return _EXCEL_EPOCH + datetime.timedelta(days=int(dob))
        except (ValueError, OverflowError):
            return None
    return None


def _parse_iso(s: str) -> datetime.date | None:
    s = s.strip()
    if not s:
        return None
    try:
        return datetime.date.fromisoformat(s[:10])
    except ValueError:
        return None


def _age_years(dob_str: str, dtofrq_str: str) -> tuple[float | None, bool]:
    """Return (age_in_years, missing_flag)."""
    dob = _parse_dob(dob_str)
    req = _parse_iso(dtofrq_str)
    if dob is None or req is None:
        return None, True
    delta_days = (req - dob).days
    return delta_days / 365.25, False


class DemographicsEncoder:
    """Fit on train case_ids, transform any set of case_ids to a float32 block."""

    def __init__(self) -> None:
        # Set by fit()
        self._df: pd.DataFrame | None = None          # full demographics table
        self._age_mean: float = 0.0
        self._age_std: float = 1.0
        self._sex_cats: list[str] = []
        self._species_cats: list[str] = []
        self._breed_cats: list[str] = []              # top-N + ["Other"]
        self.dim: int = 0

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, demo_csv: str, train_case_ids: Sequence[str]) -> "DemographicsEncoder":
        """Fit on train_case_ids only.  demographics.csv is loaded once and kept
        in memory for subsequent transform() calls."""
        df = pd.read_csv(demo_csv, encoding="utf-8-sig", dtype=str).fillna("")
        df["case_id"] = df["case_id"].astype(str).str.strip()
        df = df.set_index("case_id")
        self._df = df

        train_set = set(str(c) for c in train_case_ids)
        train_rows = df[df.index.isin(train_set)]

        # Age: compute on train rows
        ages = []
        for cid, row in train_rows.iterrows():
            age, missing = _age_years(row.get("DateOfBirth", ""), row.get("DtOfRq", ""))
            if not missing and age is not None:
                ages.append(age)
        if ages:
            self._age_mean = float(np.mean(ages))
            self._age_std = float(np.std(ages)) or 1.0
        else:
            self._age_mean = 0.0
            self._age_std = 1.0

        # Sex categories (sorted, stable)
        self._sex_cats = sorted(
            v for v in train_rows["Sex"].str.strip().unique() if v
        ) + ["Missing"]

        # Species categories
        self._species_cats = sorted(
            v for v in train_rows["Species"].str.strip().unique() if v
        ) + ["Missing"]

        # Breed: top-N by train frequency, then "Other", "Missing"
        breed_counts = train_rows["Breed"].str.strip().value_counts()
        top_breeds = [b for b in breed_counts.head(TOP_N_BREEDS).index if b]
        self._breed_cats = top_breeds + ["Other", "Missing"]

        self.dim = self._compute_dim()
        return self

    def _compute_dim(self) -> int:
        return (
            2                           # age_zscore + age_missing
            + len(self._sex_cats)       # one-hot sex (includes "Missing")
            + len(self._species_cats)   # one-hot species (includes "Missing")
            + len(self._breed_cats)     # one-hot breed (includes "Other" + "Missing")
            + 2                         # zipcode_present + rfrrVtrnZipcode_present
        )

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, case_ids: Sequence[str]) -> np.ndarray:
        """Return (N, D) float32 block.  Unknown case_ids → all-missing row."""
        N = len(case_ids)
        out = np.zeros((N, self.dim), dtype=np.float32)

        sex_idx = {c: i for i, c in enumerate(self._sex_cats)}
        spc_idx = {c: i for i, c in enumerate(self._species_cats)}
        breed_idx = {c: i for i, c in enumerate(self._breed_cats)}

        # Column offsets
        age_z_col = 0
        age_miss_col = 1
        sex_off = 2
        spc_off = sex_off + len(self._sex_cats)
        breed_off = spc_off + len(self._species_cats)
        zip_col = breed_off + len(self._breed_cats)
        rzip_col = zip_col + 1

        for i, cid in enumerate(case_ids):
            cid_str = str(cid).strip()
            if self._df is None or cid_str not in self._df.index:
                # Entire row stays 0; set missing buckets
                out[i, age_miss_col] = 1.0
                out[i, sex_off + sex_idx["Missing"]] = 1.0
                out[i, spc_off + spc_idx["Missing"]] = 1.0
                out[i, breed_off + breed_idx["Missing"]] = 1.0
                continue

            row = self._df.loc[cid_str]

            # Age
            age, missing = _age_years(row.get("DateOfBirth", ""), row.get("DtOfRq", ""))
            if missing or age is None:
                out[i, age_miss_col] = 1.0
                # age_z_col stays 0 (mean imputation)
            else:
                out[i, age_z_col] = (age - self._age_mean) / self._age_std
                out[i, age_miss_col] = 0.0

            # Sex one-hot
            sex = row.get("Sex", "").strip()
            if sex in sex_idx:
                out[i, sex_off + sex_idx[sex]] = 1.0
            else:
                out[i, sex_off + sex_idx["Missing"]] = 1.0

            # Species one-hot
            spc = row.get("Species", "").strip()
            if spc in spc_idx:
                out[i, spc_off + spc_idx[spc]] = 1.0
            else:
                out[i, spc_off + spc_idx["Missing"]] = 1.0

            # Breed one-hot
            breed = row.get("Breed", "").strip()
            if breed in breed_idx:
                out[i, breed_off + breed_idx[breed]] = 1.0
            elif "Other" in breed_idx:
                out[i, breed_off + breed_idx["Other"]] = 1.0

            # Zipcode flags
            out[i, zip_col] = 1.0 if row.get("Zipcode", "").strip() else 0.0
            out[i, rzip_col] = 1.0 if row.get("RfrrVtrnZipcode", "").strip() else 0.0

        return out

    # ------------------------------------------------------------------
    # Persist / reload
    # ------------------------------------------------------------------

    def save_spec(self, path: str) -> None:
        """Save the fitted spec (categories + age stats) to JSON.  No row data."""
        spec = {
            "age_mean": self._age_mean,
            "age_std": self._age_std,
            "sex_cats": self._sex_cats,
            "species_cats": self._species_cats,
            "breed_cats": self._breed_cats,
            "dim": self.dim,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(spec, indent=2), encoding="utf-8")

    def load_spec(self, path: str, demo_csv: str) -> "DemographicsEncoder":
        """Reload a previously fitted spec from JSON and load the demographics table."""
        spec = json.loads(Path(path).read_text(encoding="utf-8"))
        self._age_mean = float(spec["age_mean"])
        self._age_std = float(spec["age_std"])
        self._sex_cats = list(spec["sex_cats"])
        self._species_cats = list(spec["species_cats"])
        self._breed_cats = list(spec["breed_cats"])
        self.dim = int(spec["dim"])
        # Load demographics table for transform()
        df = pd.read_csv(demo_csv, encoding="utf-8-sig", dtype=str).fillna("")
        df["case_id"] = df["case_id"].astype(str).str.strip()
        self._df = df.set_index("case_id")
        return self
