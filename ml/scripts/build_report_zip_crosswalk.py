#!/usr/bin/env python3
"""Build a report_36yr -> all_years ZIP crosswalk.

The two 36-year exports are not row-aligned, but their report text can be
normalized and hashed to create a high-confidence linkage. This script avoids
writing normalized/full report text; it only writes hashes and non-text
metadata needed for demo ingestion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


SECTION_LABEL_RE = re.compile(
    r"\[(HISTOPATHOLOGICAL SUMMARY|FINAL COMMENT|ANCILLARY TESTS)\]",
    re.IGNORECASE,
)
NON_ALNUM_RE = re.compile(r"[^A-Z0-9]+")
SPACE_RE = re.compile(r"\s+")


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).upper()
    text = SECTION_LABEL_RE.sub(" ", text)
    text = NON_ALNUM_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def normalize_scalar(value: object) -> str:
    if pd.isna(value):
        return ""
    return SPACE_RE.sub(" ", str(value).strip().upper())


def sha1_or_empty(value: str) -> str:
    if not value:
        return ""
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def clean_zip(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    digits = re.sub(r"\D", "", text)
    return digits[:5] if len(digits) >= 5 else digits


def add_join_fields(
    df: pd.DataFrame,
    *,
    text_col: str,
    date_col: str,
    sex_col: str,
    species_col: str,
    breed_col: str,
) -> pd.DataFrame:
    out = df.copy()
    out["_normalized_text_hash"] = out[text_col].map(normalize_text).map(sha1_or_empty)
    out["_date"] = pd.to_datetime(out[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    out["_sex"] = out[sex_col].map(normalize_scalar)
    out["_species"] = out[species_col].map(normalize_scalar)
    out["_breed"] = out[breed_col].map(normalize_scalar)
    out["_demographic_key"] = out[["_date", "_sex", "_species", "_breed"]].astype(str).agg("|".join, axis=1)
    out["_demographic_hash"] = out["_demographic_key"].map(sha1_or_empty)
    return out


def unique_hash_matches(all_years: pd.DataFrame, report: pd.DataFrame) -> pd.DataFrame:
    ay_counts = all_years["_normalized_text_hash"].value_counts()
    report_counts = report["_normalized_text_hash"].value_counts()
    unique_hashes = {
        h
        for h in set(ay_counts.index) & set(report_counts.index)
        if h and ay_counts[h] == 1 and report_counts[h] == 1
    }
    ay_unique = all_years[all_years["_normalized_text_hash"].isin(unique_hashes)]
    report_unique = report[report["_normalized_text_hash"].isin(unique_hashes)]
    return report_unique.merge(
        ay_unique,
        on="_normalized_text_hash",
        suffixes=("_report", "_all_years"),
        how="inner",
    ).assign(match_method="text_hash_unique")


def unique_demographic_fallback(
    all_years: pd.DataFrame,
    report: pd.DataFrame,
    matched_report_indexes: set[int],
    matched_all_years_indexes: set[int],
) -> pd.DataFrame:
    ay_remaining = all_years[~all_years["_all_years_row_index"].isin(matched_all_years_indexes)]
    report_remaining = report[~report["_report36yr_row_index"].isin(matched_report_indexes)]
    ay_counts = ay_remaining["_demographic_key"].value_counts()
    report_counts = report_remaining["_demographic_key"].value_counts()
    unique_keys = {
        k
        for k in set(ay_counts.index) & set(report_counts.index)
        if k and ay_counts[k] == 1 and report_counts[k] == 1
    }
    ay_unique = ay_remaining[ay_remaining["_demographic_key"].isin(unique_keys)]
    report_unique = report_remaining[report_remaining["_demographic_key"].isin(unique_keys)]
    return report_unique.merge(
        ay_unique,
        on="_demographic_key",
        suffixes=("_report", "_all_years"),
        how="inner",
    ).assign(match_method="demographic_fallback_unique")


def build_crosswalk(all_years_path: Path, report_path: Path) -> tuple[pd.DataFrame, dict]:
    all_years = pd.read_csv(
        all_years_path,
        usecols=[
            "DtOfRq",
            "Sex",
            "Species",
            "Breed",
            "Zipcode Zipcode",
            "RfrrVtrn Zipcode Zipcode",
            "Text",
        ],
    )
    report = pd.read_csv(
        report_path,
        usecols=["case_id", "year", "dt_of_rq", "sex", "species", "breed", "full_text"],
    )
    all_years["_all_years_row_index"] = all_years.index
    report["_report36yr_row_index"] = report.index

    all_years = add_join_fields(
        all_years,
        text_col="Text",
        date_col="DtOfRq",
        sex_col="Sex",
        species_col="Species",
        breed_col="Breed",
    )
    report = add_join_fields(
        report,
        text_col="full_text",
        date_col="dt_of_rq",
        sex_col="sex",
        species_col="species",
        breed_col="breed",
    )

    text_matches = unique_hash_matches(all_years, report)
    matched_report = set(text_matches["_report36yr_row_index"].astype(int))
    matched_all_years = set(text_matches["_all_years_row_index"].astype(int))
    demo_matches = unique_demographic_fallback(all_years, report, matched_report, matched_all_years)
    matches = pd.concat([text_matches, demo_matches], ignore_index=True, sort=False)

    matches["owner_zip"] = matches["Zipcode Zipcode"].map(clean_zip)
    matches["referring_vet_zip"] = matches["RfrrVtrn Zipcode Zipcode"].map(clean_zip)
    matches["chosen_zip"] = matches["owner_zip"].where(matches["owner_zip"].ne(""), matches["referring_vet_zip"])
    matches["zip_source"] = ""
    matches.loc[matches["owner_zip"].ne(""), "zip_source"] = "owner"
    matches.loc[matches["owner_zip"].eq("") & matches["referring_vet_zip"].ne(""), "zip_source"] = "referring_vet"

    crosswalk = pd.DataFrame(
        {
            "case_id": matches["case_id"],
            "report36yr_row_index": matches["_report36yr_row_index"].astype(int),
            "all_years_row_index": matches["_all_years_row_index"].astype(int),
            "match_method": matches["match_method"],
            "normalized_text_hash": matches["_normalized_text_hash"].fillna(""),
            "demographic_hash": matches["_demographic_hash_report"].fillna(matches.get("_demographic_hash", "")),
            "date_report36yr": matches["_date_report"],
            "date_all_years": matches["_date_all_years"],
            "year": matches["year"],
            "sex": matches["sex"],
            "species": matches["species"],
            "breed": matches["breed"],
            "owner_zip": matches["owner_zip"],
            "referring_vet_zip": matches["referring_vet_zip"],
            "chosen_zip": matches["chosen_zip"],
            "zip_source": matches["zip_source"],
        }
    ).sort_values(["report36yr_row_index", "match_method"])

    summary = {
        "all_years_rows": int(len(all_years)),
        "report36yr_rows": int(len(report)),
        "matched_rows": int(len(crosswalk)),
        "unmatched_report36yr_rows": int(len(report) - crosswalk["report36yr_row_index"].nunique()),
        "match_method_counts": {
            str(k): int(v) for k, v in crosswalk["match_method"].value_counts().items()
        },
        "rows_with_chosen_zip": int(crosswalk["chosen_zip"].ne("").sum()),
        "rows_with_owner_zip": int(crosswalk["owner_zip"].ne("").sum()),
        "rows_with_referring_vet_zip": int(crosswalk["referring_vet_zip"].ne("").sum()),
        "duplicate_report_matches": int(crosswalk["report36yr_row_index"].duplicated().sum()),
        "duplicate_all_years_matches": int(crosswalk["all_years_row_index"].duplicated().sum()),
    }
    return crosswalk, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all-years",
        type=Path,
        default=Path("/Users/yugrajdhillon/Downloads/all_years_1990_2025.csv"),
    )
    parser.add_argument("--report-36yr", type=Path, default=Path("ml/data/report_36yr.csv"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("ml/output/demo_crosswalk/report36yr_all_years_zip_crosswalk.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("ml/output/demo_crosswalk/report36yr_all_years_zip_crosswalk_summary.json"),
    )
    args = parser.parse_args()

    crosswalk, summary = build_crosswalk(args.all_years, args.report_36yr)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    crosswalk.to_csv(args.out, index=False)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.out}")
    print(f"Wrote {args.summary}")


if __name__ == "__main__":
    main()
