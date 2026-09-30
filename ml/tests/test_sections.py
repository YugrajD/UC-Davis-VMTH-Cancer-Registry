"""report_mapping/sections.py: the single concat-3 spec and the report-text builders."""

from __future__ import annotations

import pandas as pd

from report_mapping import sections


def test_section_spec_shape():
    assert sections.CONCAT_3_SECTIONS == (
        ("HISTOPATHOLOGICAL SUMMARY",),
        ("FINAL COMMENT", "COMMENT"),
        ("ANCILLARY TESTS",),
    )
    assert sections.SECTION_COLUMNS == ("__sec_0__", "__sec_1__", "__sec_2__")


def test_build_section_frame_joins_multi_column_sections_with_newline():
    df = pd.DataFrame({
        "HISTOPATHOLOGICAL SUMMARY": ["hist a", ""],
        "FINAL COMMENT": ["final a", ""],
        "COMMENT": ["comment a", "comment b"],
        "ANCILLARY TESTS": ["", "anc b"],
    })
    out = sections.build_section_frame(df)
    assert out["__sec_0__"].tolist() == ["hist a", ""]
    assert out["__sec_1__"].tolist() == ["final a\ncomment a", "\ncomment b"]
    assert out["__sec_2__"].tolist() == ["", "anc b"]


def test_build_section_frame_missing_column_raises():
    df = pd.DataFrame({"HISTOPATHOLOGICAL SUMMARY": ["x"]})
    try:
        sections.build_section_frame(df)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "FINAL COMMENT" in str(exc)


def test_section_texts_cleans_whitespace_and_nan():
    df = pd.DataFrame({"__sec_0__": ["  hi  ", None], "__sec_1__": ["a", "b"], "__sec_2__": ["", ""]})
    texts = sections.section_texts(df)
    assert texts["__sec_0__"] == ["hi", ""]


def test_merge_report_columns_skips_empty_and_prefixes_column_name():
    df = sections.build_section_frame(pd.DataFrame({
        "HISTOPATHOLOGICAL SUMMARY": ["a mast cell tumor"],
        "FINAL COMMENT": ["margins clear"],
        "COMMENT": [""],
        "ANCILLARY TESTS": [""],
    }))
    merged = sections.merged_texts(df)
    assert merged == ["[__sec_0__] a mast cell tumor [__sec_1__] margins clear"]
