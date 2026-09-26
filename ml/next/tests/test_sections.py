"""report_mapping/sections.py: the single concat-3 spec, checked against both
legacy copies, and the report-text builders.

LEGACY COMPARISON (the module-level constant checks only) — deleted at
cutover, when ``production/petbert_pipeline/pipeline.py`` and
``training/contrastive/build_contrastive_dataset.py`` are removed.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

from report_mapping import sections

_ML_DIR = Path(__file__).resolve().parents[2]


def _literal_module_constant(path: Path, name: str):
    """The literal value of a top-level ``name = ...`` (or annotated) assignment,
    read via ast — never executed. pipeline.py uses package-relative imports
    that can't be run standalone, so this is the only safe way to read its
    ``CONCAT_3_SECTIONS`` without dragging in the rest of the old production tree.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name!r} not found in {path}")


def _load_legacy_pipeline_sections() -> tuple[tuple[str, ...], ...]:
    path = _ML_DIR / "production" / "petbert_pipeline" / "pipeline.py"
    return _literal_module_constant(path, "CONCAT_3_SECTIONS")


def _load_legacy_contrastive_sections() -> tuple[tuple[str, ...], ...]:
    path = _ML_DIR / "training" / "contrastive" / "build_contrastive_dataset.py"
    return _literal_module_constant(path, "_PER_SECTION_GROUPS")


def test_the_two_legacy_copies_agree_with_each_other():
    # pipeline.py:54-58 vs build_contrastive_dataset.py:21-25 — the plan asks
    # this WP to check for drift between the two pre-rewrite duplicates.
    assert _load_legacy_pipeline_sections() == _load_legacy_contrastive_sections()


def test_new_spec_matches_both_legacy_copies():
    assert sections.CONCAT_3_SECTIONS == _load_legacy_pipeline_sections()
    assert sections.CONCAT_3_SECTIONS == _load_legacy_contrastive_sections()


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
