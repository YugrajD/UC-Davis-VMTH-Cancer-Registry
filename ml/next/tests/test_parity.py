"""Parity harness levels (L1–L4) on synthetic tables: an equal pair passes, one planted
difference fails. Deleted at cutover with parity/ and scripts/parity.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from parity import levels
from parity.reference import load_embeddings

from . import fixtures as fx

GROUP = "Mast Cell Tumors"
CODE, TERM = fx.TAXONOMY_GROUPS[GROUP][0]


def verdict_table(good=0, slightly_off=0, completely_off=0, false_positive=0, false_negative=0,
                  group: str = GROUP) -> pd.DataFrame:
    """A legacy-shaped verdict table with the given verdict counts, one row per case."""
    counts = dict(good=good, slightly_off=slightly_off, completely_off=completely_off,
                  false_positive=false_positive, false_negative=false_negative)
    rows = []
    for verdict, n in counts.items():
        for _ in range(n):
            expected = "" if verdict == "false_positive" else group
            rows.append({
                "case_id": f"CASE-{len(rows) + 1:05d}", "diagnosis_index": "1",
                "predicted_term": TERM, "expected_term": TERM if expected else "",
                "predicted_group": group, "expected_group": expected,
                "case_presence_prob": "0.9000", "confidence": "0.50", "group_prob": "0.8000",
                "verdict": verdict,
            })
    return pd.DataFrame(rows)


def predictions(n: int = 1000) -> pd.DataFrame:
    """A predictions CSV as read_table returns it (all strings)."""
    return pd.DataFrame([{
        "case_id": f"CASE-{i:05d}", "diagnosis_index": "1",
        "predicted_term": TERM, "predicted_group": GROUP, "predicted_code": CODE,
        "case_presence_prob": "0.9000", "confidence": "0.50", "group_prob": "0.8000",
        "method": "label_presence",
    } for i in range(n)])


# ---------------------------------------------------------------------------
# L1
# ---------------------------------------------------------------------------


def test_l1_identical_tables_pass_regardless_of_row_order() -> None:
    ref = verdict_table(good=5, slightly_off=3, false_negative=2)
    new = ref.sample(frac=1, random_state=0).assign(predicted_code=CODE, method="label_presence")
    assert levels.l1(new, ref).passed is True


def test_l1_one_changed_verdict_fails() -> None:
    ref = verdict_table(good=5, slightly_off=3, false_negative=2)
    new = ref.copy()
    new.loc[0, "verdict"] = "completely_off"
    assert levels.l1(new, ref).passed is False


def test_l1_missing_legacy_column_fails() -> None:
    ref = verdict_table(good=3)
    assert levels.l1(ref.drop(columns="expected_group"), ref).passed is False


# ---------------------------------------------------------------------------
# L2
# ---------------------------------------------------------------------------


def test_l2_identical_predictions_pass_with_extra_columns() -> None:
    ref = predictions(20)
    assert levels.l2(ref.assign(generation_id="gen-0"), ref).passed is True


def test_l2_term_diff_fails() -> None:
    ref = predictions(20)
    new = ref.copy()
    new.loc[3, "predicted_term"] = fx.TAXONOMY_GROUPS[GROUP][1][1]
    assert levels.l2(new, ref).passed is False


@pytest.mark.parametrize("delta, passed", [(5e-6, True), (2e-5, False)])
def test_l2_probability_tolerance(delta: float, passed: bool) -> None:
    ref = predictions(20)
    new = ref.copy()
    new.loc[3, "group_prob"] = str(0.8 + delta)
    assert levels.l2(new, ref).passed is passed


def test_l2_missing_row_fails() -> None:
    ref = predictions(20)
    assert levels.l2(ref.iloc[1:], ref).passed is False


def test_l2_duplicate_key_fails() -> None:
    ref = predictions(20)
    assert levels.l2(pd.concat([ref, ref.iloc[[3]]], ignore_index=True), ref).passed is False


UNCOMMON = frozenset({"Rare Sarcomas", "Rare Carcinomas"})


def two_label_case(groups: tuple[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(ref, new): CASE-00000 has two rows with different labels; new swaps them
    between diagnosis_index 1 and 2 (probabilities travel with their label)."""
    ref = predictions(20)
    ref = pd.concat([ref, ref.iloc[[0]].assign(diagnosis_index="2")], ignore_index=True)
    labels = [(code, term, group) for group in groups for code, term in fx.TAXONOMY_GROUPS[group][:1]]
    if labels[0] == labels[1]:  # same group: take its second label
        code, term = fx.TAXONOMY_GROUPS[groups[1]][1]
        labels[1] = (code, term, groups[1])
    for row, ((code, term, group), group_prob, confidence) in zip([0, 20], zip(labels, ["0.9100", "0.0400"],
                                                                                ["0.70", "0.20"])):
        ref.loc[row, ["predicted_code", "predicted_term", "predicted_group", "group_prob", "confidence"]] = [
            code, term, group, group_prob, confidence]
    new = ref.copy()
    swap = ["predicted_code", "predicted_term", "predicted_group", "group_prob", "confidence"]
    new.loc[[0, 20], swap] = ref.loc[[20, 0], swap].to_numpy()
    return new, ref


def test_l2_uncommon_reordered_case_passes_and_is_counted() -> None:
    new, ref = two_label_case(("Rare Sarcomas", "Rare Carcinomas"))
    report = levels.l2(new, ref, UNCOMMON)
    assert report.passed is True
    assert any("uncommon-reordered 1 " in line for line in report.lines)


def test_l2_reordered_non_uncommon_case_fails() -> None:
    new, ref = two_label_case((GROUP, GROUP))
    assert levels.l2(new, ref, UNCOMMON).passed is False


def test_l2_uncommon_case_with_changed_confidence_fails() -> None:
    new, ref = two_label_case(("Rare Sarcomas", "Rare Carcinomas"))
    new.loc[0, "confidence"] = "0.21"
    assert levels.l2(new, ref, UNCOMMON).passed is False


# ---------------------------------------------------------------------------
# L2 re-embedded
# ---------------------------------------------------------------------------


def _embeddings(n: int = 1000, dim: int = 8):
    rng = np.random.default_rng(0)
    ids = np.array([f"CASE-{i:05d}" for i in range(n)])
    return ids, rng.normal(size=(n, dim)).astype(np.float32)


def _reembedded(new_emb=None, new_preds=None, new_table=None, ref_preds=None, uncommon=frozenset()):
    ref_emb = _embeddings()
    ref_preds = predictions() if ref_preds is None else ref_preds
    ref_table = verdict_table(good=400, slightly_off=200, completely_off=150, false_negative=250)
    return levels.l2_reembedded(
        new_emb or ref_emb, ref_emb,
        ref_preds if new_preds is None else new_preds, ref_preds,
        ref_table if new_table is None else new_table, ref_table, uncommon,
    )


def test_l2_reembedded_equal_passes_with_shuffled_embedding_rows() -> None:
    ids, mat = _embeddings()
    order = np.random.default_rng(1).permutation(len(ids))
    assert _reembedded(new_emb=(ids[order], mat[order])).passed is True


def test_l2_reembedded_one_low_cosine_fails() -> None:
    ids, mat = _embeddings()
    mat = mat.copy()
    mat[7] = -mat[7]
    assert _reembedded(new_emb=(ids, mat)).passed is False


@pytest.mark.parametrize("n_changed, passed", [(4, True), (6, False)])
def test_l2_reembedded_identical_row_share(n_changed: int, passed: bool) -> None:
    new = predictions()
    new.loc[: n_changed - 1, "predicted_code"] = fx.TAXONOMY_GROUPS[GROUP][1][0]
    assert _reembedded(new_preds=new).passed is passed


def test_l2_reembedded_duplicate_key_fails() -> None:
    ref = predictions()
    assert _reembedded(new_preds=pd.concat([ref, ref.iloc[[3]]], ignore_index=True)).passed is False


def test_l2_reembedded_uncommon_reordered_rows_count_as_identical() -> None:
    new, ref = two_label_case(("Rare Sarcomas", "Rare Carcinomas"))
    ref = pd.concat([ref, predictions().iloc[20:]], ignore_index=True)  # back to 1,000 cases
    new = pd.concat([new, predictions().iloc[20:]], ignore_index=True)
    report = _reembedded(new_preds=new, ref_preds=ref, uncommon=UNCOMMON)
    assert report.passed is True
    assert any("1001/1001 = 100.00%" in line for line in report.lines)


@pytest.mark.parametrize("moved, passed", [(1, True), (3, False)])
def test_l2_reembedded_gs_shift(moved: int, passed: bool) -> None:
    # 1000 rows at 60.0% G+S; each moved FN→good row adds 0.1 pp.
    new = verdict_table(good=400 + moved, slightly_off=200, completely_off=150, false_negative=250 - moved)
    assert _reembedded(new_table=new).passed is passed


# ---------------------------------------------------------------------------
# L3 / L4
# ---------------------------------------------------------------------------

# 200 rows, G+S 60.0%.
REF = dict(good=80, slightly_off=40, completely_off=30, false_positive=10, false_negative=40)


def test_l3_identical_seeds_pass() -> None:
    assert levels.l3([verdict_table(**REF)] * 3, verdict_table(**REF)).passed is True


def test_l3_gs_shift_beyond_floor_fails_without_seed_spread() -> None:
    # Every seed at 58.5% (−1.5 pp), sd 0 → tolerance 1.0 pp.
    seed = verdict_table(good=78, slightly_off=39, completely_off=32, false_positive=10, false_negative=41)
    assert levels.l3([seed] * 3, verdict_table(**REF)).passed is False


def test_l3_seed_spread_widens_tolerance() -> None:
    # Seeds at 57.5 / 58.5 / 59.5% → mean −1.5 pp, sd 1.0 → tolerance 2.0 pp.
    seeds = [
        verdict_table(good=77, slightly_off=38, completely_off=33, false_positive=10, false_negative=42),
        verdict_table(good=78, slightly_off=39, completely_off=32, false_positive=10, false_negative=41),
        verdict_table(good=80, slightly_off=39, completely_off=31, false_positive=10, false_negative=40),
    ]
    report = levels.l3(seeds, verdict_table(**REF))
    assert report.passed is True
    assert any("tolerance ±2.00" in line for line in report.lines)


def test_l3_verdict_share_out_of_tolerance_fails_even_when_gs_holds() -> None:
    # good −2 pp, slightly_off +2 pp: G+S unchanged.
    seed = verdict_table(good=76, slightly_off=44, completely_off=30, false_positive=10, false_negative=40)
    assert levels.l3([seed] * 3, verdict_table(**REF)).passed is False


def test_l3_lists_groups_losing_more_than_5pp_without_failing() -> None:
    # 4 of the other group's 60 rows go good → CO: −6.7 pp for the group, −0.9 pp overall.
    other = fx.GROUP_NAMES[2]
    base = verdict_table(**{k: 2 * v for k, v in REF.items()})
    ref = pd.concat([base, verdict_table(good=40, completely_off=20, group=other)])
    seed = pd.concat([base, verdict_table(good=36, completely_off=24, group=other)])
    report = levels.l3([seed] * 3, ref)
    assert report.passed is True
    assert any(line.strip().startswith(f"{other}:") for line in report.lines)
    assert not any(line.strip().startswith(f"{GROUP}:") for line in report.lines)


def test_l4_is_report_only() -> None:
    seed = verdict_table(good=40, slightly_off=20, completely_off=70, false_positive=30, false_negative=40)
    report = levels.l3([seed], verdict_table(**REF), gate=False, name="L4")
    assert report.passed is None
    assert report.lines[-1].startswith("L4 REPORT")


# ---------------------------------------------------------------------------
# Embedding loader
# ---------------------------------------------------------------------------


def test_load_embeddings_reads_legacy_npz_layout(tmp_path: Path) -> None:
    ids, mat = _embeddings(5)
    path = tmp_path / "cache.npz"
    np.savez(path, case_ids=ids.astype(object), col_concat_3=mat, mean_embeddings=mat[:, :4],
             label_texts=np.array(["synthetic"], dtype=object))
    got_ids, got = load_embeddings(path)
    assert list(got_ids) == list(ids)
    np.testing.assert_array_equal(got, mat)


def test_load_embeddings_npy_needs_ids_file(tmp_path: Path) -> None:
    ids, mat = _embeddings(5)
    np.save(tmp_path / "emb.npy", mat)
    (tmp_path / "ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    got_ids, got = load_embeddings(tmp_path / "emb.npy", ids_path=tmp_path / "ids.txt")
    assert list(got_ids) == list(ids)
    np.testing.assert_array_equal(got, mat)
    with pytest.raises(ValueError):
        load_embeddings(tmp_path / "emb.npy")


def test_load_embeddings_reads_the_new_content_hash_cache_format(tmp_path: Path) -> None:
    # An L2b re-embed writes report_mapping.inference.embedding_cache's own
    # format (report_mapping/inference/embedding_cache.py::save), not a
    # hand-built legacy-shaped npz. Its "concat_3" column becomes the array
    # key "col_concat_3" (npz_col_key("concat_3") == "concat_3") and its ids
    # array is "case_ids" -- both exactly what load_embeddings' key=None
    # default already looks for, so `parity.py l2 --reembedded --embeddings
    # <that .npz>` needs no extra --embeddings-key/--embeddings-ids flags.
    from report_mapping.inference.embedding_cache import EmbeddingCache
    from report_mapping.inference.embedding_cache import save as save_cache

    ids, mat = _embeddings(5, dim=8)
    cache = EmbeddingCache(
        case_ids=list(ids),
        col_embeddings={"__sec_0__": mat[:, :4], "concat_3": mat},
        col_has_content={"__sec_0__": np.ones(5, dtype=bool)},
        label_texts=["synthetic label"],
        label_embeddings=np.zeros((1, 4), dtype=np.float32),
    )
    path = tmp_path / "reembedded.npz"
    save_cache("reembedded", cache, tmp_path)
    assert path.is_file()

    got_ids, got = load_embeddings(path)
    assert list(got_ids) == list(ids)
    np.testing.assert_array_equal(got, mat)
