"""All stages -> predictions with the legacy columns plus ``generation_id``.

Legacy columns (``production/petbert_pipeline/io.py::write_predictions_csv``):
``case_id, diagnosis_index, predicted_term, predicted_group, predicted_code,
case_presence_prob, confidence, group_prob, method``. Drops the similarity,
visualization, provenance, neighbors and embeddings-npz debug outputs
(Decisions: the similarity/visualization outputs are explicitly dropped; the
rest have no destination in the plan's Old -> new table and this deliverable's
scope is "predictions with the legacy columns plus generation_id").
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config
import io_utils
from report_mapping import sections
from report_mapping.inference import embedding_cache as embedding_cache_mod
from report_mapping.inference import stages
from report_mapping.model import backbone as backbone_mod
from report_mapping.model import generation as generation_mod
from taxonomy.taxonomy import build_taxonomy_label_texts, resolve_taxonomy_matches

# Methods whose CSV row never carries a code (mirrors pipeline.py's _NO_CODE_METHODS).
_NO_CODE_METHODS = frozenset({"empty", "low_confidence", "unidentified_cancer"})


def _read_reports() -> tuple[pd.DataFrame, list[str]]:
    dataframe = io_utils.read_csv(config.REPORT_CSV, encoding="latin-1")
    if "case_id" not in dataframe.columns:
        raise ValueError(f"Missing id column 'case_id'. Available: {dataframe.columns.tolist()}")
    ids = dataframe["case_id"].map(sections.clean_text).tolist()
    return dataframe, ids


def build_fresh_cache(
    model_dir_or_name: str, taxonomy_labels, *, local_only: bool, device, dataframe: pd.DataFrame,
    ids: list[str],
) -> embedding_cache_mod.EmbeddingCache:
    """Embed report sections + taxonomy label texts into a fresh ``EmbeddingCache``.

    The one place this embedding loop lives: shared by this module's own
    cache-miss path (``run_predict``) and ``report_mapping.training.embeddings.
    get_or_build`` (heads-only training's cache-miss path), so the two never
    drift apart. ``dataframe`` must already have the section columns
    (``sections.build_section_frame``).
    """
    backbone = backbone_mod.load_backbone(model_dir_or_name, local_only=local_only)
    texts = sections.section_texts(dataframe)
    col_embeddings: dict[str, np.ndarray] = {}
    col_has_content: dict[str, np.ndarray] = {}
    for col in sections.SECTION_COLUMNS:
        col_embeddings[col] = backbone_mod.embed_texts(backbone, texts[col], device=device)
        col_has_content[col] = np.array([bool(t) for t in texts[col]], dtype=bool)
    col_embeddings[sections.CONCAT_3_KEY] = np.concatenate(
        [col_embeddings[c] for c in sections.SECTION_COLUMNS], axis=1
    ).astype(np.float32)

    label_texts = build_taxonomy_label_texts(taxonomy_labels)
    label_embeddings = backbone_mod.embed_texts(backbone, label_texts, device=device)
    return embedding_cache_mod.EmbeddingCache(
        case_ids=ids, col_embeddings=col_embeddings, col_has_content=col_has_content,
        label_texts=label_texts, label_embeddings=label_embeddings,
    )


def group_classifier_input(cache: embedding_cache_mod.EmbeddingCache, sel: list[int]) -> np.ndarray:
    """Concatenation of the 3 section embeddings, each zeroed where that section
    had no content — the GroupClassifier's own input view (distinct from the
    unmasked concat-3 the gate + Stage 3a use). Mirrors pipeline.py's ``col_emb_concat``."""
    return np.concatenate(
        [
            np.where(cache.col_has_content[col][sel][:, None], cache.col_embeddings[col][sel], 0.0)
            for col in sections.SECTION_COLUMNS
        ],
        axis=1,
    ).astype(np.float32)


def prediction_rows(ids: list[str], result: stages.CategorizationResult, taxonomy_labels, labels: list[str],
                    case_presence_probs: np.ndarray, generation_id: str) -> list[dict]:
    """One dict per written prediction row (legacy columns + ``generation_id``), in output order."""
    rows = []
    for i, case_id in enumerate(ids):
        cp_prob = case_presence_probs[i]
        terms, groups, codes = resolve_taxonomy_matches(result.top_k_indices[i], labels, taxonomy_labels)
        codes = [c if m not in _NO_CODE_METHODS else "" for c, m in zip(codes, result.top_k_methods[i])]
        for rank, (term, group, code, score, grp_prob, method) in enumerate(
            zip(terms, groups, codes, result.top_k_scores[i], result.top_k_group_probs[i],
                result.top_k_methods[i]),
            start=1,
        ):
            if method == "low_confidence":
                term, group, method = "Non-Cancer", "Non-Cancer", "rejected_by_case_presence"
            elif method == "unidentified_cancer":
                term, group = "Unidentified Group", "Unidentified Group"
            rows.append({
                "case_id": case_id,
                "diagnosis_index": rank,
                "predicted_term": term,
                "predicted_group": group,
                "predicted_code": code,
                "case_presence_prob": "" if np.isnan(cp_prob) else f"{cp_prob:.4f}",
                "confidence": f"{score:.2f}",
                "group_prob": f"{grp_prob:.4f}",
                "method": method,
                "generation_id": generation_id,
            })
    return rows


def _cache_key(gen, model_dir_or_name: str) -> str:
    # Same fingerprint function report_mapping.training.embeddings.get_or_build calls, so the two
    # code paths compute the same key for the same (report, labels, backbone) — see test_embedding_cache.py.
    fingerprint = generation_mod.compute_embedding_fingerprint(model_dir_or_name)
    return embedding_cache_mod.content_key(config.REPORT_CSV, generation_mod.generation_paths(gen.root).labels_csv,
                                           fingerprint)


def load_cached_embeddings(gen) -> embedding_cache_mod.EmbeddingCache:
    """The generation's cached report embeddings (its own petbert/, the key run_predict uses).
    Raises FileNotFoundError on a miss; never embeds."""
    key = _cache_key(gen, str(gen.petbert_dir))
    cache = embedding_cache_mod.load(key)
    if cache is None:
        raise FileNotFoundError(f"no embedding cache for {gen.root} (key {key}); "
                                "run scripts/predict.py --embed-only on this generation first")
    return cache


def classify(gen, cache: embedding_cache_mod.EmbeddingCache, dataframe: pd.DataFrame, ids: list[str]) -> list[dict]:
    """Every stage on already-embedded reports: one dict per prediction row (``prediction_rows``).
    ``dataframe`` must have the section columns (``sections.build_section_frame``)."""
    if cache.case_ids == ids:
        # Embedded for exactly these rows (predict_frame): match by position, since an upload
        # may repeat an anon_id and an id lookup would give every repeat the last row's embedding.
        sel = list(range(len(ids)))
    else:
        cache_index = {cid: i for i, cid in enumerate(cache.case_ids)}
        missing = [cid for cid in ids if cid not in cache_index]
        if missing:
            raise ValueError(f"{len(missing)} case(s) missing from the embedding cache, e.g. {missing[:5]}")
        sel = [cache_index[cid] for cid in ids]

    concat_3 = cache.col_embeddings[sections.CONCAT_3_KEY][sel].astype(np.float32)
    group_input = group_classifier_input(cache, sel)
    texts = sections.merged_texts(dataframe)
    labels = [tl.term for tl in gen.taxonomy_labels]

    gate_mask, case_presence_probs = stages.run_case_presence(
        gen.case_presence, concat_3, gen.thresholds["case_presence_gate"],
    )
    group_probs = stages.run_group(gen.group_head, group_input, gate_mask)

    result = stages.categorize_cases(
        texts=texts,
        lp_embeddings=concat_3,
        label_embeddings=cache.label_embeddings,
        taxonomy_labels=gen.taxonomy_labels,
        labels=labels,
        group_probs=group_probs,
        group_names=gen.group_names,
        group_threshold=gen.thresholds["group"],
        tail_max_predictions=gen.thresholds["tail_max_predictions"],
        tail_max_group_prob_gap=gen.thresholds["tail_max_group_prob_gap"],
        presence_mask=gate_mask,
        uncommon_groups=gen.uncommon_groups,
        label_presence_heads=gen.label_presence_heads,
        label_presence_fallback=gen.thresholds["label_presence_fallback"],
        lp_thresholds=gen.lp_thresholds,
    )

    rows = prediction_rows(ids, result, gen.taxonomy_labels, labels, case_presence_probs, gen.generation_id)
    return rows


def predict_frame(gen, reports: pd.DataFrame, ids: list[str], *, device_arg: str = "auto",
                  local_only: bool = True) -> list[dict]:
    """Prediction rows for reports held in memory (ml-worker): embedded fresh with the generation's
    own backbone, never through the on-disk cache. ``reports`` has the raw report columns."""
    dataframe = sections.build_section_frame(reports)
    cache = build_fresh_cache(str(gen.petbert_dir), gen.taxonomy_labels, local_only=local_only,
                              device=backbone_mod.device_from_arg(device_arg), dataframe=dataframe, ids=ids)
    return classify(gen, cache, dataframe, ids)


def run_predict(
    *,
    generation_dir: str | Path | None = None,
    model_override: str | None = None,
    device_arg: str = "auto",
    local_only: bool = False,
    embed_only: bool = False,
    out_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> Path | None:
    """Load the generation, get/build its report embeddings (content-hash cache),
    and run every stage. Returns the predictions CSV path, or None with
    ``embed_only`` (the cache is populated; no classification runs).

    ``embed_only`` never reads thresholds, so it's allowed on an uncalibrated
    (``calibration.status != "calibrated"``) generation — e.g. calibrate.py's
    own workflow embeds a freshly trained candidate before calibrating it.
    Real classification refuses an uncalibrated generation (see
    ``report_mapping.model.generation.load_generation``).

    ``cache_dir`` overrides ``config.EMBEDDING_CACHE_DIR`` for both the lookup
    and the write — e.g. an L2b re-embed run on a scratch directory that must
    never read or overwrite the real (Syncthing-shared) cache; default
    (``None``) resolves to ``config.EMBEDDING_CACHE_DIR`` inside
    ``embedding_cache.cache_path``. An empty/new ``cache_dir`` always misses,
    so this always re-embeds and writes the fresh cache there.
    """
    gen = generation_mod.load_generation(generation_dir, allow_uncalibrated=embed_only)
    device = backbone_mod.device_from_arg(device_arg)
    model_dir_or_name = model_override if model_override is not None else str(gen.petbert_dir)

    dataframe, ids = _read_reports()
    dataframe = sections.build_section_frame(dataframe)

    key = _cache_key(gen, model_dir_or_name)
    cache = embedding_cache_mod.load(key, cache_dir)
    if cache is None:
        cache = build_fresh_cache(model_dir_or_name, gen.taxonomy_labels, local_only=local_only, device=device,
                                   dataframe=dataframe, ids=ids)
        embedding_cache_mod.save(key, cache, cache_dir)

    if embed_only:
        return None

    rows = classify(gen, cache, dataframe, ids)
    if out_path is None:
        out_path = config.PREDICTIONS_DIR / f"{gen.generation_id}_predictions.csv"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    io_utils.write_csv(pd.DataFrame(rows, columns=[
        "case_id", "diagnosis_index", "predicted_term", "predicted_group", "predicted_code",
        "case_presence_prob", "confidence", "group_prob", "method", "generation_id",
    ]), out_path)
    print(f"{len(rows)} prediction rows for {len(ids)} cases -> {out_path}")
    return out_path
