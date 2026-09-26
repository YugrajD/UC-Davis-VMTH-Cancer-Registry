"""report_mapping/inference/embedding_cache.py: content-hash keying, hit/miss, legacy import."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from report_mapping.inference import embedding_cache as ec

from . import fixtures as fx


@pytest.fixture(autouse=True)
def _redirect_cache_dir(tmp_path, monkeypatch):
    import config

    monkeypatch.setattr(config, "EMBEDDING_CACHE_DIR", tmp_path / "embedding_cache")


def _fingerprint(backbone_sha="abc", max_length=512, version=1):
    return {"backbone_sha256": backbone_sha, "section_spec_version": version, "max_length": max_length}


def test_content_key_changes_with_report_bytes(tmp_path):
    report_a = tmp_path / "report_a.csv"
    report_b = tmp_path / "report_b.csv"
    labels = tmp_path / "labels.csv"
    report_a.write_text("case_id\nCASE-0001\n", encoding="utf-8")
    report_b.write_text("case_id\nCASE-0002\n", encoding="utf-8")
    labels.write_text("labels\n", encoding="utf-8")
    key_a = ec.content_key(report_a, labels, _fingerprint())
    key_b = ec.content_key(report_b, labels, _fingerprint())
    assert key_a != key_b


def test_content_key_changes_with_fingerprint(tmp_path):
    report = tmp_path / "report.csv"
    labels = tmp_path / "labels.csv"
    report.write_text("case_id\nCASE-0001\n", encoding="utf-8")
    labels.write_text("labels\n", encoding="utf-8")
    key1 = ec.content_key(report, labels, _fingerprint(backbone_sha="abc"))
    key2 = ec.content_key(report, labels, _fingerprint(backbone_sha="def"))
    key3 = ec.content_key(report, labels, _fingerprint(version=2))
    assert len({key1, key2, key3}) == 3


def test_content_key_deterministic(tmp_path):
    report = tmp_path / "report.csv"
    labels = tmp_path / "labels.csv"
    report.write_text("case_id\nCASE-0001\n", encoding="utf-8")
    labels.write_text("labels\n", encoding="utf-8")
    assert ec.content_key(report, labels, _fingerprint()) == ec.content_key(report, labels, _fingerprint())


def test_load_miss_returns_none():
    assert ec.load("does-not-exist") is None


def _sample_cache() -> ec.EmbeddingCache:
    return ec.EmbeddingCache(
        case_ids=["CASE-0001", "CASE-0002"],
        col_embeddings={
            "__sec_0__": np.random.randn(2, 8).astype(np.float32),
            "__sec_1__": np.random.randn(2, 8).astype(np.float32),
            "__sec_2__": np.random.randn(2, 8).astype(np.float32),
            "concat_3": np.random.randn(2, 24).astype(np.float32),
        },
        col_has_content={
            "__sec_0__": np.array([True, False]),
            "__sec_1__": np.array([True, True]),
            "__sec_2__": np.array([False, True]),
        },
        label_texts=["Term A Group A", "Term B Group B"],
        label_embeddings=np.random.randn(2, 8).astype(np.float32),
    )


def test_save_then_load_round_trips_a_hit():
    cache = _sample_cache()
    ec.save("some-key", cache)
    loaded = ec.load("some-key")
    assert loaded is not None
    assert loaded.case_ids == cache.case_ids
    assert loaded.label_texts == cache.label_texts
    np.testing.assert_array_equal(loaded.col_embeddings["concat_3"], cache.col_embeddings["concat_3"])
    np.testing.assert_array_equal(loaded.col_has_content["__sec_0__"], cache.col_has_content["__sec_0__"])
    np.testing.assert_array_equal(loaded.label_embeddings, cache.label_embeddings)


def test_cache_dir_override_reads_and_writes_a_separate_directory(tmp_path):
    # An explicit cache_dir must never touch config.EMBEDDING_CACHE_DIR (the
    # autouse fixture's tmp_path/embedding_cache) -- e.g. an L2b scratch dir
    # that must not read or overwrite the real (Syncthing-shared) cache.
    scratch_dir = tmp_path / "scratch"
    cache = _sample_cache()
    ec.save("some-key", cache, scratch_dir)

    assert ec.load("some-key") is None  # a miss in the default directory
    assert not any(Path(ec.cache_path("some-key")).parent.glob("*.npz"))

    loaded = ec.load("some-key", scratch_dir)
    assert loaded is not None
    assert loaded.case_ids == cache.case_ids
    assert ec.cache_path("some-key", scratch_dir) == scratch_dir / "some-key.npz"


def test_import_legacy_cache_reads_read_only_and_writes_new_format(tmp_path, monkeypatch, report_mapping_bundle):
    import config

    legacy_npz = tmp_path / "legacy_embedding_cache.npz"
    legacy_col_names = ["__sec_0__", "__sec_1__", "__sec_2__", "concat_3"]
    arrays = {
        "case_ids": np.array(["CASE-0001", "CASE-0002"], dtype=object),
        "col_names": np.array(legacy_col_names, dtype=object),
        "label_texts": np.array(["Term A Group A"], dtype=object),
        "label_embeddings": np.random.randn(1, 8).astype(np.float32),
    }
    for col in legacy_col_names:
        arrays[f"col_{col}"] = np.random.randn(2, 8 if col != "concat_3" else 24).astype(np.float32)
        arrays[f"has_{col}"] = np.array([True, True])
    np.savez(legacy_npz, **arrays)
    original_bytes = legacy_npz.read_bytes()
    monkeypatch.setattr(config, "LEGACY_EMBEDDING_CACHE_NPZ", legacy_npz)
    report_csv = tmp_path / "report.csv"  # the cache key hashes report bytes; never read the real data
    report_csv.write_text("case_id\nCASE-0001\nCASE-0002\n", encoding="latin-1")
    monkeypatch.setattr(config, "REPORT_CSV", report_csv)

    key = ec.import_legacy_cache(report_mapping_bundle)

    assert legacy_npz.read_bytes() == original_bytes  # read-only: never modified
    loaded = ec.load(key)
    assert loaded is not None
    assert loaded.case_ids == ["CASE-0001", "CASE-0002"]
    assert loaded.label_texts == ["Term A Group A"]
    assert loaded.col_embeddings["concat_3"].shape == (2, 24)


def test_predict_and_training_embeddings_compute_the_same_cache_key(tmp_path, monkeypatch, report_mapping_bundle):
    # report_mapping.inference.predict._embed_fresh's cache-miss path and
    # report_mapping.training.embeddings.get_or_build share one embedding
    # loop (report_mapping.inference.predict.build_fresh_cache) and must
    # resolve to the SAME content-hash key for the same generation, report
    # and labels -- otherwise heads-only training would silently re-embed
    # instead of reusing predict.py's cache. Proven behaviourally: building
    # the cache via one path, then running the other, must be a hit (one
    # cache file total), not a miss (a second, differently-keyed file).
    import config
    from report_mapping.inference.predict import run_predict
    from report_mapping.model.generation import generation_paths, load_generation
    from report_mapping.training import embeddings as embeddings_mod
    from taxonomy.taxonomy import load_labels_taxonomy

    monkeypatch.setattr(config, "REPORT_CSV", fx.make_reports_csv(tmp_path / "report.csv"))
    monkeypatch.setattr(config, "PREDICTIONS_DIR", tmp_path / "predictions")
    cache_dir = tmp_path / "embedding_cache"
    monkeypatch.setattr(config, "EMBEDDING_CACHE_DIR", cache_dir)

    gen = load_generation(report_mapping_bundle)
    labels_csv = str(generation_paths(report_mapping_bundle).labels_csv)
    taxonomy_labels = load_labels_taxonomy(labels_csv)

    embeddings_mod.get_or_build(str(gen.petbert_dir), labels_csv, taxonomy_labels, local_only=True, device="cpu")
    assert len(list(cache_dir.glob("*.npz"))) == 1

    run_predict(generation_dir=report_mapping_bundle, local_only=True, device_arg="cpu", embed_only=True)
    assert len(list(cache_dir.glob("*.npz"))) == 1  # still just one file: same key, a hit not a second miss
