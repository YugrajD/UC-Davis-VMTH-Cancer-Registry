"""report_mapping/model.py and batching.py: the joint classifier, its on-disk layout, and the micro-batching."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from report_mapping import batching, generation
from report_mapping.inference import scoring
from report_mapping.model import JointClassifier, load_joint, load_tokenizer, new_encoder

TERMS = ["t1", "t2", "t3", "t4"]
TEXTS = ["mast cell tumor present", "no evidence of neoplasia", "carcinoma malignant with a longer sentence here"]


def _model(encoder_dir, terms=TERMS):
    torch.manual_seed(0)
    return JointClassifier(new_encoder(str(encoder_dir)), terms), load_tokenizer(str(encoder_dir))


def test_new_encoder_uses_sdpa_and_no_reference_compile(tiny_modernbert_dir):
    encoder = new_encoder(str(tiny_modernbert_dir))
    assert encoder.config._attn_implementation == "sdpa" and encoder.config.reference_compile is False


def test_heads_are_fp32_and_one_forward_gives_term_and_gate_logits(tiny_modernbert_dir):
    model, tokenizer = _model(tiny_modernbert_dir)
    batch = tokenizer(TEXTS[:2], padding=True, return_tensors="pt")
    with torch.no_grad():
        term, gate = model(batch["input_ids"], batch["attention_mask"])
    assert term.shape == (2, 4) and gate.shape == (2,) and term.dtype == gate.dtype == torch.float32
    assert model.head.weight.dtype == model.gate_head.weight.dtype == torch.float32


def test_mean_pooling_ignores_padding(tiny_modernbert_dir):
    model, tokenizer = _model(tiny_modernbert_dir)
    model.eval()
    padded = tokenizer([TEXTS[0], TEXTS[2]], padding=True, return_tensors="pt")
    alone = tokenizer([TEXTS[0]], return_tensors="pt")
    assert padded["attention_mask"][0].min() == 0       # the first text really is padded
    with torch.no_grad():
        pooled_padded = model.pooled(padded["input_ids"], padded["attention_mask"])[0]
        pooled_alone = model.pooled(alone["input_ids"], alone["attention_mask"])[0]
    assert torch.allclose(pooled_padded, pooled_alone, atol=1e-5)


def test_biases_start_at_the_train_priors(tiny_modernbert_dir):
    model, _ = _model(tiny_modernbert_dir)
    model.init_biases(torch.tensor([0.5, 0.1, 0.0, 1.0]), 0.3)
    assert torch.sigmoid(model.head.bias).tolist() == pytest.approx([0.5, 0.1, 1e-6, 1 - 1e-6], abs=1e-6)
    assert torch.sigmoid(model.gate_head.bias).item() == pytest.approx(0.3, abs=1e-6)


def test_save_writes_the_generation_layout_and_load_round_trips_the_logits(tmp_path, tiny_modernbert_dir):
    model, tokenizer = _model(tiny_modernbert_dir)
    model.init_biases(torch.tensor([0.1, 0.2, 0.3, 0.4]), 0.3)
    model.save(tmp_path / "g", tokenizer)
    names = {p.name for p in (tmp_path / "g").iterdir()}
    assert names == {generation.ENCODER_DIR, generation.HEAD_FILE, generation.GATE_HEAD_FILE, generation.TERMS_FILE}
    assert (tmp_path / "g" / "encoder" / "model.safetensors").is_file()
    assert json.loads((tmp_path / "g" / "terms.json").read_text(encoding="utf-8")) == TERMS

    term, gate, _ = scoring.logits_for(model, tokenizer, TEXTS, 64, "cpu")
    loaded, loaded_tokenizer = load_joint(tmp_path / "g")
    term2, gate2, _ = scoring.logits_for(loaded, loaded_tokenizer, TEXTS, 64, "cpu")
    assert loaded.terms == TERMS and np.array_equal(term, term2) and np.array_equal(gate, gate2)


def test_heads_are_plain_state_dicts_that_load_with_weights_only(tmp_path, tiny_modernbert_dir):
    model, tokenizer = _model(tiny_modernbert_dir)
    model.save(tmp_path / "g", tokenizer)
    for name in (generation.HEAD_FILE, generation.GATE_HEAD_FILE):
        state = torch.load(tmp_path / "g" / name, map_location="cpu", weights_only=True)
        assert set(state) == {"weight", "bias"}


def test_load_refuses_a_head_that_does_not_match_terms_json(tmp_path, tiny_modernbert_dir):
    model, tokenizer = _model(tiny_modernbert_dir)
    model.save(tmp_path / "g", tokenizer)
    (tmp_path / "g" / "terms.json").write_text(json.dumps(TERMS + ["extra"]), encoding="utf-8")
    with pytest.raises(ValueError, match="5 terms but the term head has 4"):
        load_joint(tmp_path / "g")


def test_logits_do_not_depend_on_how_the_texts_are_batched(tiny_modernbert_dir):
    model, tokenizer = _model(tiny_modernbert_dir)
    term, gate, over = scoring.logits_for(model, tokenizer, TEXTS, 64, "cpu", token_budget=40)
    assert over == 0 and term.shape == (3, 4) and gate.shape == (3,)
    for i, one in enumerate(TEXTS):    # one text at a time (no padding) gives the same numbers, in input order
        t1, g1, _ = scoring.logits_for(model, tokenizer, [one], 64, "cpu", token_budget=10**6)
        assert t1[0] == pytest.approx(term[i], abs=1e-4) and g1[0] == pytest.approx(gate[i], abs=1e-4)


def test_texts_over_max_length_are_counted_and_cut_keeping_the_closing_sep(tiny_modernbert_dir):
    tokenizer = load_tokenizer(str(tiny_modernbert_dir))
    ids, over = batching.encode_reports(tokenizer, ["mast cell tumor present with margin", "mast"], 5)
    assert over == 1 and len(ids[0]) == 5 and ids[0][-1] == tokenizer.sep_token_id and len(ids[1]) == 3


def test_micro_batches_respect_the_padded_token_budget_and_cover_every_case():
    lengths = [10, 3, 8, 3, 9, 1]
    batches = batching.micro_batches(list(range(6)), lengths, 20)
    assert sorted(i for b in batches for i in b) == list(range(6))
    assert all(max(lengths[i] for i in b) * len(b) <= 20 or len(b) == 1 for b in batches)


def test_bucketed_steps_partition_the_cases_and_are_deterministic_in_the_seed():
    lengths = list(range(50))
    steps = batching.bucketed_steps(lengths, 8, seed=3)
    assert sorted(i for s in steps for i in s) == list(range(50)) and max(len(s) for s in steps) == 8
    assert steps == batching.bucketed_steps(lengths, 8, seed=3) != batching.bucketed_steps(lengths, 8, seed=4)
