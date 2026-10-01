"""The term retriever: PooledClassifier with one BCE-trained logit per diagnosis term.

Layout on disk (identical to the Phase 0 retrievers, so rr_phase0/bce loads unchanged):
``encoder/``, ``head.pt``, ``terms.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from bronze_rr import encoder as enc

TERMS_FILE = "terms.json"


class TermRetriever(enc.PooledClassifier):
    def __init__(self, encoder, terms: list[str]):
        super().__init__(encoder, len(terms))
        self.terms = terms

    def save(self, directory: Path, tokenizer) -> None:
        super().save(directory, tokenizer)
        (directory / TERMS_FILE).write_text(json.dumps(self.terms), encoding="utf-8")


def new_retriever(terms: list[str]) -> TermRetriever:
    return TermRetriever(enc.new_encoder(), terms)


def read_terms(directory: Path) -> list[str]:
    return json.loads((directory / TERMS_FILE).read_text(encoding="utf-8"))


def load_retriever(directory: Path) -> tuple[TermRetriever, object]:
    model = TermRetriever(enc.new_encoder(str(directory / enc.ENCODER_DIR)), read_terms(directory))
    model.head.load_state_dict(torch.load(directory / enc.HEAD_FILE, map_location="cpu"))
    return model, enc.load_tokenizer(str(directory / enc.ENCODER_DIR))
