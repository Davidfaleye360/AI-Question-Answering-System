"""Shared test fixtures.

The tests build a tiny WordPiece tokenizer from a local vocabulary file, so they run
offline in a few seconds. It behaves like DistilBERT's tokenizer where it matters here:
[CLS]/[SEP] special tokens, lowercasing, "##" subword pieces and character offsets.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

# Text used across the tests. Every word goes into the vocabulary unless listed in EXCLUDE,
# so we can force a word to split into subword pieces.
CONTEXT_CURIE = "Marie Curie won the Nobel Prize in Physics in 1903 and in Chemistry in 1911."
CONTEXT_FAME = "She rose to fame in the late 1990s as lead singer of the group."
FILLER_WORDS = [f"filler{i}" for i in range(120)]
EXTRA_TEXT = [
    "When did Curie win the Chemistry prize? What did Curie win in 1903? Where was Curie born?",
    "When did she rise to fame? In what decade? What is near the water?",
    "The golden gate bridge is near the water.",
]
EXCLUDE = {"1990s"}            # becomes "1990" + "##s"
EXTRA_PIECES = ["##s"]


def build_vocab(texts: list[str]) -> list[str]:
    words = set()
    for t in texts:
        words.update(re.findall(r"\w+|[^\w\s]", t.lower()))
    words -= EXCLUDE
    words.add("1990")
    return SPECIAL_TOKENS + sorted(words) + EXTRA_PIECES


@pytest.fixture(scope="session")
def tokenizer(tmp_path_factory):
    from transformers import BertTokenizerFast

    vocab = build_vocab([CONTEXT_CURIE, CONTEXT_FAME, " ".join(FILLER_WORDS), *EXTRA_TEXT])
    path = tmp_path_factory.mktemp("tok") / "vocab.txt"
    path.write_text("\n".join(vocab) + "\n", encoding="utf-8")
    return BertTokenizerFast(vocab_file=str(path), do_lower_case=True)


def make_example(ex_id: str, question: str, context: str, answer: str | None) -> dict:
    """A SQuAD-style example. answer=None makes it unanswerable."""
    if answer is None:
        answers = {"text": [], "answer_start": []}
    else:
        start = context.index(answer)
        answers = {"text": [answer], "answer_start": [start]}
    return {"id": ex_id, "title": "Test", "question": question, "context": context, "answers": answers}


def to_batch(rows: list[dict]) -> dict[str, list]:
    """List of examples -> column dict, the format Dataset.map(batched=True) passes in."""
    return {k: [r[k] for r in rows] for k in rows[0]}
