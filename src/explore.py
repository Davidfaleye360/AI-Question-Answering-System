"""Dataset statistics used by notebooks/01_explore_data.ipynb.

All functions accept either a Hugging Face `Dataset` or a plain list of SQuAD-style dicts,
so they can be unit-tested without downloading anything.
"""
from __future__ import annotations

import random
import re
from collections import Counter

WH_WORDS = ("what", "which", "who", "whom", "whose", "when", "where", "why", "how")


def _column(ds, name: str) -> list:
    """A column as a plain list, for both Dataset objects and lists of dicts."""
    if isinstance(ds, list):
        return [row[name] for row in ds]
    return list(ds[name])


def question_type(question: str) -> str:
    """Bucket a question by its first wh-word ("In what year..." -> "what").

    "how many" and "how much" get their own buckets because they usually ask for numbers.
    """
    words = re.findall(r"[a-z]+", question.lower())
    for i, w in enumerate(words):
        if w in WH_WORDS:
            if w == "how" and i + 1 < len(words) and words[i + 1] in ("many", "much"):
                return f"how {words[i + 1]}"
            return "who" if w in ("whom", "whose") else w
    return "other"


def dataset_summary(ds) -> dict:
    """Counts of questions, answerable/unanswerable, passages and articles."""
    answers = _column(ds, "answers")
    n = len(answers)
    n_ans = sum(1 for a in answers if len(a["text"]) > 0)
    return {
        "questions": n,
        "answerable": n_ans,
        "unanswerable": n - n_ans,
        "unanswerable_pct": round(100 * (n - n_ans) / n, 1) if n else 0.0,
        "passages": len(set(_column(ds, "context"))),
        "articles": len(set(_column(ds, "title"))),
    }


def question_type_counts(ds) -> Counter:
    """How many questions start with each wh-word."""
    return Counter(question_type(q) for q in _column(ds, "question"))


def answer_lengths_words(ds) -> list[int]:
    """Length in words of the first gold answer, for answerable questions only."""
    return [len(a["text"][0].split()) for a in _column(ds, "answers") if len(a["text"]) > 0]


def sample_rows(ds, n: int, seed: int = 42) -> list[dict]:
    """A reproducible random sample of rows."""
    size = len(ds)
    idx = sorted(random.Random(seed).sample(range(size), min(n, size)))
    return [ds[i] for i in idx]


def token_lengths(rows: list[dict], tokenizer) -> tuple[list[int], list[int]]:
    """(question token counts, passage token counts) for a list of rows, without special tokens."""
    q = tokenizer([r["question"].strip() for r in rows], add_special_tokens=False)["input_ids"]
    c = tokenizer([r["context"] for r in rows], add_special_tokens=False)["input_ids"]
    return [len(x) for x in q], [len(x) for x in c]


def share_needing_windows(q_lens: list[int], c_lens: list[int], max_length: int = 384) -> float:
    """Share of examples whose question + passage + 3 special tokens exceed max_length.

    Those examples get split into more than one window during preprocessing.
    """
    if not q_lens:
        return 0.0
    over = sum(1 for q, c in zip(q_lens, c_lens) if q + c + 3 > max_length)
    return over / len(q_lens)


def percentile(values: list[int], p: float) -> float:
    """Simple percentile (nearest rank), p in [0, 100]."""
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, round(p / 100 * (len(s) - 1))))
    return float(s[k])
