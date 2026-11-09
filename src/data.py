"""Load the SQuAD 2.0 dataset.

Each example has: id, title, context, question, answers.
`answers` is {"text": [...], "answer_start": [...]}; both lists are empty for unanswerable questions.
`answer_start` is a CHARACTER offset into `context`, not a word or token index.
"""
from __future__ import annotations

from datasets import Dataset, DatasetDict, load_dataset

DATASET_NAME = "rajpurkar/squad_v2"


def is_answerable(example: dict) -> bool:
    """True if the question has at least one gold answer."""
    return len(example["answers"]["text"]) > 0


def take_fraction(ds: Dataset, fraction: float, seed: int = 42) -> Dataset:
    """Return a random fraction of a split (e.g. 0.01 for a quick smoke test)."""
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if fraction == 1:
        return ds
    n = max(1, int(len(ds) * fraction))
    return ds.shuffle(seed=seed).select(range(n))


def load_squad(
    dataset_name: str = DATASET_NAME,
    train_fraction: float = 1.0,
    eval_fraction: float = 1.0,
    seed: int = 42,
) -> DatasetDict:
    """Download (or load from cache) SQuAD 2.0 with "train" and "validation" splits.

    We use "validation" (the official dev set, 11,873 questions) as our test set,
    because the official SQuAD 2.0 test set is hidden.
    """
    raw = load_dataset(dataset_name)
    return DatasetDict(
        train=take_fraction(raw["train"], train_fraction, seed),
        validation=take_fraction(raw["validation"], eval_fraction, seed),
    )
