"""Tests for src/explore.py, src/config.py and the preprocessing sanity-check script."""
from __future__ import annotations

from pathlib import Path

import pytest

from conftest import CONTEXT_CURIE, CONTEXT_FAME, FILLER_WORDS, ROOT, make_example, to_batch
from src.config import load_config
from src.explore import (
    answer_lengths_words,
    dataset_summary,
    percentile,
    question_type,
    question_type_counts,
    share_needing_windows,
    token_lengths,
)


@pytest.mark.parametrize("question,expected", [
    ("When did Beyonce start becoming popular?", "when"),
    ("In what city and state did Beyonce grow up?", "what"),
    ("How many Grammy awards did she win?", "how many"),
    ("How much did the tour earn?", "how much"),
    ("Whose album was released first?", "who"),
    ("Name the group she led.", "other"),
])
def test_question_type(question, expected):
    assert question_type(question) == expected


def _rows():
    return [
        make_example("a", "When did Curie win the Chemistry prize?", CONTEXT_CURIE, "1911"),
        make_example("b", "Where was Curie born?", CONTEXT_CURIE, None),
        make_example("c", "When did she rise to fame?", CONTEXT_FAME, "in the late 1990s"),
    ]


def test_dataset_summary_counts():
    s = dataset_summary(_rows())
    assert s == {"questions": 3, "answerable": 2, "unanswerable": 1, "unanswerable_pct": 33.3,
                 "passages": 2, "articles": 1}


def test_question_types_and_answer_lengths():
    assert question_type_counts(_rows()) == {"when": 2, "where": 1}
    assert answer_lengths_words(_rows()) == [1, 4]


def test_token_lengths_and_window_share(tokenizer):
    rows = _rows() + [make_example("d", "What is near the water?", " ".join(FILLER_WORDS[:100]), None)]
    q, c = token_lengths(rows, tokenizer)
    assert len(q) == len(c) == 4
    assert share_needing_windows(q, c, max_length=64) == 0.25


def test_percentile():
    assert percentile([5, 1, 3, 2, 4], 50) == 3
    assert percentile([], 90) == 0.0


def test_project_config_loads():
    cfg = load_config(ROOT / "configs" / "distilbert.yaml")
    assert cfg.model.checkpoint == "distilbert-base-uncased"
    assert cfg.preprocessing.max_length == 384 and cfg.preprocessing.doc_stride == 128
    assert cfg.preprocessing.doc_stride < cfg.preprocessing.max_length
    assert cfg.training.learning_rate == pytest.approx(3e-5)


def test_config_rejects_typos(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("training:\n  learning_rat: 0.1\n")
    with pytest.raises(ValueError, match="learning_rat"):
        load_config(bad)


def test_sanity_check_script_passes_on_good_labels(tokenizer):
    import importlib.util

    spec = importlib.util.spec_from_file_location("check", ROOT / "scripts" / "check_preprocessing.py")
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    rows = _rows() + [make_example("d", "In what decade?", CONTEXT_FAME, "1990")]
    report = check.run_checks(to_batch(rows), tokenizer, max_length=64, stride=16)
    assert report["passed"]
    assert report["answerable"] == 3 and report["exact"] == 3
    assert not report["bad_unanswerable"]
