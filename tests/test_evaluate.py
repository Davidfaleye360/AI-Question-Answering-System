"""Tests for src/evaluate.py.

The scoring tests need no model. The end-to-end test runs a tiny, randomly initialised
DistilBERT on the CPU: its answers are nonsense, but that is enough to check the wiring
(windows -> scores -> answers -> EM/F1 -> result files).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from conftest import CONTEXT_CURIE, CONTEXT_FAME, FILLER_WORDS, make_example
from src.config import Config
from src.evaluate import (
    PADDING_SCORE,
    evaluate_model,
    exact_match,
    f1_score,
    make_eval_features,
    normalize_answer,
    official_scores,
    per_example_scores,
    predict_logits,
    save_results,
    score_example,
    scores_agree,
    summarize,
)


@pytest.mark.parametrize("text,expected", [
    ("The  Cat.", "cat"),
    ("an Apple, a day!", "apple day"),
    ("1990s", "1990s"),
    ("there", "there"),          # "the" is only removed as a whole word
    ("", ""),
    ("The.", ""),
])
def test_normalize_answer(text, expected):
    assert normalize_answer(text) == expected


def test_exact_match_ignores_case_punctuation_and_articles():
    assert exact_match("the Nobel Prize", "Nobel prize.") == 1
    assert exact_match("Nobel Prize", "Nobel Prize in Physics") == 0


def test_f1_gives_partial_credit():
    # 2 shared words; prediction has 2 (precision 1.0), gold has 4 (recall 0.5) -> F1 = 2/3
    assert f1_score("Nobel Prize", "the Nobel Prize in Physics") == pytest.approx(2 / 3)
    assert f1_score("in 1911", "1911") == pytest.approx(2 / 3)
    assert f1_score("1903", "1911") == 0.0
    assert f1_score("1911", "1911") == 1.0


def test_f1_counts_repeated_words_only_as_often_as_they_match():
    # "prize prize" shares one "prize" with "prize": precision 1/2, recall 1 -> 2/3
    assert f1_score("prize prize", "prize") == pytest.approx(2 / 3)


def test_no_answer_scoring():
    assert f1_score("", "") == 1.0
    assert f1_score("1911", "") == 0.0
    assert f1_score("", "1911") == 0.0
    assert score_example("", []) == (1, 1.0)              # correctly abstained
    assert score_example("1911", []) == (0, 0.0)          # answered an unanswerable question
    assert score_example("", ["1911"]) == (0, 0.0)        # abstained on an answerable one


def test_best_gold_answer_counts():
    em, f1 = score_example("late 1990s", ["in the late 1990s", "late 1990s", "1990s"])
    assert (em, f1) == (1, 1.0)
    em, f1 = score_example("the late", ["in the late 1990s", "1990s"])
    assert em == 0 and f1 == pytest.approx(0.5)           # "late" vs "in late 1990s": P 1, R 1/3


def _rows():
    return [
        make_example("a", "When did Curie win the Chemistry prize?", CONTEXT_CURIE, "1911"),
        make_example("b", "What did Curie win in 1903?", CONTEXT_CURIE, "the Nobel Prize in Physics"),
        make_example("c", "Where was Curie born?", CONTEXT_CURIE, None),
        make_example("d", "When did she rise to fame?", CONTEXT_FAME, "in the late 1990s"),
        make_example("e", "What is near the water?", CONTEXT_FAME, None),
    ]


def test_summarize_splits_answerable_and_unanswerable():
    answers = {"a": "1911", "b": "Nobel Prize", "c": "", "d": "", "e": "the group"}
    scores = per_example_scores(answers, _rows())
    assert scores["b"] == {"em": 0, "f1": pytest.approx(2 / 3), "answerable": True}
    assert scores["c"] == {"em": 1, "f1": 1.0, "answerable": False}

    m = summarize(scores)
    assert (m["total"], m["HasAns_total"], m["NoAns_total"]) == (5, 3, 2)
    assert m["exact"] == pytest.approx(40.0)                       # a and c
    assert m["f1"] == pytest.approx(100 * (1 + 2 / 3 + 1) / 5)
    assert m["HasAns_exact"] == pytest.approx(100 / 3)
    assert m["HasAns_f1"] == pytest.approx(100 * (1 + 2 / 3) / 3)
    assert m["NoAns_exact"] == m["NoAns_f1"] == pytest.approx(50.0)


def test_summarize_without_unanswerable_questions():
    scores = per_example_scores({"a": "1911"}, _rows()[:1])
    assert summarize(scores) == {"exact": 100.0, "f1": 100.0, "total": 1,
                                 "HasAns_exact": 100.0, "HasAns_f1": 100.0, "HasAns_total": 1}


def test_always_abstaining_scores_only_on_unanswerable_questions():
    m = summarize(per_example_scores({k: "" for k in "abcde"}, _rows()))
    assert m["HasAns_f1"] == 0.0 and m["NoAns_f1"] == 100.0 and m["exact"] == pytest.approx(40.0)


def test_scores_agree():
    ours = {"exact": 40.0, "f1": 53.333, "total": 5}
    assert scores_agree(ours, {"exact": 40.0, "f1": 53.3333})
    assert not scores_agree(ours, {"exact": 40.0, "f1": 54.0})
    assert not scores_agree(ours, {"exact": 40.0, "HasAns_f1": 10.0})   # a number we don't have


def test_matches_the_official_metric():
    """Skipped offline: the official metric script is downloaded from the Hugging Face Hub."""
    answers = {"a": "in 1911", "b": "Nobel Prize", "c": "", "d": "", "e": "the group"}
    try:
        official = official_scores(answers, _rows())
    except Exception as error:   # no internet, or the `evaluate` library is missing
        pytest.skip(f"official squad_v2 metric not available: {error}")
    ours = summarize(per_example_scores(answers, _rows()))
    assert scores_agree(ours, official), (ours, official)
    assert set(official) == set(ours)


def _tiny_setup(tokenizer):
    datasets = pytest.importorskip("datasets")
    from transformers import DistilBertConfig, DistilBertForQuestionAnswering

    long_row = make_example("long", "When did Curie win the Chemistry prize?",
                            " ".join(FILLER_WORDS[:100]) + " " + CONTEXT_CURIE, "1911")
    examples = datasets.Dataset.from_list(_rows() + [long_row])
    cfg = Config()
    cfg.preprocessing.max_length, cfg.preprocessing.doc_stride = 64, 16
    model = DistilBertForQuestionAnswering(DistilBertConfig(
        vocab_size=tokenizer.vocab_size, dim=32, hidden_dim=64, n_layers=1, n_heads=2,
        max_position_embeddings=cfg.preprocessing.max_length))
    return model, examples, cfg


def test_predict_logits_shape_and_batching(tokenizer):
    import torch

    model, examples, cfg = _tiny_setup(tokenizer)
    features = make_eval_features(examples, tokenizer, cfg.preprocessing)
    assert len(features) > len(examples), "the long passage should give extra windows"

    cpu = torch.device("cpu")
    start, end = predict_logits(model, features, batch_size=4, device=cpu)
    assert start.shape == end.shape == (len(features), 64)

    # The batch size, and the trick of cutting each batch at its longest window, must not change the scores.
    start1, end1 = predict_logits(model, features, batch_size=1, device=cpu)
    real = np.array(features["attention_mask"]) == 1
    np.testing.assert_allclose(start[real], start1[real], atol=1e-4)
    np.testing.assert_allclose(end[real], end1[real], atol=1e-4)
    assert (start1[~real] == PADDING_SCORE).all(), "padding must never look like a good answer"


def test_evaluate_model_end_to_end_and_saved_files(tokenizer, tmp_path):
    import torch

    model, examples, cfg = _tiny_setup(tokenizer)
    summary, predictions, scores = evaluate_model(
        model, tokenizer, examples, cfg, batch_size=4, device=torch.device("cpu"), use_official_metric=False)

    assert summary["questions"] == 6 and summary["windows"] > 6
    assert summary["metrics"]["total"] == 6
    assert (summary["metrics"]["HasAns_total"], summary["metrics"]["NoAns_total"]) == (4, 2)
    assert summary["settings"]["null_threshold"] == cfg.postprocessing.null_threshold
    assert set(predictions) == set(scores) == set(examples["id"])
    for example in examples:
        p = predictions[example["id"]]
        assert p.text in example["context"], "answers are always cut out of the passage"
        assert p.text == ("" if p.score_diff > 0 else p.best_span_text)

    summary_path, predictions_path = save_results(tmp_path / "res", "tiny", summary, predictions, scores)
    assert json.loads(summary_path.read_text())["metrics"] == summary["metrics"]
    saved = json.loads(predictions_path.read_text())
    assert set(saved) == set(predictions)
    assert set(saved["a"]) == {"prediction", "best_span", "score_diff", "em", "f1"}
    assert saved["a"]["prediction"] == predictions["a"].text
