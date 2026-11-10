"""Tests for src/predict.py and src/benchmark.py, with a tiny random DistilBERT on the CPU."""
from __future__ import annotations

import pytest

from conftest import CONTEXT_CURIE, FILLER_WORDS, make_example
from src.benchmark import compare, model_size, summarize_times, time_questions
from src.config import Config
from src.predict import answer_question


@pytest.fixture(scope="module")
def tiny(tokenizer):
    import torch
    from transformers import DistilBertConfig, DistilBertForQuestionAnswering

    torch.manual_seed(0)
    cfg = Config()
    cfg.preprocessing.max_length, cfg.preprocessing.doc_stride = 64, 16
    model = DistilBertForQuestionAnswering(DistilBertConfig(
        vocab_size=tokenizer.vocab_size, dim=32, hidden_dim=64, n_layers=1, n_heads=2,
        max_position_embeddings=cfg.preprocessing.max_length))
    model.eval()
    return model, cfg


LONG_CONTEXT = " ".join(FILLER_WORDS[:100]) + " " + CONTEXT_CURIE


def test_answer_question_returns_a_span_from_the_passage(tokenizer, tiny):
    model, cfg = tiny
    p = answer_question(model, tokenizer, "When did Curie win the Chemistry prize?", CONTEXT_CURIE, cfg)
    assert p.best_span_text and p.best_span_text in CONTEXT_CURIE
    assert p.text in ("", p.best_span_text)
    assert 1 <= len(p.candidates) <= 5
    assert all(CONTEXT_CURIE[c["start_char"]:c["end_char"]] == c["text"] for c in p.candidates)


def test_threshold_argument_overrides_the_config(tokenizer, tiny):
    model, cfg = tiny
    question = "When did Curie win the Chemistry prize?"
    always = answer_question(model, tokenizer, question, CONTEXT_CURIE, cfg, null_threshold=1000.0)
    never = answer_question(model, tokenizer, question, CONTEXT_CURIE, cfg, null_threshold=-1000.0)
    assert always.text == always.best_span_text != ""
    assert never.text == ""
    assert always.score_diff == pytest.approx(never.score_diff), "only the decision changes, not the scores"


def test_long_passages_are_handled_with_several_windows(tokenizer, tiny):
    model, cfg = tiny
    p = answer_question(model, tokenizer, "When did Curie win the Chemistry prize?", LONG_CONTEXT, cfg,
                        null_threshold=1000.0, n_candidates=3)
    assert p.text in LONG_CONTEXT and len(p.candidates) == 3


def test_same_answer_as_the_evaluation_pipeline(tokenizer, tiny):
    """The demo path (one question) and the evaluation path (whole dataset) must agree."""
    datasets = pytest.importorskip("datasets")
    import torch

    from src.evaluate import evaluate_model

    model, cfg = tiny
    rows = [
        make_example("a", "When did Curie win the Chemistry prize?", CONTEXT_CURIE, "1911"),
        make_example("b", "Where was Curie born?", CONTEXT_CURIE, None),
        make_example("long", "What did Curie win in 1903?", LONG_CONTEXT, "the Nobel Prize in Physics"),
    ]
    _, batch_predictions, _ = evaluate_model(model, tokenizer, datasets.Dataset.from_list(rows), cfg,
                                             batch_size=4, device=torch.device("cpu"), use_official_metric=False)
    for row in rows:
        single = answer_question(model, tokenizer, row["question"], row["context"], cfg)
        batch = batch_predictions[row["id"]]
        assert single.text == batch.text and single.best_span_text == batch.best_span_text
        assert single.score_diff == pytest.approx(batch.score_diff, abs=1e-3)


def test_model_size_counts_every_weight(tiny):
    model, _ = tiny
    size = model_size(model)
    n = sum(p.numel() for p in model.parameters())
    assert size["parameters_millions"] == round(n / 1e6, 1)
    assert size["size_mb"] == round(4 * n / 1e6, 1)       # float32 = 4 bytes per weight


def test_time_questions_times_every_question_once(tokenizer, tiny):
    model, cfg = tiny
    rows = [make_example(str(i), "Where was Curie born?", CONTEXT_CURIE, None) for i in range(4)]
    ticks = iter(range(100))
    times = time_questions(model, tokenizer, rows, cfg, warmup=2, clock=lambda: next(ticks))
    assert times == [1, 1, 1, 1], "one start and one stop reading per question; warm-up is not timed"

    real = time_questions(model, tokenizer, rows, cfg, warmup=1)
    assert len(real) == 4 and all(t > 0 for t in real)


def test_summarize_times_uses_the_median():
    s = summarize_times([0.010, 0.012, 0.011, 0.500, 0.010])    # one slow outlier
    assert s["median_ms"] == 11.0
    assert s["mean_ms"] == pytest.approx(108.6)
    assert s["p90_ms"] == 500.0
    assert s["questions"] == 5 and s["questions_per_second"] == pytest.approx(90.9)


def test_compare_reports_size_ratio_and_speedup():
    results = {"small": {"size_mb": 250.0, "median_ms": 40.0}, "big": {"size_mb": 500.0, "median_ms": 100.0}}
    assert compare(results, "big") == {"small": {"size_ratio": 0.5, "speedup": 2.5}}
