"""Tests for src/demo.py: what the demo shows, without starting Gradio."""
from __future__ import annotations

import pytest

from conftest import CONTEXT_CURIE
from src import demo
from src.config import Config
from src.demo import (
    ANSWER_LABEL,
    DEFAULT_MODEL,
    EXAMPLES,
    MODELS,
    NO_ANSWER_TEXT,
    REJECTED_LABEL,
    ask,
    confidence_shares,
    get_model,
    highlight,
    render,
)


def _result(score_diff=-2.0):
    """A hand-written model result: best span "1911", runner-up "1903"."""
    start = CONTEXT_CURIE.index("1911")
    other = CONTEXT_CURIE.index("1903")
    return {
        "context": CONTEXT_CURIE,
        "best_span": "1911",
        "best_span_score": 6.0,
        "null_score": 6.0 + score_diff,
        "score_diff": score_diff,
        "candidates": [
            {"text": "1911", "score": 6.0, "start_char": start, "end_char": start + 4},
            {"text": "1903", "score": 3.0, "start_char": other, "end_char": other + 4},
        ],
        "seconds": 0.031,
    }


def test_highlight_keeps_the_whole_passage_and_marks_only_the_span():
    start = CONTEXT_CURIE.index("1911")
    pieces = highlight(CONTEXT_CURIE, start, start + 4, ANSWER_LABEL)
    assert "".join(text for text, _ in pieces) == CONTEXT_CURIE
    assert [(text, label) for text, label in pieces if label] == [("1911", ANSWER_LABEL)]
    # A span at the very start leaves no empty piece in front of it.
    assert highlight(CONTEXT_CURIE, 0, 5, ANSWER_LABEL)[0] == ("Marie", ANSWER_LABEL)


def test_confidence_shares_add_up_and_follow_the_scores():
    shares = confidence_shares(_result()["candidates"], null_score=4.0)
    assert sum(shares) == pytest.approx(100.0)
    assert shares[0] > shares[2] > shares[1]            # scores 6 > 4 (no answer) > 3
    assert confidence_shares([{"score": 2.0}], 2.0) == pytest.approx([50.0, 50.0])
    assert sum(confidence_shares([{"score": 900.0}], -900.0)) == pytest.approx(100.0)   # no overflow


def test_render_shows_the_answer_when_the_span_wins():
    headline, passage, rows, why = render(_result(score_diff=-2.0), tau=0.0)
    assert headline == "### 1911"
    assert ("1911", ANSWER_LABEL) in passage and "".join(t for t, _ in passage) == CONTEXT_CURIE
    assert [r[0] for r in rows] == ["1911", "1903", NO_ANSWER_TEXT]
    assert rows[-1][1] == 4.0 and all(r[2].endswith("%") for r in rows)
    assert "beats" in why and "31 ms" in why


def test_render_shows_no_answer_and_the_rejected_guess():
    headline, passage, rows, why = render(_result(score_diff=1.5), tau=0.0)
    assert "No answer" in headline
    assert ("1911", REJECTED_LABEL) in passage, "the rejected span is still pointed out"
    assert "1.50" in why and "Raise the threshold" in why


def test_explanation_for_a_span_that_wins_but_not_by_enough():
    # The span leads by 1.14, but a cautious threshold of -1.4 wants a lead of 1.4.
    headline, _, _, why = render(_result(score_diff=-1.14), tau=-1.4)
    assert "No answer" in headline
    assert "by only 1.14" in why and "at least 1.40" in why and "-1.14 or more" in why


def test_the_slider_flips_the_decision_without_a_new_model_run():
    result = _result(score_diff=1.5)
    assert "No answer" in render(result, tau=1.0)[0]
    assert render(result, tau=2.0)[0] == "### 1911"
    assert render(result, tau=1.5)[0] == "### 1911", "at exactly the threshold the span is given"
    assert "No answer" in render(_result(score_diff=-2.0), tau=-3.0)[0]
    assert result == _result(score_diff=1.5), "render must not change the stored result"


def test_render_with_nothing_asked_yet():
    assert render(None, 0.0) == ("", [], [], "")


def test_ask_runs_a_model_and_its_result_renders(tokenizer):
    from transformers import DistilBertConfig, DistilBertForQuestionAnswering

    cfg = Config()
    cfg.preprocessing.max_length, cfg.preprocessing.doc_stride = 64, 16
    model = DistilBertForQuestionAnswering(DistilBertConfig(
        vocab_size=tokenizer.vocab_size, dim=32, hidden_dim=64, n_layers=1, n_heads=2, max_position_embeddings=64))
    model.eval()

    result = ask(model, tokenizer, "  When did Curie win the Chemistry prize? ", CONTEXT_CURIE, cfg)
    assert result["best_span"] in CONTEXT_CURIE and result["seconds"] > 0
    assert result["score_diff"] == pytest.approx(result["null_score"] - result["best_span_score"])
    import json
    json.dumps(result)   # must be plain data: Gradio keeps it in the session between clicks

    headline, passage, rows, _ = render(result, tau=1000.0)
    assert headline == f"### {result['best_span']}"
    assert "".join(t for t, _ in passage) == CONTEXT_CURIE and len(rows) == len(result["candidates"]) + 1
    assert "No answer" in render(result, tau=-1000.0)[0]


def test_models_are_loaded_once_and_reused(monkeypatch):
    monkeypatch.setattr(demo, "_loaded", {})
    calls = []

    def fake_loader(hub_id, device):
        calls.append((hub_id, device))
        return ("model", "tokenizer")

    assert get_model(DEFAULT_MODEL, loader=fake_loader) == ("model", "tokenizer")
    get_model(DEFAULT_MODEL, loader=fake_loader)
    assert calls == [("Ingsyling/distilbert-squad2", "cpu")]


def test_model_list_and_examples_are_well_formed():
    assert DEFAULT_MODEL in MODELS and len(MODELS) == 2
    assert all(-8 <= m.default_tau <= 8 and "/" in m.hub_id for m in MODELS.values())
    assert all(len(row) == 2 and row[1].endswith("?") for row in EXAMPLES)
