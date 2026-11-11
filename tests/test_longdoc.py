"""Tests for src/longdoc.py, with a tiny random DistilBERT on the CPU."""
from __future__ import annotations

import pytest

from conftest import CONTEXT_CURIE, CONTEXT_FAME, FILLER_WORDS, make_example
from src.config import Config
from src.longdoc import PARAGRAPH_BREAK, as_long_examples, build_document, paragraph_scores, right_paragraph_rate
from src.postprocess import Prediction

FILLER = " ".join(FILLER_WORDS[:80]) + "."


def _article():
    """Three paragraphs, five questions, in the order SQuAD would list them."""
    return [
        make_example("a", "When did Curie win the Chemistry prize?", CONTEXT_CURIE, "1911"),
        make_example("b", "Where was Curie born?", CONTEXT_CURIE, None),
        make_example("c", "What is near the water?", FILLER, None),
        make_example("d", "When did she rise to fame?", CONTEXT_FAME, "in the late 1990s"),
        make_example("e", "What did Curie win in 1903?", CONTEXT_CURIE, "the Nobel Prize in Physics"),
    ]


def test_build_document_keeps_each_paragraph_once_and_in_order():
    document, starts = build_document(_article())
    assert document == PARAGRAPH_BREAK.join([CONTEXT_CURIE, FILLER, CONTEXT_FAME])
    assert list(starts) == [CONTEXT_CURIE, FILLER, CONTEXT_FAME]
    for paragraph, start in starts.items():
        assert document[start:start + len(paragraph)] == paragraph


def test_long_examples_point_at_the_same_answers_inside_the_document():
    long_examples, home = as_long_examples(_article())
    document = long_examples[0]["context"]
    assert all(ex["context"] == document for ex in long_examples)
    assert [ex["id"] for ex in long_examples] == list("abcde")
    for ex in long_examples:
        for text, start in zip(ex["answers"]["text"], ex["answers"]["answer_start"]):
            assert document[start:start + len(text)] == text, "answer_start must be moved with the paragraph"
    assert long_examples[1]["answers"] == {"text": [], "answer_start": []}
    low, high = home["d"]
    assert document[low:high] == CONTEXT_FAME
    assert home["a"] == home["b"] == home["e"] == (0, len(CONTEXT_CURIE))


def _prediction(text, start_char):
    candidate = {"text": text, "score": 1.0, "start_char": start_char, "end_char": start_char + len(text)}
    return Prediction(text=text, best_span_text=text, best_span_score=1.0, null_score=0.0, score_diff=-1.0,
                      candidates=[candidate] if text else [])


def test_right_paragraph_rate_counts_only_answered_answerable_questions():
    long_examples, home = as_long_examples(_article())
    document = long_examples[0]["context"]
    predictions = {
        "a": _prediction("1911", document.index("1911")),                       # right paragraph
        "b": _prediction("Marie", 0),                                           # unanswerable: not counted
        "c": _prediction("", 0),                                                # abstained: not counted
        "d": _prediction("1903", document.index("1903")),                       # answered from the WRONG paragraph
        "e": _prediction("", 0),                                                # abstained: not counted
    }
    assert right_paragraph_rate(predictions, home, long_examples) == 50.0
    nobody = {k: _prediction("", 0) for k in "abcde"}
    assert right_paragraph_rate(nobody, home, long_examples) is None


def test_paragraph_scores_average_the_saved_scores_of_these_questions():
    saved = {k: {"em": em, "f1": f1} for k, em, f1 in
             [("a", 1, 1.0), ("b", 0, 0.0), ("c", 1, 1.0), ("d", 0, 0.5), ("e", 1, 1.0), ("other", 0, 0.0)]}
    m = paragraph_scores(saved, _article())
    assert m["total"] == 5 and m["exact"] == 60.0 and m["f1"] == 70.0
    assert (m["HasAns_total"], m["NoAns_total"]) == (3, 2)
    assert m["HasAns_f1"] == pytest.approx(83.33) and m["NoAns_f1"] == 50.0


def test_evaluate_article_end_to_end(tokenizer):
    datasets = pytest.importorskip("datasets")
    import torch
    from transformers import DistilBertConfig, DistilBertForQuestionAnswering

    from src.longdoc import WINDOW_RULES, evaluate_article

    torch.manual_seed(0)
    cfg = Config()
    cfg.preprocessing.max_length, cfg.preprocessing.doc_stride = 64, 16
    model = DistilBertForQuestionAnswering(DistilBertConfig(
        vocab_size=tokenizer.vocab_size, dim=32, hidden_dim=64, n_layers=1, n_heads=2, max_position_embeddings=64))

    article = evaluate_article(model, tokenizer, datasets.Dataset.from_list(_article()), cfg,
                               batch_size=8, device=torch.device("cpu"))
    assert article["questions"] == 5 and article["paragraphs"] == 3
    assert article["windows_per_question"] > 2, "the whole article needs several windows per question"
    assert set(article["rules"]) == set(WINDOW_RULES)
    for rule in WINDOW_RULES:
        r = article["rules"][rule]
        assert r["at_tau_0"]["total"] == 5 and r["at_tau_0"]["HasAns_total"] == 3
        assert r["at_best_tau"]["f1"] >= r["at_tau_0"]["f1"] - 1e-6
        assert 0 <= r["answered_pct"] <= 100
