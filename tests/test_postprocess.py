"""Tests for src/postprocess.py.

No model is needed: each test writes the start/end scores by hand (a high number where we
want the "model" to point) and checks what post-processing makes of them.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import CONTEXT_CURIE, FILLER_WORDS, make_example, to_batch
from src.postprocess import apply_threshold, best_spans_in_window, choose_answer, postprocess_predictions
from src.preprocess import prepare_eval_features

LOW = -10.0   # score of every token we don't point at


def _features(rows, tokenizer, max_length=64, stride=16):
    """Windows for the given examples, as a list of dicts (one per window)."""
    enc = prepare_eval_features(to_batch(rows), tokenizer, max_length=max_length, stride=stride)
    return [{"example_id": enc["example_id"][i], "offset_mapping": enc["offset_mapping"][i]}
            for i in range(len(enc["input_ids"]))]


def _token_at(feature, context, text, last=False):
    """Index of the token that starts (or, with last=True, ends) `text` in this window."""
    start_char = context.index(text)
    end_char = start_char + len(text)
    for k, offset in enumerate(feature["offset_mapping"]):
        if offset is None:
            continue
        if (not last and offset[0] == start_char) or (last and offset[1] == end_char):
            return k
    raise AssertionError(f"{text!r} is not in this window")


def _blank_scores(features):
    shape = (len(features), len(features[0]["offset_mapping"]))
    return np.full(shape, LOW), np.full(shape, LOW)


def _point_at(features, start, end, window, context, text, score=5.0):
    start[window, _token_at(features[window], context, text)] = score
    end[window, _token_at(features[window], context, text, last=True)] = score


@pytest.fixture
def curie(tokenizer):
    rows = [make_example("q1", "What did Curie win in 1903?", CONTEXT_CURIE, "the Nobel Prize in Physics")]
    features = _features(rows, tokenizer)
    assert len(features) == 1
    return rows, features


def test_picks_the_highest_scoring_span_and_keeps_the_original_text(curie):
    rows, features = curie
    start, end = _blank_scores(features)
    _point_at(features, start, end, 0, CONTEXT_CURIE, "Nobel Prize in Physics")

    pred = postprocess_predictions(rows, features, start, end)["q1"]
    assert pred.text == "Nobel Prize in Physics"          # capital letters come from the passage
    assert pred.best_span_score == pytest.approx(10.0)
    assert pred.null_score == pytest.approx(2 * LOW)
    assert pred.score_diff == pytest.approx(2 * LOW - 10.0)
    assert pred.candidates[0]["text"] == "Nobel Prize in Physics"
    assert CONTEXT_CURIE[pred.candidates[0]["start_char"]: pred.candidates[0]["end_char"]] == pred.text


def test_says_no_answer_when_the_cls_score_wins(curie):
    rows, features = curie
    start, end = _blank_scores(features)
    _point_at(features, start, end, 0, CONTEXT_CURIE, "1911", score=2.0)   # span score 4
    start[0, 0] = end[0, 0] = 3.0                                          # null score 6

    pred = postprocess_predictions(rows, features, start, end)["q1"]
    assert pred.text == ""
    assert pred.best_span_text == "1911", "the runner-up answer is still reported"
    assert pred.score_diff == pytest.approx(2.0)


def test_threshold_moves_the_decision(curie):
    rows, features = curie
    start, end = _blank_scores(features)
    _point_at(features, start, end, 0, CONTEXT_CURIE, "1911", score=2.0)
    start[0, 0] = end[0, 0] = 3.0                                          # null beats the span by 2

    def answer(tau):
        return postprocess_predictions(rows, features, start, end, null_threshold=tau)["q1"].text

    assert answer(0.0) == ""
    assert answer(1.9) == ""
    assert answer(2.5) == "1911"            # a higher tau makes the model answer more often

    preds = postprocess_predictions(rows, features, start, end)
    assert apply_threshold(preds, 2.5) == {"q1": "1911"}
    assert apply_threshold(preds, -50.0) == {"q1": ""}
    assert preds["q1"].text == "", "apply_threshold must not change the predictions it was given"


def test_choose_answer_boundary():
    assert choose_answer("x", score_diff=0.0, null_threshold=0.0) == "x"   # a tie goes to the answer
    assert choose_answer("x", score_diff=0.1, null_threshold=0.0) == ""


def test_never_answers_from_the_question(curie):
    rows, features = curie
    start, end = _blank_scores(features)
    # Token 1 is the first word of the question ("what"). Point at it as hard as possible.
    start[0, 1] = end[0, 1] = 50.0
    _point_at(features, start, end, 0, CONTEXT_CURIE, "Chemistry", score=1.0)

    pred = postprocess_predictions(rows, features, start, end)["q1"]
    assert pred.best_span_text == "Chemistry"
    assert all(c["text"] in CONTEXT_CURIE for c in pred.candidates)


def test_rejects_backwards_and_overlong_spans(curie):
    rows, features = curie
    feature = features[0]
    start, end = _blank_scores(features)
    # Best start is late in the passage, best end is early: together they run backwards.
    start[0, _token_at(feature, CONTEXT_CURIE, "1911")] = 9.0
    end[0, _token_at(feature, CONTEXT_CURIE, "Marie", last=True)] = 9.0
    # Second-best pair is a real 2-token span.
    start[0, _token_at(feature, CONTEXT_CURIE, "Nobel")] = 4.0
    end[0, _token_at(feature, CONTEXT_CURIE, "Prize", last=True)] = 4.0

    spans = best_spans_in_window(start[0], end[0], feature["offset_mapping"])
    assert all(s < e for _, s, e in spans)
    pred = postprocess_predictions(rows, features, start, end)["q1"]
    assert pred.best_span_text == "Nobel Prize"

    # "Marie ... Prize" is 6 tokens. With a limit of 6 it is allowed, with 5 or 3 it is not.
    start2, end2 = _blank_scores(features)
    start2[0, _token_at(feature, CONTEXT_CURIE, "Marie")] = 9.0
    end2[0, _token_at(feature, CONTEXT_CURIE, "Prize", last=True)] = 9.0
    long = postprocess_predictions(rows, features, start2, end2, max_answer_length=6)["q1"]
    assert long.best_span_text == "Marie Curie won the Nobel Prize"
    for limit in (5, 3):
        short = postprocess_predictions(rows, features, start2, end2, max_answer_length=limit)["q1"]
        assert short.best_span_text != long.best_span_text
        assert len(short.best_span_text.split()) <= limit


def test_long_passage_uses_the_best_window_and_the_lowest_null_score(tokenizer):
    context = " ".join(FILLER_WORDS[:100]) + " " + CONTEXT_CURIE
    rows = [make_example("long", "When did Curie win the Chemistry prize?", context, "1911")]
    features = _features(rows, tokenizer, max_length=64, stride=16)
    assert len(features) > 1, "the passage must need more than one window for this test"
    last = len(features) - 1

    start, end = _blank_scores(features)
    start[:, 0] = end[:, 0] = 4.0                      # every window says "no answer" (null score 8)...
    start[last, 0] = end[last, 0] = 1.0                # ...except the last one, which sees the answer
    _point_at(features, start, end, last, context, "1911", score=3.0)   # span score 6

    pred = postprocess_predictions(rows, features, start, end)["long"]
    assert pred.null_score == pytest.approx(2.0), "lowest null score over the windows"
    assert pred.text == "1911"


def test_span_seen_in_two_windows_is_listed_once(tokenizer):
    context = " ".join(FILLER_WORDS[:60])
    rows = [make_example("dup", "What is near the water?", context, None)]
    features = _features(rows, tokenizer, max_length=48, stride=24)
    word = next(w for w in FILLER_WORDS[:60]
                if sum(any(o is not None and context[o[0]:o[1]] == w for o in f["offset_mapping"])
                       for f in features[:2]) == 2)
    start, end = _blank_scores(features)
    _point_at(features, start, end, 0, context, word, score=2.0)
    _point_at(features, start, end, 1, context, word, score=3.0)

    pred = postprocess_predictions(rows, features, start, end, n_candidates=50)["dup"]
    assert [c["text"] for c in pred.candidates].count(word) == 1
    assert pred.best_span_text == word and pred.best_span_score == pytest.approx(6.0)
    scores = [c["score"] for c in pred.candidates]
    assert scores == sorted(scores, reverse=True)


def test_several_questions_are_kept_apart(tokenizer):
    rows = [
        make_example("a", "When did Curie win the Chemistry prize?", CONTEXT_CURIE, "1911"),
        make_example("b", "What did Curie win in 1903?", CONTEXT_CURIE, "the Nobel Prize in Physics"),
    ]
    features = _features(rows, tokenizer)
    start, end = _blank_scores(features)
    _point_at(features, start, end, 0, CONTEXT_CURIE, "1911")
    _point_at(features, start, end, 1, CONTEXT_CURIE, "1903")

    preds = postprocess_predictions(rows, features, start, end)
    assert {k: p.text for k, p in preds.items()} == {"a": "1911", "b": "1903"}


def test_mismatched_scores_raise(curie):
    rows, features = curie
    start, end = _blank_scores(features)
    with pytest.raises(ValueError, match="windows"):
        postprocess_predictions(rows, features, np.vstack([start, start]), end)
