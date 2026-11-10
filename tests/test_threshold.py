"""Tests for src/threshold.py. No model or internet: the records are written by hand."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import CONTEXT_CURIE, make_example
from src.threshold import (
    analyse_model,
    assign_folds,
    best_threshold,
    build_records,
    cross_validate,
    plot_sweep,
    scores_at,
    sweep,
)


def rec(score_diff, answerable, answer_f1=0.0, answer_em=None, title="A", ex_id=None):
    """One question: its score_diff, whether it has an answer, and its score if the model answers."""
    answer_em = int(answer_f1 == 1.0) if answer_em is None else answer_em
    return {"id": ex_id or f"q{score_diff}", "title": title, "answerable": answerable,
            "score_diff": score_diff, "answer_em": answer_em, "answer_f1": answer_f1}


# Four questions, sorted by how keen the model is to answer:
#   -5: answerable, best span right        -1: answerable, best span half right
#   +2: unanswerable                        +6: answerable, best span right (but the model is unsure)
RECORDS = [rec(-5, True, 1.0), rec(-1, True, 0.5), rec(2, False), rec(6, True, 1.0)]


def test_scores_at_the_default_threshold():
    s = scores_at(RECORDS, 0.0)
    # Answers -5 (1.0) and -1 (0.5); abstains on +2 (correct, 1.0) and +6 (wrong, 0).
    assert s["f1"] == pytest.approx(100 * 2.5 / 4)
    assert s["exact"] == pytest.approx(100 * 2 / 4)
    assert s["HasAns_f1"] == pytest.approx(100 * 1.5 / 3)
    assert s["NoAns_f1"] == pytest.approx(100.0)
    assert s["answered_pct"] == pytest.approx(50.0)


def test_extreme_thresholds_mean_always_or_never_answering():
    never, always = scores_at(RECORDS, -100.0), scores_at(RECORDS, 100.0)
    assert never["answered_pct"] == 0.0 and never["HasAns_f1"] == 0.0 and never["NoAns_f1"] == 100.0
    assert never["f1"] == pytest.approx(25.0)            # only the unanswerable question is right
    assert always["answered_pct"] == 100.0 and always["NoAns_f1"] == 0.0
    assert always["HasAns_f1"] == pytest.approx(100 * 2.5 / 3)
    assert always["f1"] == pytest.approx(100 * 2.5 / 4)


def test_a_question_exactly_at_the_threshold_is_answered():
    # Same rule as postprocess.choose_answer: "no answer" only when score_diff > tau.
    assert scores_at(RECORDS, 2.0)["answered_pct"] == pytest.approx(75.0)
    assert scores_at(RECORDS, 1.999)["answered_pct"] == pytest.approx(50.0)


def test_sweep_matches_deciding_question_by_question():
    rng = np.random.default_rng(0)
    records = [rec(float(d), bool(a), float(f) if a else 0.0, ex_id=str(i))
               for i, (d, a, f) in enumerate(zip(rng.normal(0, 3, 200), rng.random(200) < 0.5,
                                                 rng.choice([0.0, 0.4, 1.0], 200)))]
    taus = [-4.0, -0.5, 0.0, 1.3, 5.0]
    fast = sweep(records, taus)
    for i, tau in enumerate(taus):
        total = sum(r["answer_f1"] if r["score_diff"] <= tau else float(not r["answerable"]) for r in records)
        assert fast["f1"][i] == pytest.approx(100 * total / len(records))


def test_sweep_does_not_depend_on_the_order_of_the_records():
    assert scores_at(RECORDS[::-1], 0.0) == scores_at(RECORDS, 0.0)


def test_best_threshold_finds_the_peak():
    # The model is right about its unsure question (+6) and there is nothing between +2 and +6
    # worth abstaining on... except the unanswerable +2. Best: answer -5 and -1, abstain on +2,
    # and that is all tau can do, because answering +6 means answering +2 as well.
    #   tau in (-1, 2): F1 = (1 + 0.5 + 1 + 0) / 4 = 62.5
    #   tau >= 6:       F1 = (1 + 0.5 + 0 + 1) / 4 = 62.5   (a tie: the one closer to 0 wins)
    tau = best_threshold(RECORDS)
    assert -1 < tau < 2
    assert scores_at(RECORDS, tau)["f1"] == pytest.approx(62.5)

    # Make the unsure question worth more than the unanswerable one costs: now answering all wins.
    records = RECORDS + [rec(7, True, 1.0)]
    tau = best_threshold(records)
    assert tau > 7 and scores_at(records, tau)["answered_pct"] == 100.0


def test_best_threshold_beats_or_equals_every_grid_value():
    rng = np.random.default_rng(1)
    records = [rec(float(d), bool(a), 1.0 if a and d < 1 else 0.0, ex_id=str(i))
               for i, (d, a) in enumerate(zip(rng.normal(0, 3, 300), rng.random(300) < 0.5))]
    best = scores_at(records, best_threshold(records))["f1"]
    assert best >= sweep(records, np.arange(-10, 10, 0.1))["f1"].max() - 1e-9


def test_folds_split_by_article_and_are_repeatable():
    titles = [f"Article {i}" for i in range(10)] * 3
    folds = assign_folds(titles)
    assert folds == assign_folds(reversed(titles)), "same split whatever order the questions come in"
    assert sorted(folds.values()).count(0) == 5 and sorted(folds.values()).count(1) == 5
    assert assign_folds(titles, seed=1) != folds


def test_cross_validation_tunes_on_one_half_and_scores_the_other():
    # Article A wants a high threshold (its unsure answers are right), article B a low one.
    a = [rec(3, True, 1.0, title="A", ex_id="a1"), rec(4, True, 1.0, title="A", ex_id="a2"),
         rec(-3, True, 1.0, title="A", ex_id="a3")]
    b = [rec(3, False, title="B", ex_id="b1"), rec(4, False, title="B", ex_id="b2"),
         rec(-3, True, 1.0, title="B", ex_id="b3")]
    result = cross_validate(a + b)
    fold_of = assign_folds(["A", "B"])
    tau_for_a, tau_for_b = result["fold_taus"][fold_of["A"]], result["fold_taus"][fold_of["B"]]
    assert tau_for_a < 3, "A is scored with the low threshold learned from B"
    assert tau_for_b > 4, "B is scored with the high threshold learned from A"
    # So each half gets the WRONG threshold: A abstains on 2 right answers, B answers 2 unanswerable ones.
    assert result["scores"]["f1"] == pytest.approx(100 * 2 / 6)
    # Tuning on everything at once looks better. That gap is why we report the cross-validated number.
    assert scores_at(a + b, best_threshold(a + b))["f1"] > result["scores"]["f1"]


def test_build_records_scores_the_best_span_even_when_the_model_abstained():
    examples = [
        make_example("a", "When did Curie win the Chemistry prize?", CONTEXT_CURIE, "1911"),
        make_example("b", "Where was Curie born?", CONTEXT_CURIE, None),
    ]
    saved = {
        "a": {"prediction": "", "best_span": "in 1911", "score_diff": 0.4, "em": 0, "f1": 0.0},
        "b": {"prediction": "", "best_span": "Chemistry", "score_diff": 3.0, "em": 1, "f1": 1.0},
    }
    a, b = build_records(saved, examples)
    assert (a["answerable"], a["answer_em"], a["score_diff"]) == (True, 0, 0.4)
    assert a["answer_f1"] == pytest.approx(2 / 3)        # "in 1911" vs "1911"
    assert (b["answerable"], b["answer_em"], b["answer_f1"]) == (False, 0, 0.0)
    assert a["title"] == "Test"
    # Raising tau above 0.4 recovers the partial credit on question a.
    assert scores_at([a, b], 1.0)["f1"] > scores_at([a, b], 0.0)["f1"]


def test_analyse_model_and_figure(tmp_path):
    records = RECORDS + [rec(1, True, 1.0, title="B"), rec(-2, False, title="B"), rec(3, False, title="C")]
    grid = np.arange(-8, 8.01, 0.5)
    analysis = analyse_model(records, grid)
    assert analysis["questions"] == 7
    assert analysis["at_best_tau"]["f1"] >= analysis["at_tau_0"]["f1"]
    assert len(analysis["curve"]["tau"]) == len(analysis["curve"]["f1"]) == len(grid)
    assert len(analysis["cross_validated"]["fold_taus"]) == 2

    pytest.importorskip("matplotlib")
    path = tmp_path / "sweep.png"
    plot_sweep({"ours": analysis, "baseline": analysis}, {"ours": "Ours"}, path)
    assert path.stat().st_size > 5000
