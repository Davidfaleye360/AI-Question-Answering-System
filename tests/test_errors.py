"""Tests for src/errors.py. No model or internet: predictions are written by hand."""
from __future__ import annotations

import pytest

from conftest import CONTEXT_CURIE, CONTEXT_FAME, make_example
from src.errors import (
    ERROR_TYPES,
    OUTCOMES,
    analyse,
    breakdown,
    build_rows,
    classify,
    closest_gold,
    draw_sample,
    label_summary,
    length_bucket,
    outcome_counts,
    plot_analysis,
    read_sample_labels,
    snippet,
    write_sample_csv,
)


@pytest.mark.parametrize("prediction,golds,em,expected", [
    ("Richard I", ["Richard I"], 1, "correct"),
    ("", [], 1, "correct"),                                             # correctly abstained
    ("Normandy", [], 0, "false answer"),
    ("", ["Rollo"], 0, "missed answer"),
    ("Richard I of Normandy", ["Richard I"], 0, "too long"),
    ("Nobel Prize", ["the Nobel Prize in Physics"], 0, "too short"),
    ("Prize in Chemistry", ["the Nobel Prize in Physics"], 0, "partial overlap"),
    ("Christian", ["Catholic", "Catholic orthodoxy"], 0, "wrong span"),
])
def test_classify(prediction, golds, em, expected):
    assert classify(prediction, golds, em) == expected


def test_classify_compares_with_the_closest_gold_answer():
    golds = ["in the late 1990s", "1990s"]
    assert closest_gold("the late 1990s decade", golds) == "in the late 1990s"
    # Against "1990s" this would be "too long"; against the closer gold it only partly overlaps.
    assert classify("the late 1990s decade", golds, 0) == "partial overlap"
    assert classify("late 1990s era", ["1990s"], 0) == "too long"


def test_classify_ignores_case_punctuation_and_articles_like_the_metric():
    assert classify("The Nobel Prize, in physics today", ["Nobel Prize in Physics"], 0) == "too long"


@pytest.mark.parametrize("golds,expected", [
    (["1911"], "1 word"),
    (["Nobel Prize"], "2 words"),
    (["the Nobel Prize in Physics", "Nobel Prize in Physics"], "3-4 words"),   # shortest gold counts
    (["a b c d e f g h i j k"], "10+ words"),
    ([], "no answer"),
])
def test_length_bucket(golds, expected):
    assert length_bucket(golds) == expected


def _examples():
    return [
        make_example("a", "When did Curie win the Chemistry prize?", CONTEXT_CURIE, "1911"),
        make_example("b", "What did Curie win in 1903?", CONTEXT_CURIE, "the Nobel Prize in Physics"),
        make_example("c", "Where was Curie born?", CONTEXT_CURIE, None),
        make_example("d", "When did she rise to fame?", CONTEXT_FAME, "in the late 1990s"),
        make_example("e", "Who was lead singer?", CONTEXT_FAME, None),
    ]


def _saved():
    def p(prediction, best_span, em, f1, score_diff=-1.0):
        return {"prediction": prediction, "best_span": best_span, "em": em, "f1": f1, "score_diff": score_diff}
    return {
        "a": p("1911", "1911", 1, 1.0),
        "b": p("Nobel Prize", "Nobel Prize", 0, 0.6667),
        "c": p("", "Chemistry", 1, 1.0, 2.0),
        "d": p("", "the late 1990s", 0, 0.0, 0.5),
        "e": p("She", "She", 0, 0.0),
    }


def test_build_rows_and_counts():
    rows = build_rows(_saved(), _examples())
    assert [r["outcome"] for r in rows] == ["correct", "too short", "correct", "missed answer", "false answer"]
    assert [r["question_type"] for r in rows] == ["when", "what", "where", "when", "who"]
    assert [r["answer_length"] for r in rows] == ["1 word", "5-9 words", "no answer", "3-4 words", "no answer"]
    counts = outcome_counts(rows)
    assert list(counts) == list(OUTCOMES) and sum(counts.values()) == 5
    assert counts["correct"] == 2 and counts["wrong span"] == 0


def test_breakdown_and_analyse():
    rows = build_rows(_saved(), _examples())
    by_type = {r["question_type"]: r for r in breakdown(rows, "question_type")}
    assert by_type["when"] == {"question_type": "when", "questions": 2, "exact": 50.0, "f1": 50.0}
    assert breakdown(rows, "question_type")[0]["question_type"] == "when", "biggest group first"

    a = analyse(rows)
    assert a["questions"] == 5 and a["errors"] == 3
    assert sum(a["error_shares_pct"].values()) == pytest.approx(100.0, abs=0.2)
    assert {r["question_type"] for r in a["by_question_type_answerable"]} == {"when", "what"}
    assert [r["answer_length"] for r in a["by_answer_length"]] == ["1 word", "3-4 words", "5-9 words"]
    assert all("no answer" != r["answer_length"] for r in a["by_answer_length"])


def _many_rows():
    rows = []
    for i in range(300):
        outcome = ("false answer" if i % 2 else "missed answer") if i >= 10 else ERROR_TYPES[2 + i % 4]
        rows.append({"id": f"{i:04d}", "outcome": outcome})
    return rows + [{"id": f"ok{i}", "outcome": "correct"} for i in range(50)]


def test_sample_is_repeatable_covers_every_type_and_has_only_errors():
    rows = _many_rows()
    sample = draw_sample(rows, n=60, seed=42)
    assert len(sample) == 60 and len({r["id"] for r in sample}) == 60
    assert all(r["outcome"] != "correct" for r in sample)
    assert {r["outcome"] for r in sample} == set(ERROR_TYPES), "rare types are not left out"
    assert [r["id"] for r in sample] == [r["id"] for r in draw_sample(rows[::-1], n=60, seed=42)]
    assert [r["id"] for r in sample] != [r["id"] for r in draw_sample(rows, n=60, seed=7)]


def test_sample_never_asks_for_more_than_there_is():
    rows = [{"id": "1", "outcome": "too long"}, {"id": "2", "outcome": "false answer"}]
    assert len(draw_sample(rows, n=60)) == 2


def test_snippet_shows_the_text_around_the_answer():
    text = "x" * 500 + " the answer " + "y" * 500
    s = snippet(text, "the answer", width=20)
    assert "the answer" in s and s.startswith("...") and s.endswith("...") and len(s) < 70
    assert snippet(text, "not there") == "" and snippet(text, "") == ""
    assert snippet(CONTEXT_CURIE, "Marie", width=5).startswith("Marie")


def test_sample_file_keeps_labels_when_written_again(tmp_path):
    rows = [r for r in build_rows(_saved(), _examples()) if r["outcome"] != "correct"]
    path = tmp_path / "sample.csv"
    assert read_sample_labels(path) == {}

    write_sample_csv(rows, path, {"b": {"label": "boundary", "note": "dropped 'in Physics'"}})
    labels = read_sample_labels(path)
    assert labels["b"] == {"label": "boundary", "note": "dropped 'in Physics'"}
    assert labels["d"]["label"] == ""

    labels["d"] = {"label": "over-cautious", "note": "corrected by hand"}
    write_sample_csv(rows, path, labels)
    again = read_sample_labels(path)
    assert again["b"]["label"] == "boundary" and again["d"]["note"] == "corrected by hand"
    summary = label_summary(path)
    assert summary["labelled"] == 2
    assert summary["labels"] == {"boundary": 1, "over-cautious": 1}
    assert summary["by_error_type"] == {"missed answer": {"over-cautious": 1}, "too short": {"boundary": 1}}
    text = path.read_text(encoding="utf-8")
    assert "(no answer)" in text and "the late 1990s" in text, "an abstention shows the rejected best span"


def test_figure_is_written(tmp_path):
    pytest.importorskip("matplotlib")
    analysis = analyse(build_rows(_saved(), _examples()) * 30)
    path = tmp_path / "errors.png"
    plot_analysis({"ours": analysis, "baseline": analysis}, {"ours": "Ours"}, path)
    assert path.stat().st_size > 5000
