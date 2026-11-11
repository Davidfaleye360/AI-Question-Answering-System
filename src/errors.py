"""Error analysis: what kinds of mistakes do the models make, and on which questions? (task T8)

    python -m src.errors --models distilbert-squad2 roberta-base-squad2

One F1 number says how good a model is, not what is wrong with it. This script reads the
predictions saved by `src/evaluate.py` and answers three questions without running a model:

  1. WHAT KIND of mistake is each wrong prediction? Every dev question falls into exactly
     one of these, decided automatically from the prediction and the gold answers:
       correct          exact match
       false answer     the question is unanswerable, but the model gave a span
       missed answer    the question is answerable, but the model said "no answer"
       too long         the right answer plus extra words   ("Richard I of Normandy" for "Richard I")
       too short        only part of the right answer        ("Nobel Prize" for "Nobel Prize in Physics")
       partial overlap  shares some words with the answer, but neither contains the other
       wrong span       shares no words with any gold answer
  2. Which QUESTION WORDS are hard? F1 by who / what / when / why / ...
  3. Do LONG ANSWERS hurt? F1 by the length of the gold answer.

It also draws a fixed random sample of the wrong predictions for labelling by hand
(results/error_sample.csv): the automatic types say what happened, a person has to say why.

Output: results/error_breakdown.json and results/error_analysis.png.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from src.data import is_answerable
from src.evaluate import f1_score, normalize_answer
from src.explore import question_type

ERROR_TYPES = ("false answer", "missed answer", "too long", "too short", "partial overlap", "wrong span")
OUTCOMES = ("correct",) + ERROR_TYPES

# Gold-answer length buckets in words: (label, shortest, longest).
LENGTH_BUCKETS = (("1 word", 1, 1), ("2 words", 2, 2), ("3-4 words", 3, 4), ("5-9 words", 5, 9), ("10+ words", 10, 10**6))
NO_ANSWER_BUCKET = "no answer"


def closest_gold(prediction: str, gold_texts: Sequence[str]) -> str:
    """The gold answer the prediction overlaps with most (the one its F1 is measured against)."""
    golds = [g for g in gold_texts if normalize_answer(g)]
    return max(golds, key=lambda g: f1_score(prediction, g)) if golds else ""


def classify(prediction: str, gold_texts: Sequence[str], em: int) -> str:
    """Which of OUTCOMES one prediction is. `prediction` is "" when the model abstained."""
    if em:
        return "correct"
    answerable = any(normalize_answer(g) for g in gold_texts)
    if not answerable:
        return "false answer"            # em is 0, so the model must have answered
    if not prediction:
        return "missed answer"

    gold = closest_gold(prediction, gold_texts)
    pred_words, gold_words = Counter(normalize_answer(prediction).split()), Counter(normalize_answer(gold).split())
    shared = sum((pred_words & gold_words).values())
    if shared == 0:
        return "wrong span"
    if shared == sum(gold_words.values()):
        return "too long"                # every gold word is in the prediction, plus more
    if shared == sum(pred_words.values()):
        return "too short"               # every predicted word is in the gold answer, but not all of it
    return "partial overlap"


def length_bucket(gold_texts: Sequence[str]) -> str:
    """Bucket by the length in words of the SHORTEST gold answer (annotators often give several)."""
    lengths = [len(g.split()) for g in gold_texts if g.strip()]
    if not lengths:
        return NO_ANSWER_BUCKET
    shortest = min(lengths)
    return next(label for label, low, high in LENGTH_BUCKETS if low <= shortest <= high)


def build_rows(saved_predictions: dict[str, dict], examples: Iterable[dict]) -> list[dict]:
    """One row per dev question: the question, the answers, the scores and the labels used below."""
    rows = []
    for example in examples:
        saved = saved_predictions[example["id"]]
        golds = list(dict.fromkeys(example["answers"]["text"]))   # unique, original order
        rows.append({
            "id": example["id"],
            "title": example["title"],
            "question": example["question"].strip(),
            "context": example["context"],
            "gold": golds,
            "answerable": is_answerable(example),
            "prediction": saved["prediction"],
            "best_span": saved["best_span"],
            "score_diff": saved["score_diff"],
            "em": saved["em"],
            "f1": saved["f1"],
            "outcome": classify(saved["prediction"], golds, saved["em"]),
            "question_type": question_type(example["question"]),
            "answer_length": length_bucket(golds),
        })
    return rows


def outcome_counts(rows: Sequence[dict]) -> dict[str, int]:
    """How many questions ended in each outcome, in the fixed order of OUTCOMES."""
    counts = Counter(r["outcome"] for r in rows)
    return {outcome: counts.get(outcome, 0) for outcome in OUTCOMES}


def breakdown(rows: Sequence[dict], key: str, order: Sequence[str] | None = None) -> list[dict]:
    """EM and F1 for every value of `key` (for example "question_type"), biggest group first."""
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    names = list(order) if order else sorted(groups, key=lambda name: -len(groups[name]))
    return [
        {
            key: name,
            "questions": len(groups[name]),
            "exact": round(100 * sum(r["em"] for r in groups[name]) / len(groups[name]), 2),
            "f1": round(100 * sum(r["f1"] for r in groups[name]) / len(groups[name]), 2),
        }
        for name in names if groups.get(name)
    ]


def analyse(rows: Sequence[dict]) -> dict:
    """Everything reported for one model."""
    answerable = [r for r in rows if r["answerable"]]
    counts = outcome_counts(rows)
    errors = sum(counts[e] for e in ERROR_TYPES)
    return {
        "questions": len(rows),
        "errors": errors,
        "outcomes": counts,
        "error_shares_pct": {e: round(100 * counts[e] / errors, 1) if errors else 0.0 for e in ERROR_TYPES},
        # Question words: answerable questions only, so "did it find the span" isn't mixed with "did it abstain".
        "by_question_type_answerable": breakdown(answerable, "question_type"),
        "by_question_type_all": breakdown(rows, "question_type"),
        "by_answer_length": breakdown(answerable, "answer_length", [label for label, _, _ in LENGTH_BUCKETS]),
    }


def draw_sample(rows: Sequence[dict], n: int = 60, seed: int = 42) -> list[dict]:
    """A fixed random sample of wrong predictions, with every error type represented.

    Each type gets a share of the sample in proportion to how common it is, but at least 4,
    so rare types are not left out. The same seed always gives the same sample.
    """
    rng = random.Random(seed)
    by_type = {e: [r for r in rows if r["outcome"] == e] for e in ERROR_TYPES}
    total = sum(len(v) for v in by_type.values())
    quota = {e: min(len(v), max(4, round(n * len(v) / total))) for e, v in by_type.items() if v}
    # Rounding and the minimum can overshoot n: take the surplus from the largest types.
    while sum(quota.values()) > n:
        quota[max(quota, key=quota.get)] -= 1
    sample = []
    for e in ERROR_TYPES:
        if e in quota:
            sample.extend(rng.sample(sorted(by_type[e], key=lambda r: r["id"]), quota[e]))
    return sample


def snippet(context: str, text: str, width: int = 160) -> str:
    """The part of the passage around `text`, so a labeller doesn't have to read all of it."""
    position = context.find(text) if text else -1
    if position < 0:
        return ""
    start, end = max(0, position - width), min(len(context), position + len(text) + width)
    return ("..." if start else "") + context[start:end].replace("\n", " ") + ("..." if end < len(context) else "")


SAMPLE_COLUMNS = ("id", "error_type", "question", "correct_answers", "prediction", "best_span_if_abstained",
                  "score_diff", "passage_near_answer", "passage_near_prediction", "label", "note")


def write_sample_csv(sample: Sequence[dict], path: str | Path, labels: dict[str, dict] | None = None) -> None:
    """Write the sample for labelling. `labels` maps id -> {"label", "note"} to keep earlier work."""
    labels = labels or {}
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SAMPLE_COLUMNS)
        writer.writeheader()
        for r in sample:
            shown = r["prediction"] or r["best_span"]
            done = labels.get(r["id"], {})
            writer.writerow({
                "id": r["id"],
                "error_type": r["outcome"],
                "question": r["question"],
                "correct_answers": " | ".join(r["gold"]) or "(no answer)",
                "prediction": r["prediction"] or "(no answer)",
                "best_span_if_abstained": "" if r["prediction"] else r["best_span"],
                "score_diff": r["score_diff"],
                "passage_near_answer": snippet(r["context"], r["gold"][0]) if r["gold"] else "",
                "passage_near_prediction": snippet(r["context"], shown),
                "label": done.get("label", ""),
                "note": done.get("note", ""),
            })


def read_sample_labels(path: str | Path) -> dict[str, dict]:
    """{id: {"label", "note"}} from an existing sample file ({} if there is none)."""
    if not Path(path).exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {row["id"]: {k: row.get(k, "") for k in ("label", "note")} for row in csv.DictReader(f)}


def label_summary(path: str | Path) -> dict:
    """Counts of the hand-written labels in the sample file, overall and inside each error type."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row["label"].strip()]
    by_type = defaultdict(Counter)
    for row in rows:
        by_type[row["error_type"]][row["label"]] += 1
    return {
        "labelled": len(rows),
        "labels": dict(Counter(row["label"] for row in rows).most_common()),
        "by_error_type": {e: dict(by_type[e].most_common()) for e in ERROR_TYPES if e in by_type},
    }


# --------------------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------------------

SERIES_COLORS = ["#2a78d6", "#eb6834"]   # our model is always blue, the baseline always orange
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3de"


def plot_analysis(analyses: dict[str, dict], labels: dict[str, str], path: str | Path) -> None:
    """Three panels: error types (counts), F1 by question word, F1 by answer length. One bar colour per model."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(analyses)
    first = analyses[names[0]]
    question_types = [row["question_type"] for row in first["by_question_type_answerable"] if row["questions"] >= 50]
    lengths = [label for label, _, _ in LENGTH_BUCKETS]
    panels = [
        ("Errors by type (dev questions)", list(ERROR_TYPES),
         lambda a, name: a["outcomes"][name], False),
        ("F1 by question word (answerable)", question_types,
         lambda a, name: next((r["f1"] for r in a["by_question_type_answerable"] if r["question_type"] == name), 0), True),
        ("F1 by gold answer length (answerable)", lengths,
         lambda a, name: next((r["f1"] for r in a["by_answer_length"] if r["answer_length"] == name), 0), True),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1.15, 1.5, 1]})
    height = 0.38
    for ax, (title, categories, value, is_f1) in zip(axes, panels):
        for i, (name, color) in enumerate(zip(names, SERIES_COLORS)):
            values = [value(analyses[name], c) for c in categories]
            positions = [k + (i - (len(names) - 1) / 2) * (height + 0.04) for k in range(len(categories))]
            bars = ax.barh(positions, values, height=height, color=color, label=labels.get(name, name))
            for bar, v in zip(bars, values):
                ax.annotate(f"{v:.0f}" if is_f1 else f"{v:,}", xy=(v, bar.get_y() + bar.get_height() / 2),
                            xytext=(3, 0), textcoords="offset points", va="center", fontsize=8, color=MUTED)
        ax.set_yticks(range(len(categories)), categories)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        if is_f1:
            ax.set_xlim(0, 100)
        else:
            ax.set_xlim(0, ax.get_xlim()[1] * 1.12)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9, length=0)
        for tick in ax.get_yticklabels():
            tick.set_color(INK)
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def print_report(label: str, analysis: dict) -> None:
    print(f"\n{label}: {analysis['errors']:,} wrong out of {analysis['questions']:,}")
    for e in ERROR_TYPES:
        print(f"  {e:16}{analysis['outcomes'][e]:6,}  {analysis['error_shares_pct'][e]:5.1f}% of errors")
    print("  F1 by question word, answerable questions:")
    print("   " + "  ".join(f"{r['question_type']} {r['f1']:.1f}" for r in analysis["by_question_type_answerable"]))
    print("  F1 by gold answer length:")
    print("   " + "  ".join(f"{r['answer_length']} {r['f1']:.1f}" for r in analysis["by_answer_length"]))


def load_rows(name: str, dev, results: Path) -> list[dict]:
    saved = json.loads((results / f"{name}_predictions.json").read_text(encoding="utf-8"))
    return build_rows(saved, dev)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--models", nargs="+", required=True,
                        help="names used with src.evaluate --name; the sample is drawn from the first one")
    parser.add_argument("--labels", nargs="*", default=None, help="display names, same order")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--config", default="configs/distilbert.yaml")
    parser.add_argument("--sample-size", type=int, default=60)
    args = parser.parse_args()

    from datasets import load_dataset

    from src.config import load_config

    cfg = load_config(args.config)
    results = Path(args.results_dir)
    dev = load_dataset(cfg.data.dataset_name, split="validation")
    labels = dict(zip(args.models, args.labels or args.models))

    analyses, rows_of = {}, {}
    for name in args.models:
        rows_of[name] = load_rows(name, dev, results)
        analyses[name] = analyse(rows_of[name])
        print_report(labels[name], analyses[name])

    (results / "error_breakdown.json").write_text(
        json.dumps({"dataset": f"{cfg.data.dataset_name} (validation)", "models": analyses}, indent=1) + "\n",
        encoding="utf-8")
    plot_analysis(analyses, labels, results / "error_analysis.png")

    # The sample keeps any labels already written into the file, so re-running doesn't erase hand work.
    sample_path = results / "error_sample.csv"
    sample = draw_sample(rows_of[args.models[0]], args.sample_size, cfg.training.seed)
    write_sample_csv(sample, sample_path, read_sample_labels(sample_path))
    print(f"\nSaved {results / 'error_breakdown.json'}, {results / 'error_analysis.png'} "
          f"and {sample_path} ({len(sample)} wrong predictions of {labels[args.models[0]]} to label)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
