"""Tune the no-answer threshold tau and plot F1 against it (task T7).

    python -m src.threshold --models distilbert-squad2 roberta-base-squad2

The model answers a question when
    score_diff = null score - best span score   is at most tau,
and says "no answer" otherwise (see src/postprocess.py). We trained and scored with tau = 0.
Nothing says 0 is the best choice: a higher tau makes the model answer more often, which helps
answerable questions and hurts unanswerable ones, and a lower tau does the opposite.

This script needs no model. `src/evaluate.py` saved every question's best span and
score_diff in results/<name>_predictions.json, so trying another tau is just re-deciding.

Three numbers are reported for each model:
  * F1 at tau = 0, the number in the results table.
  * F1 at the best tau found on the whole dev set. This is OPTIMISTIC: the threshold was
    picked using the same questions it is scored on.
  * Cross-validated F1, the honest one. The dev set is split in two halves by Wikipedia
    article; tau is picked on one half and scored on the other, and the other way round.

Output: results/threshold_sweep.json and results/threshold_sweep.png.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from src.data import is_answerable
from src.evaluate import score_example

METRICS = ("exact", "f1", "HasAns_exact", "HasAns_f1", "NoAns_exact", "NoAns_f1", "answered_pct")


def build_records(saved_predictions: dict[str, dict], examples: Iterable[dict]) -> list[dict]:
    """One record per question with everything a threshold decision needs.

    answer_em / answer_f1: the score IF the model gives its best span.
    The score if it abstains needs no lookup: 1 for an unanswerable question, 0 otherwise.
    """
    records = []
    for example in examples:
        saved = saved_predictions[example["id"]]
        em, f1 = score_example(saved["best_span"], example["answers"]["text"])
        records.append({
            "id": example["id"],
            "title": example["title"],
            "answerable": is_answerable(example),
            "score_diff": float(saved["score_diff"]),
            "answer_em": em,
            "answer_f1": f1,
        })
    return records


def sweep(records: Sequence[dict], taus: Sequence[float]) -> dict[str, np.ndarray]:
    """Scores (0-100) at every tau in `taus`. Returns {"tau": ..., "f1": ..., "HasAns_f1": ..., ...}.

    How it works: sort the questions by score_diff. At a given tau the model answers the
    first k of them (those with score_diff <= tau) and abstains on the rest. Running totals
    of the scores then give the result for every k without looping over the questions again.
    """
    taus = np.asarray(taus, dtype=float)
    order = np.argsort([r["score_diff"] for r in records], kind="stable")
    diffs = np.array([records[i]["score_diff"] for i in order], dtype=float)
    has = np.array([records[i]["answerable"] for i in order], dtype=float)
    em = np.array([records[i]["answer_em"] for i in order], dtype=float)
    f1 = np.array([records[i]["answer_f1"] for i in order], dtype=float)
    no = 1.0 - has
    n, n_has, n_no = len(diffs), has.sum(), no.sum()

    def running_total(values: np.ndarray) -> np.ndarray:
        return np.concatenate([[0.0], np.cumsum(values)])   # total[k] = sum of the first k values

    k = np.searchsorted(diffs, taus, side="right")           # how many questions are answered at each tau
    # Unanswerable questions among the abstained ones each score 1.
    correct_abstentions = n_no - running_total(no)[k]

    def percent(points: np.ndarray, count: float) -> np.ndarray:
        return 100.0 * points / count if count else np.full(len(taus), np.nan)

    return {
        "tau": taus,
        "exact": percent(running_total(em)[k] + correct_abstentions, n),
        "f1": percent(running_total(f1)[k] + correct_abstentions, n),
        "HasAns_exact": percent(running_total(em * has)[k], n_has),
        "HasAns_f1": percent(running_total(f1 * has)[k], n_has),
        "NoAns_exact": percent(running_total(em * no)[k] + correct_abstentions, n_no),
        "NoAns_f1": percent(running_total(f1 * no)[k] + correct_abstentions, n_no),
        "answered_pct": percent(k.astype(float), n),
    }


def scores_at(records: Sequence[dict], tau: float) -> dict[str, float]:
    """The scores at one tau, as plain numbers."""
    result = sweep(records, [tau])
    return {name: float(result[name][0]) for name in METRICS}


def best_threshold(records: Sequence[dict], metric: str = "f1") -> float:
    """The tau with the highest `metric` on these questions.

    The score only changes when tau crosses some question's score_diff, so it is enough to
    try one tau between every two neighbouring score_diffs (plus one below and one above
    all of them). Among equally good taus we take the one closest to 0, the default.
    """
    diffs = np.unique([r["score_diff"] for r in records])
    candidates = np.concatenate([[diffs[0] - 1.0], (diffs[:-1] + diffs[1:]) / 2.0, [diffs[-1] + 1.0]])
    scores = sweep(records, candidates)[metric]
    best = np.flatnonzero(np.isclose(scores, np.nanmax(scores)))
    return float(candidates[best[np.argmin(np.abs(candidates[best]))]])


def assign_folds(titles: Iterable[str], n_folds: int = 2, seed: int = 42) -> dict[str, int]:
    """{article title: fold number}, shuffled with a fixed seed.

    We split by ARTICLE, not by question: questions about the same passage are alike, and
    putting them on both sides would make the tuned threshold look better than it is.
    """
    unique = sorted(set(titles))
    random.Random(seed).shuffle(unique)
    return {title: i % n_folds for i, title in enumerate(unique)}


def cross_validate(records: Sequence[dict], n_folds: int = 2, seed: int = 42, metric: str = "f1") -> dict:
    """Pick tau on the other fold(s), score on this fold, and combine the folds.

    Returns the combined scores (every question is scored exactly once, with a tau that
    was chosen without looking at it) and the tau picked for each fold.
    """
    fold_of = assign_folds((r["title"] for r in records), n_folds, seed)
    taus, decided = [], []
    for fold in range(n_folds):
        tune = [r for r in records if fold_of[r["title"]] != fold]
        test = [r for r in records if fold_of[r["title"]] == fold]
        tau = best_threshold(tune, metric)
        taus.append(tau)
        # Re-express each test question relative to its fold's tau, so that one sweep at 0
        # over all of them applies the right threshold to each.
        decided.extend({**r, "score_diff": r["score_diff"] - tau} for r in test)
    return {"fold_taus": taus, "scores": scores_at(decided, 0.0)}


def analyse_model(records: Sequence[dict], grid: Sequence[float]) -> dict:
    """Everything we report for one model."""
    tau_best = best_threshold(records)
    curve = sweep(records, grid)
    return {
        "questions": len(records),
        "at_tau_0": scores_at(records, 0.0),
        "best_tau": tau_best,
        "at_best_tau": scores_at(records, tau_best),          # optimistic: tuned and scored on the same questions
        "cross_validated": cross_validate(records),            # honest: tuned on one half, scored on the other
        "curve": {name: [round(float(v), 3) for v in values] for name, values in curve.items()},
    }


# --------------------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------------------

# Colours follow the entity: our model is always blue, the baseline always orange.
SERIES_COLORS = ["#2a78d6", "#eb6834"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3de"


def plot_sweep(analyses: dict[str, dict], labels: dict[str, str], path: str | Path) -> None:
    """Left: F1 against tau for every model. Right: the trade-off inside the first model."""
    import matplotlib

    matplotlib.use("Agg")   # draw to a file; no window needed
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

    for i, ((name, analysis), color) in enumerate(zip(analyses.items(), SERIES_COLORS)):
        curve, label = analysis["curve"], labels.get(name, name)
        left.plot(curve["tau"], curve["f1"], color=color, linewidth=2)
        best_tau, best_f1 = analysis["best_tau"], analysis["at_best_tau"]["f1"]
        left.plot([0], [analysis["at_tau_0"]["f1"]], marker="o", markersize=8, color=color,
                  markeredgecolor="white", markeredgewidth=2, linestyle="none")
        left.plot([best_tau], [best_f1], marker="D", markersize=8, color=color,
                  markeredgecolor="white", markeredgewidth=2, linestyle="none")
        left.annotate(f"{label}\nbest {best_f1:.1f} at τ = {best_tau:+.1f}", xy=(best_tau, best_f1),
                      # First model's label goes under its curve, the next one's above, so they can't collide.
                      xytext=(0, -12 if i == 0 else 10), textcoords="offset points", ha="center",
                      va="top" if i == 0 else "bottom", fontsize=9, color=INK)
    left.plot([], [], marker="o", color=MUTED, linestyle="none", markersize=7, label="τ = 0 (default)")
    left.plot([], [], marker="D", color=MUTED, linestyle="none", markersize=7, label="best τ")
    left.legend(frameon=False, fontsize=9, loc="lower center", ncol=2)
    left.set_title("F1 on all dev questions", fontsize=11, color=INK, loc="left")
    left.set_ylabel("F1")

    first_name = next(iter(analyses))
    curve = analyses[first_name]["curve"]
    right.plot(curve["tau"], curve["HasAns_f1"], color=SERIES_COLORS[0], linewidth=2, label="answerable questions")
    right.plot(curve["tau"], curve["NoAns_f1"], color=SERIES_COLORS[0], linewidth=2, linestyle=(0, (4, 2)),
               label="unanswerable questions")
    right.legend(frameon=False, fontsize=9, loc="lower center", ncol=2)
    right.set_title(f"{labels.get(first_name, first_name)}: the trade-off behind the curve",
                    fontsize=11, color=INK, loc="left")

    for ax in (left, right):
        ax.axvline(0, color=GRID, linewidth=1, zorder=0)
        ax.set_xlabel("no-answer threshold τ   (higher = answers more often)")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)

    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def print_report(name: str, analysis: dict) -> None:
    zero, best, cv = analysis["at_tau_0"], analysis["at_best_tau"], analysis["cross_validated"]
    print(f"\n{name} ({analysis['questions']:,} questions)")
    print(f"{'':34}{'tau':>13}{'EM':>8}{'F1':>8}{'HasAns F1':>11}{'NoAns F1':>10}{'answered':>10}")

    def row(label: str, tau: str, s: dict) -> None:
        print(f"{label:34}{tau:>13}{s['exact']:8.2f}{s['f1']:8.2f}{s['HasAns_f1']:11.2f}"
              f"{s['NoAns_f1']:10.2f}{s['answered_pct']:9.1f}%")

    row("default", "0.00", zero)
    row("best on the whole dev set", f"{analysis['best_tau']:.2f}", best)
    row("cross-validated (honest)", "/".join(f"{t:.2f}" for t in cv["fold_taus"]), cv["scores"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--models", nargs="+", required=True,
                        help="names used with src.evaluate --name; the first one is 'our' model in the figure")
    parser.add_argument("--labels", nargs="*", default=None, help="display names for the figure, same order")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--config", default="configs/distilbert.yaml")
    parser.add_argument("--tau-min", type=float, default=-8.0)
    parser.add_argument("--tau-max", type=float, default=8.0)
    parser.add_argument("--tau-step", type=float, default=0.25)
    args = parser.parse_args()

    from datasets import load_dataset

    from src.config import load_config

    cfg = load_config(args.config)
    results = Path(args.results_dir)
    dev = load_dataset(cfg.data.dataset_name, split="validation")
    grid = np.round(np.arange(args.tau_min, args.tau_max + args.tau_step / 2, args.tau_step), 6)
    labels = dict(zip(args.models, args.labels or args.models))

    analyses = {}
    for name in args.models:
        saved = json.loads((results / f"{name}_predictions.json").read_text(encoding="utf-8"))
        reported = json.loads((results / f"{name}.json").read_text(encoding="utf-8"))["metrics"]
        analysis = analyse_model(build_records(saved, dev), grid)
        # Safety check: at tau = 0 the sweep must give the scores src.evaluate reported.
        for key in ("exact", "f1"):
            if abs(analysis["at_tau_0"][key] - reported[key]) > 0.02:
                raise RuntimeError(f"{name}: sweep gives {key} {analysis['at_tau_0'][key]:.2f} at tau 0, "
                                   f"but results/{name}.json says {reported[key]}")
        analyses[name] = analysis
        print_report(labels[name], analysis)

    out = {"dataset": f"{cfg.data.dataset_name} (validation)", "tau_grid_step": args.tau_step, "models": analyses}
    (results / "threshold_sweep.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    plot_sweep(analyses, labels, results / "threshold_sweep.png")
    print(f"\nSaved {results / 'threshold_sweep.json'} and {results / 'threshold_sweep.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
