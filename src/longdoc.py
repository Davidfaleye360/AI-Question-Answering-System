"""Long-document mode: answer questions about a whole article instead of one paragraph (task T10).

    python -m src.longdoc --model Ingsyling/distilbert-squad2 --name distilbert-squad2

In SQuAD every question comes with the one paragraph that holds its answer. A real user has
a whole document and does not know where the answer is. The model can read at most 384
tokens at a time, so a long document is cut into overlapping windows (the same code as in
training, src/preprocess.py), every window is scored, and the windows' scores are combined
into one answer (src/postprocess.py).

How we test it, with numbers instead of a few hand-picked questions:
  1. Take three dev articles of different lengths and glue each one's paragraphs back
     together into one document (about 3,000, 5,000 and 7,000 tokens).
  2. Ask every SQuAD question of that article against the WHOLE document. We still know the
     gold answers, so we can compute EM and F1 exactly as before.
  3. Compare with the scores the same questions got when the model was handed the right
     paragraph. The drop is the price of having to find the place as well as the answer.

Two ways of combining windows are compared (see `window_rule` in src/postprocess.py):
  * "global": best span and lowest no-answer score over all windows, as in the evaluation.
  * "per_window": the window that beats its own no-answer score by the most decides.

A caveat we report rather than hide: a question that is unanswerable from its paragraph
might be answerable from another part of the article. The gold labels were written for the
paragraph, so "unanswerable" scores in long-document mode are a lower bound.

Output: results/long_documents.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable, Sequence

from src.config import Config, load_config
from src.evaluate import make_eval_features, per_example_scores, pick_device, predict_logits, summarize
from src.postprocess import postprocess_predictions
from src.threshold import best_threshold, scores_at

# Three dev articles: short, medium and long. Fixed, so the experiment can be repeated.
DEFAULT_TITLES = ("Amazon_rainforest", "Normans", "Oxygen")
PARAGRAPH_BREAK = "\n\n"
WINDOW_RULES = ("global", "per_window")


def build_document(examples: Iterable[dict]) -> tuple[str, dict[str, int]]:
    """Join an article's paragraphs, in their original order, into one document.

    Returns (document, {paragraph text: character position where it starts in the document}).
    """
    paragraphs = list(dict.fromkeys(example["context"] for example in examples))
    starts, position = {}, 0
    for paragraph in paragraphs:
        starts[paragraph] = position
        position += len(paragraph) + len(PARAGRAPH_BREAK)
    return PARAGRAPH_BREAK.join(paragraphs), starts


def as_long_examples(examples: Sequence[dict]) -> tuple[list[dict], dict[str, tuple[int, int]]]:
    """The same questions, but each one's `context` is now the whole article.

    Also returns {question id: (start, end)}: where the question's ORIGINAL paragraph sits in
    the document, so we can check whether the model's answer came from the right place.
    """
    document, starts = build_document(examples)
    long_examples, home = [], {}
    for example in examples:
        offset = starts[example["context"]]
        answers = example["answers"]
        long_examples.append({
            **example,
            "context": document,
            "answers": {"text": list(answers["text"]),
                        "answer_start": [offset + s for s in answers["answer_start"]]},
        })
        home[example["id"]] = (offset, offset + len(example["context"]))
    return long_examples, home


def right_paragraph_rate(predictions: dict, home: dict[str, tuple[int, int]], examples: Sequence[dict]) -> float | None:
    """Of the answerable questions the model answered, the share (in %) whose span lies in the right paragraph."""
    hits = total = 0
    for example in examples:
        p = predictions[example["id"]]
        if not example["answers"]["text"] or not p.text or not p.candidates:
            continue
        start = p.candidates[0]["start_char"]   # the first candidate is always the span behind the answer
        low, high = home[example["id"]]
        hits += low <= start < high
        total += 1
    return round(100.0 * hits / total, 1) if total else None


def _records(predictions: dict, examples: Sequence[dict]) -> list[dict]:
    """Threshold-sweep records (see src/threshold.py) from in-memory predictions."""
    from src.evaluate import score_example

    records = []
    for example in examples:
        p = predictions[example["id"]]
        em, f1 = score_example(p.best_span_text, example["answers"]["text"])
        records.append({"id": example["id"], "title": example["title"], "answerable": bool(example["answers"]["text"]),
                        "score_diff": p.score_diff, "answer_em": em, "answer_f1": f1})
    return records


def evaluate_article(model, tokenizer, examples, cfg: Config, batch_size: int = 32, device=None) -> dict:
    """Score one article's questions in long-document mode, under both window rules.

    `examples` is a datasets.Dataset holding the questions of ONE article.
    """
    from datasets import Dataset

    rows = list(examples)
    long_examples, home = as_long_examples(rows)
    long_dataset = Dataset.from_list(long_examples)
    features = make_eval_features(long_dataset, tokenizer, cfg.preprocessing)

    started = time.time()
    start_logits, end_logits = predict_logits(model, features, batch_size, device or pick_device())
    seconds = time.time() - started

    post = cfg.postprocessing
    result = {
        "questions": len(rows),
        "document_characters": len(long_examples[0]["context"]),
        "paragraphs": len(set(r["context"] for r in rows)),
        "windows_per_question": round(len(features) / len(rows), 1),
        "seconds_per_question": round(seconds / len(rows), 3),
        "rules": {},
    }
    for rule in WINDOW_RULES:
        predictions = postprocess_predictions(
            long_examples, features, start_logits, end_logits, n_best=post.n_best,
            max_answer_length=post.max_answer_length, null_threshold=post.null_threshold, window_rule=rule)
        answers = {ex_id: p.text for ex_id, p in predictions.items()}
        metrics = summarize(per_example_scores(answers, long_examples))
        records = _records(predictions, long_examples)
        tau = best_threshold(records)
        result["rules"][rule] = {
            "at_tau_0": {k: round(v, 2) if isinstance(v, float) else v for k, v in metrics.items()},
            "right_paragraph_pct": right_paragraph_rate(predictions, home, long_examples),
            "answered_pct": round(scores_at(records, post.null_threshold)["answered_pct"], 1),
            # Optimistic (tuned on the same questions); it shows how far off tau = 0 is in this mode.
            "best_tau": round(tau, 2),
            "at_best_tau": {k: round(v, 2) for k, v in scores_at(records, tau).items()},
        }
    return result


def paragraph_scores(saved_predictions: dict[str, dict], examples: Sequence[dict]) -> dict:
    """The scores the same questions got with the right paragraph, from results/<name>_predictions.json."""
    scores = {ex["id"]: {"em": saved_predictions[ex["id"]]["em"], "f1": saved_predictions[ex["id"]]["f1"],
                         "answerable": bool(ex["answers"]["text"])} for ex in examples}
    return {k: round(v, 2) if isinstance(v, float) else v for k, v in summarize(scores).items()}


def print_article(title: str, article: dict) -> None:
    print(f"\n{title}: {article['questions']} questions, {article['paragraphs']} paragraphs, "
          f"{article['windows_per_question']} windows and {article['seconds_per_question']:.2f}s per question")
    print(f"{'':34}{'EM':>7}{'F1':>7}{'HasAns F1':>11}{'NoAns F1':>10}{'right paragraph':>17}")

    def row(label: str, m: dict, right: float | None = None) -> None:
        place = f"{right:.1f}%" if right is not None else ""
        print(f"{label:34}{m['exact']:7.2f}{m['f1']:7.2f}{m.get('HasAns_f1', float('nan')):11.2f}"
              f"{m.get('NoAns_f1', float('nan')):10.2f}{place:>17}")

    row("given the right paragraph", article["paragraph_mode"])
    for rule in WINDOW_RULES:
        r = article["rules"][rule]
        row(f"whole article, {rule} rule", r["at_tau_0"], r["right_paragraph_pct"])
        row(f"  ... at its best tau ({r['best_tau']:+.2f})", r["at_best_tau"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True, help="Hub id or local folder")
    parser.add_argument("--name", required=True, help="name used with src.evaluate --name (for the paragraph scores)")
    parser.add_argument("--titles", nargs="+", default=list(DEFAULT_TITLES), help="dev article titles to use")
    parser.add_argument("--config", default="configs/distilbert.yaml")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    cfg = load_config(args.config)
    results = Path(args.results_dir)
    dev = load_dataset(cfg.data.dataset_name, split="validation")
    saved = json.loads((results / f"{args.name}_predictions.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model)
    device = pick_device(args.cpu)

    articles = {}
    for title in args.titles:
        questions = dev.filter(lambda ex: ex["title"] == title)
        if len(questions) == 0:
            raise ValueError(f"No dev article is called {title!r}")
        article = evaluate_article(model, tokenizer, questions, cfg, args.batch_size, device)
        article["paragraph_mode"] = paragraph_scores(saved, list(questions))
        articles[title] = article
        print_article(title, article)

    out = {"model": args.model, "dataset": f"{cfg.data.dataset_name} (validation)", "device": str(device),
           "null_threshold": cfg.postprocessing.null_threshold, "articles": articles}
    path = results / "long_documents.json"
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print("\nSaved", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
