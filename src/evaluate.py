"""Score a question-answering model on the SQuAD 2.0 dev set.

    python -m src.evaluate --model deepset/roberta-base-squad2 --name roberta-base-squad2
    python -m src.evaluate --model outputs/distilbert-squad2 --name distilbert-squad2
    python -m src.evaluate --model deepset/roberta-base-squad2 --limit 20 --show 20   # quick look

The two numbers everyone reports for SQuAD:
  * EM (exact match): 1 if the prediction equals a gold answer, else 0.
  * F1: overlap between the words of the prediction and the words of a gold answer, so
    "Nobel Prize" against "the Nobel Prize in Physics" gets partial credit instead of 0.
Both are computed after "normalising" the text (lower case, no punctuation, no a/an/the),
and a question with several gold answers gets the score of the best-matching one.
For an unanswerable question the only correct prediction is the empty string.

Each number is also reported separately for answerable questions (HasAns) and unanswerable
ones (NoAns). That split matters: a model that always says "no answer" gets 50% EM on the
dev set, and only the split shows that it never answers anything.

We compute the scores twice:
  * with our own functions below, which also give a score PER QUESTION (needed for the
    error analysis), and
  * with the official `squad_v2` metric from the `evaluate` library, the number we report.
The script stops with an error if the two disagree.

Output: results/<name>.json (scores and settings) and results/<name>_predictions.json
(every question's prediction and score).
"""
from __future__ import annotations

import argparse
import collections
import inspect
import json
import re
import string
import time
from pathlib import Path

import numpy as np
import torch

from src.config import Config, PreprocessingConfig, load_config
from src.data import is_answerable
from src.postprocess import Prediction, postprocess_predictions
from src.preprocess import prepare_eval_features

PADDING_SCORE = -10000.0   # score given to padding positions the model was never shown


# --------------------------------------------------------------------------------------
# EM and F1, following the official SQuAD 2.0 evaluation script
# --------------------------------------------------------------------------------------

def normalize_answer(text: str) -> str:
    """Lower case, drop punctuation, drop a/an/the, squeeze spaces: "The  Cat." -> "cat"."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> int:
    return int(normalize_answer(prediction) == normalize_answer(gold))


def f1_score(prediction: str, gold: str) -> float:
    """Word-overlap F1 between a prediction and one gold answer.

    precision = shared words / words in the prediction
    recall    = shared words / words in the gold answer
    F1        = 2 * precision * recall / (precision + recall)
    """
    pred_words = normalize_answer(prediction).split()
    gold_words = normalize_answer(gold).split()
    if not pred_words or not gold_words:
        # One side is "no answer": full credit only if both are.
        return float(pred_words == gold_words)
    shared = sum((collections.Counter(pred_words) & collections.Counter(gold_words)).values())
    if shared == 0:
        return 0.0
    precision = shared / len(pred_words)
    recall = shared / len(gold_words)
    return 2 * precision * recall / (precision + recall)


def score_example(prediction: str, gold_texts: list[str]) -> tuple[int, float]:
    """(EM, F1) of one prediction against the best-matching gold answer."""
    golds = [g for g in gold_texts if normalize_answer(g)] or [""]   # no gold answers = unanswerable
    return (
        max(exact_match(prediction, g) for g in golds),
        max(f1_score(prediction, g) for g in golds),
    )


def per_example_scores(answers: dict[str, str], examples) -> dict[str, dict]:
    """{question id: {"em", "f1", "answerable"}} for every example."""
    scores = {}
    for example in examples:
        em, f1 = score_example(answers[example["id"]], example["answers"]["text"])
        scores[example["id"]] = {"em": em, "f1": f1, "answerable": is_answerable(example)}
    return scores


def summarize(scores: dict[str, dict]) -> dict:
    """Average the per-question scores, overall and for answerable / unanswerable questions.

    Uses the same names as the official metric: exact, f1, total, HasAns_*, NoAns_*.
    """
    def average(rows: list[dict], prefix: str) -> dict:
        if not rows:
            return {}
        return {
            f"{prefix}exact": 100.0 * sum(r["em"] for r in rows) / len(rows),
            f"{prefix}f1": 100.0 * sum(r["f1"] for r in rows) / len(rows),
            f"{prefix}total": len(rows),
        }

    rows = list(scores.values())
    return {
        **average(rows, ""),
        **average([r for r in rows if r["answerable"]], "HasAns_"),
        **average([r for r in rows if not r["answerable"]], "NoAns_"),
    }


def official_scores(answers: dict[str, str], examples) -> dict:
    """The same numbers from the official `squad_v2` metric (needs internet the first time)."""
    import evaluate as hf_evaluate   # the Hugging Face library, not this file

    metric = hf_evaluate.load("squad_v2")
    predictions = [
        # The metric wants a no-answer probability. We have already made the decision, so it is 1 or 0.
        {"id": ex_id, "prediction_text": text, "no_answer_probability": 0.0 if text else 1.0}
        for ex_id, text in answers.items()
    ]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    result = metric.compute(predictions=predictions, references=references)
    # best_exact / best_f1 are the metric's own threshold search; we do ours in T7.
    return {k: v for k, v in result.items() if not k.startswith("best_")}


def scores_agree(ours: dict, official: dict, tolerance: float = 0.01) -> bool:
    """True if every number the official metric reports matches ours."""
    return all(key in ours and abs(ours[key] - value) <= tolerance for key, value in official.items())


# --------------------------------------------------------------------------------------
# Running the model
# --------------------------------------------------------------------------------------

def make_eval_features(raw, tokenizer, pre: PreprocessingConfig):
    """Tokenize questions into windows that remember their question id and character offsets."""
    return raw.map(
        prepare_eval_features,
        batched=True,
        remove_columns=raw.column_names,
        fn_kwargs={"tokenizer": tokenizer, "max_length": pre.max_length, "stride": pre.doc_stride},
        desc="Preprocessing",
    )


def pick_device(use_cpu: bool = False) -> torch.device:
    """A CUDA GPU if there is one (Colab), else the Mac's GPU, else the CPU."""
    if use_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict_logits(model, features, batch_size: int = 32, device: torch.device | None = None,
                   progress_every: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over every window. Returns (start_logits, end_logits), one row per window."""
    device = device or pick_device()
    model.to(device)
    model.eval()   # switch off dropout: we want the same answer every time

    # Only pass the columns the model's forward() accepts. DistilBERT and RoBERTa take
    # input_ids and attention_mask; BERT also takes token_type_ids.
    accepted = set(inspect.signature(model.forward).parameters)
    columns = [c for c in ("input_ids", "attention_mask", "token_type_ids")
               if c in accepted and c in features.column_names]
    tensors = features.with_format("torch", columns=columns)

    n_windows, width = len(features), len(features[0]["input_ids"])
    start_logits = np.full((n_windows, width), PADDING_SCORE, dtype=np.float32)
    end_logits = np.full((n_windows, width), PADDING_SCORE, dtype=np.float32)

    with torch.no_grad():   # no training here, so don't keep what backpropagation would need
        for first in range(0, n_windows, batch_size):
            batch = tensors[first:first + batch_size]
            # Every window is padded to max_length, but most passages are much shorter. Cutting
            # the batch at its longest real window gives the same scores in about half the time.
            longest = int(batch["attention_mask"].sum(dim=1).max())
            inputs = {name: tensor[:, :longest].to(device) for name, tensor in batch.items()}
            outputs = model(**inputs)
            start_logits[first:first + batch_size, :longest] = outputs.start_logits.float().cpu().numpy()
            end_logits[first:first + batch_size, :longest] = outputs.end_logits.float().cpu().numpy()
            done = min(first + batch_size, n_windows)
            if progress_every and (done // batch_size) % progress_every == 0:
                print(f"  {done:,} / {n_windows:,} windows", flush=True)
    return start_logits, end_logits


def evaluate_model(model, tokenizer, examples, cfg: Config, batch_size: int = 32,
                   device: torch.device | None = None, progress_every: int = 0,
                   use_official_metric: bool = True) -> tuple[dict, dict[str, Prediction], dict[str, dict]]:
    """Preprocess, predict, post-process and score.

    Returns (summary, predictions, per-question scores). `examples` is a datasets.Dataset
    with the SQuAD columns. Takes the model as an argument so the tests can pass a tiny one.
    """
    device = device or pick_device()
    post = cfg.postprocessing
    features = make_eval_features(examples, tokenizer, cfg.preprocessing)

    started = time.time()
    start_logits, end_logits = predict_logits(model, features, batch_size, device, progress_every)
    runtime = time.time() - started

    predictions = postprocess_predictions(
        examples, features, start_logits, end_logits,
        n_best=post.n_best, max_answer_length=post.max_answer_length, null_threshold=post.null_threshold,
    )
    answers = {ex_id: p.text for ex_id, p in predictions.items()}
    scores = per_example_scores(answers, examples)
    metrics = summarize(scores)

    summary = {
        "questions": len(examples),
        "windows": len(features),
        "device": str(device),
        "predict_seconds": round(runtime, 1),
        "settings": {
            "max_length": cfg.preprocessing.max_length,
            "doc_stride": cfg.preprocessing.doc_stride,
            "n_best": post.n_best,
            "max_answer_length": post.max_answer_length,
            "null_threshold": post.null_threshold,
        },
        "metrics": {k: round(v, 2) if isinstance(v, float) else v for k, v in metrics.items()},
    }
    if use_official_metric:
        official = official_scores(answers, examples)
        if not scores_agree(metrics, official):
            raise RuntimeError(f"Our scores {metrics} differ from the official metric {official}")
        summary["matches_official_metric"] = True
    return summary, predictions, scores


# --------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------

def save_results(out_dir: str | Path, name: str, summary: dict,
                 predictions: dict[str, Prediction], scores: dict[str, dict]) -> tuple[Path, Path]:
    """Write <name>.json (scores) and <name>_predictions.json (one entry per question)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{name}.json"
    predictions_path = out_dir / f"{name}_predictions.json"

    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    per_question = {
        ex_id: {
            "prediction": p.text,
            "best_span": p.best_span_text,
            # score_diff = null score - best span score. Kept so the threshold can be changed later.
            "score_diff": round(p.score_diff, 4),
            "em": scores[ex_id]["em"],
            "f1": round(scores[ex_id]["f1"], 4),
        }
        for ex_id, p in predictions.items()
    }
    predictions_path.write_text(json.dumps(per_question, indent=0) + "\n", encoding="utf-8")
    return summary_path, predictions_path


def print_summary(summary: dict) -> None:
    m = summary["metrics"]
    print(f"\n{summary['model']} on {summary['questions']:,} dev questions "
          f"({summary['windows']:,} windows, {summary['predict_seconds']:.0f}s on {summary['device']})")
    print(f"{'':14}{'EM':>8}{'F1':>8}{'questions':>11}")
    for label, prefix in (("All", ""), ("Answerable", "HasAns_"), ("Unanswerable", "NoAns_")):
        if f"{prefix}total" in m:
            print(f"{label:14}{m[prefix + 'exact']:8.2f}{m[prefix + 'f1']:8.2f}{m[prefix + 'total']:11,}")
    if summary.get("matches_official_metric"):
        print("These match the official squad_v2 metric.")


def print_examples(examples, predictions: dict[str, Prediction], scores: dict[str, dict], n: int) -> None:
    """Show the first n questions with the gold answers and the model's answer."""
    for example in list(examples)[:n]:
        p, s = predictions[example["id"]], scores[example["id"]]
        gold = sorted(set(example["answers"]["text"])) or ["(no answer)"]
        print(f"\nQ: {example['question'].strip()}")
        print(f"   gold:      {' | '.join(gold)}")
        print(f"   predicted: {p.text or '(no answer)'}    EM {s['em']}  F1 {s['f1']:.2f}")
        if not p.text:
            print(f"   best span if forced to answer: {p.best_span_text!r} "
                  f"(null score ahead by {p.score_diff:.2f})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True, help="Hub id or local folder of a fine-tuned QA model")
    parser.add_argument("--name", default=None, help="name for the result files (default: from --model)")
    parser.add_argument("--config", default="configs/distilbert.yaml")
    parser.add_argument("--limit", type=int, default=None, help="only the first N dev questions (quick checks)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--show", type=int, default=0, help="print this many questions with their answers")
    parser.add_argument("--no-save", action="store_true", help="don't write result files")
    parser.add_argument("--cpu", action="store_true", help="force the CPU")
    args = parser.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    from src.data import take_fraction

    cfg = load_config(args.config)
    name = args.name or Path(args.model.rstrip("/")).name
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model)

    dev = take_fraction(load_dataset(cfg.data.dataset_name, split="validation"),
                        cfg.data.eval_fraction, cfg.training.seed)
    if args.limit:
        dev = dev.select(range(min(args.limit, len(dev))))

    summary, predictions, scores = evaluate_model(
        model, tokenizer, dev, cfg, batch_size=args.batch_size, device=pick_device(args.cpu), progress_every=20)
    summary = {"model": args.model, "dataset": f"{cfg.data.dataset_name} (validation)", **summary}

    if args.show:
        print_examples(dev, predictions, scores, args.show)
    print_summary(summary)
    if not args.no_save:
        paths = save_results(args.output_dir, name, summary, predictions, scores)
        print("Saved " + " and ".join(str(p) for p in paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
