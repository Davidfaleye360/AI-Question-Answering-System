"""Sanity check: do our training labels point at the right answers?

Runs preprocessing on real SQuAD 2.0 examples, turns every (start, end) label back into
text, and compares it with the gold answer. Run this before any training run.

    python scripts/check_preprocessing.py --n-examples 2000

Passes when:
  * every unanswerable question is labeled (0, 0) in all of its windows
  * every answerable question has at least one window whose label decodes to its answer
  * at least 95% of those decode to the answer exactly (the rest are "near matches",
    usually a gold answer that cuts through a word, like "1990" inside "1990s")
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

# Make `src` importable when running from the repo root or the scripts folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocess import answer_char_span, decode_label, prepare_train_features  # noqa: E402


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def run_checks(examples: dict[str, list], tokenizer, max_length: int = 384, stride: int = 128) -> dict:
    """Preprocess a batch of examples (column dict) and compare decoded labels with gold answers."""
    feats = prepare_train_features(examples, tokenizer, max_length, stride, keep_mapping=True)
    n_examples, n_features = len(examples["id"]), len(feats["input_ids"])

    decoded = defaultdict(list)  # example index -> decoded text of each of its windows' labels
    for i in range(n_features):
        ex = feats["overflow_to_sample_mapping"][i]
        decoded[ex].append(decode_label(
            feats["offset_mapping"][i], examples["context"][ex],
            feats["start_positions"][i], feats["end_positions"][i],
        ))

    report = {"examples": n_examples, "features": n_features, "answerable": 0, "exact": 0,
              "near": [], "missing": [], "bad_unanswerable": [], "decoded": dict(decoded)}
    for ex in range(n_examples):
        spans = [t for t in decoded[ex] if t]
        if answer_char_span(examples["answers"][ex]) is None:
            if spans:
                report["bad_unanswerable"].append(ex)
            continue
        report["answerable"] += 1
        gold = examples["answers"][ex]["text"][0]
        if gold in spans:
            report["exact"] += 1
        elif any(_norm(gold) in _norm(t) or _norm(t) in _norm(gold) for t in spans):
            report["near"].append((ex, gold, spans[0]))
        else:
            report["missing"].append((ex, gold, spans))

    exact_rate = report["exact"] / max(report["answerable"], 1)
    report["exact_rate"] = exact_rate
    report["passed"] = (not report["bad_unanswerable"] and not report["missing"] and exact_rate >= 0.95)
    return report


def print_report(report: dict, examples: dict[str, list], n_show: int = 20, seed: int = 42) -> None:
    n_unans = report["examples"] - report["answerable"]
    print(f"Examples: {report['examples']:,} -> features (windows): {report['features']:,} "
          f"({report['features'] / report['examples']:.3f} per example)")
    print(f"Answerable: {report['answerable']:,} | exact label match: {report['exact']:,} "
          f"({100 * report['exact_rate']:.2f}%) | near match: {len(report['near'])} "
          f"| missing: {len(report['missing'])}")
    print(f"Unanswerable: {n_unans:,} | wrongly given a span: {len(report['bad_unanswerable'])}")

    answerable = [ex for ex in range(report["examples"]) if answer_char_span(examples["answers"][ex]) is not None]
    print(f"\n{min(n_show, len(answerable))} random answerable labels (gold -> decoded):")
    for ex in random.Random(seed).sample(answerable, min(n_show, len(answerable))):
        gold = examples["answers"][ex]["text"][0]
        got = next((t for t in report["decoded"][ex] if t), "(no window labeled)")
        print(f"  {'OK' if got == gold else '~ '}  {gold!r:42.42} -> {got!r:42.42}")

    if report["near"]:
        print("\nNear matches (fine in small numbers):")
        for _, gold, got in report["near"][:5]:
            print(f"  {gold!r} -> {got!r}")
    if report["missing"]:
        print("\nMissing (investigate these):")
        for ex, gold, got in report["missing"][:5]:
            print(f"  example {ex}: {gold!r} -> {got}")
    if report["bad_unanswerable"]:
        print(f"\nUnanswerable examples given a span: {report['bad_unanswerable'][:10]}")
    print("\nRESULT:", "PASS" if report["passed"] else "FAIL")


def main() -> int:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    from src.config import load_config

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/distilbert.yaml")
    parser.add_argument("--n-examples", type=int, default=2000, help="training examples to check")
    parser.add_argument("--n-show", type=int, default=20, help="random labels to print")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    pre = cfg.preprocessing
    print(f"Model: {cfg.model.checkpoint} | max_length={pre.max_length} stride={pre.doc_stride}\n")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.checkpoint)
    # Random sample: SQuAD is ordered by article, so the first N rows would all be about one topic.
    train = load_dataset(cfg.data.dataset_name, split="train")
    n = min(args.n_examples, len(train))
    examples = train.shuffle(seed=args.seed).select(range(n)).to_dict()

    report = run_checks(examples, tokenizer, pre.max_length, pre.doc_stride)
    print_report(report, examples, args.n_show, args.seed)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
