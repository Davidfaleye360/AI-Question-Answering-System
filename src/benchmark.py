"""Measure how big and how fast each model is (task T7).

    python -m src.benchmark --models Ingsyling/distilbert-squad2 deepset/roberta-base-squad2

Accuracy is only half of the comparison. DistilBERT exists because it is smaller and faster
than BERT-sized models, so we measure what we get in return for the lower F1:
  * size: number of parameters, and megabytes of weights
  * speed: seconds to answer ONE question on a CPU, the way the demo does it
    (tokenize -> model -> pick the answer), one question at a time

How the timing is kept fair:
  * both models answer the same questions (a fixed random sample of the dev set),
  * on the CPU, because the demo's free Hugging Face Space has no GPU,
  * after a few warm-up questions, because the first calls are slower than the rest,
  * and we report the median, which one slow outlier cannot move.
The numbers depend on the computer, so the result file records which one was used.

Output: results/efficiency.json.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from src.config import Config, load_config
from src.predict import answer_question, load_for_inference


def model_size(model) -> dict:
    """Parameters (in millions) and megabytes of weights as stored in memory."""
    parameters = sum(p.numel() for p in model.parameters())
    n_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return {"parameters_millions": round(parameters / 1e6, 1), "size_mb": round(n_bytes / 1e6, 1)}


def time_questions(model, tokenizer, examples, cfg: Config, warmup: int = 5, clock=time.perf_counter) -> list[float]:
    """Seconds taken to answer each question on the CPU, one at a time.

    The first `warmup` questions are answered but not timed (and timed again later, so no
    question is left out).
    """
    examples = list(examples)
    for example in examples[:warmup]:
        answer_question(model, tokenizer, example["question"], example["context"], cfg)

    times = []
    for example in examples:
        started = clock()
        answer_question(model, tokenizer, example["question"], example["context"], cfg)
        times.append(clock() - started)
    return times


def summarize_times(times: list[float]) -> dict:
    """Milliseconds per question: median, mean and the slowest 10% (90th percentile)."""
    ordered = sorted(times)
    p90 = ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]
    median = statistics.median(ordered)
    return {
        "questions": len(ordered),
        "median_ms": round(1000 * median, 1),
        "mean_ms": round(1000 * statistics.fmean(ordered), 1),
        "p90_ms": round(1000 * p90, 1),
        "questions_per_second": round(1 / median, 1),
    }


def machine_description() -> dict:
    return {
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "system": f"{platform.system()} {platform.release()}",
        "torch": torch.__version__,
        "cpu_threads": torch.get_num_threads(),
    }


def compare(results: dict[str, dict], baseline: str) -> dict[str, dict]:
    """How each model compares with `baseline`: 0.5 means half the size, 2.0 means twice as fast."""
    base = results[baseline]
    return {
        name: {
            "size_ratio": round(r["size_mb"] / base["size_mb"], 2),
            "speedup": round(base["median_ms"] / r["median_ms"], 2),
        }
        for name, r in results.items() if name != baseline
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--models", nargs="+", required=True, help="Hub ids or local folders")
    parser.add_argument("--n-questions", type=int, default=200)
    parser.add_argument("--config", default="configs/distilbert.yaml")
    parser.add_argument("--output", default="results/efficiency.json")
    args = parser.parse_args()

    from datasets import load_dataset

    cfg = load_config(args.config)
    dev = load_dataset(cfg.data.dataset_name, split="validation")
    sample = dev.shuffle(seed=cfg.training.seed).select(range(min(args.n_questions, len(dev))))

    results = {}
    for name in args.models:   # one model at a time, so they don't compete for the CPU
        model, tokenizer = load_for_inference(name, "cpu")
        results[name] = {**model_size(model), **summarize_times(time_questions(model, tokenizer, sample, cfg))}
        r = results[name]
        print(f"{name}: {r['parameters_millions']} M parameters, {r['size_mb']} MB, "
              f"{r['median_ms']} ms per question (median of {r['questions']}), p90 {r['p90_ms']} ms")
        del model

    out = {
        "what": "CPU time to answer one dev question end to end (tokenize, model, post-process), batch size 1",
        "dataset": f"{cfg.data.dataset_name} (validation), {len(sample)} random questions, seed {cfg.training.seed}",
        "machine": machine_description(),
        "models": results,
        "compared_with_last_model": compare(results, args.models[-1]) if len(args.models) > 1 else {},
    }
    Path(args.output).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("Saved", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
