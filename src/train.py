"""Fine-tune a question answering model on SQuAD 2.0.

    # Smoke test (T3): 50 steps on 1% of the data, a few minutes at most
    python -m src.train --config configs/distilbert.yaml --overrides configs/smoke.yaml

    # Full run (T6): 2 epochs on all of the data, on a GPU
    python -m src.train --config configs/distilbert.yaml

What happens:
  1. Load the pretrained model. It gets a new, untrained "QA head": one small layer that
     gives every token a start score and an end score.
  2. Turn the training questions into windows with start/end labels (src/preprocess.py).
  3. Train with the Hugging Face Trainer. The loss measures how far the predicted start and
     end positions are from the labels, so it should fall as the model learns.
  4. Save checkpoints along the way (a Colab session can disconnect), then the final model.

The run ends with "RESULT: PASS" if the loss went down and a checkpoint and the final model
were saved. That is what the smoke test checks before we spend hours on the full run.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from src.config import Config, PreprocessingConfig, TrainingConfig, load_config
from src.data import take_fraction
from src.preprocess import prepare_train_features

SUMMARY_FILE = "train_summary.json"


def make_train_features(raw_train, tokenizer, pre: PreprocessingConfig):
    """Tokenize the training questions into windows with start/end labels."""
    return raw_train.map(
        prepare_train_features,
        batched=True,
        remove_columns=raw_train.column_names,  # the model only needs the token columns
        fn_kwargs={"tokenizer": tokenizer, "max_length": pre.max_length, "stride": pre.doc_stride},
        desc="Preprocessing",
    )


def build_training_arguments(training: TrainingConfig, fp16: bool, use_cpu: bool = False) -> TrainingArguments:
    """Translate our config into the Trainer's settings. No numbers are written here."""
    return TrainingArguments(
        output_dir=training.output_dir,
        learning_rate=training.learning_rate,
        per_device_train_batch_size=training.batch_size,
        num_train_epochs=training.num_epochs,
        max_steps=training.max_steps,        # -1 means "use num_train_epochs"
        warmup_ratio=training.warmup_ratio,  # learning rate climbs for the first 10% of steps, then falls to 0
        weight_decay=training.weight_decay,
        fp16=fp16,
        logging_steps=training.logging_steps,
        save_strategy="steps",
        save_steps=training.save_steps,
        save_total_limit=training.save_total_limit,  # keep only the newest checkpoints to save disk space
        seed=training.seed,
        report_to="none",                    # no Weights & Biases or TensorBoard prompts in Colab
        use_cpu=use_cpu,
    )


def training_losses(log_history: list[dict]) -> list[tuple[int, float]]:
    """(step, loss) for every training loss the Trainer recorded.

    The Trainer's log also holds a final summary entry ("train_loss", "train_runtime"),
    which has no "loss" key and is skipped.
    """
    return [(entry["step"], entry["loss"]) for entry in log_history if "loss" in entry]


def loss_went_down(losses: list[float]) -> bool:
    """True if the last third of the recorded losses averages lower than the first third.

    We compare averages because the loss on a single batch jumps around: one hard batch
    near the end should not make a healthy run look broken.
    """
    if len(losses) < 2:
        return False
    k = max(1, len(losses) // 3)
    return sum(losses[-k:]) / k < sum(losses[:k]) / k


def run_training(model, tokenizer, train_features, cfg: Config, resume: bool = True, use_cpu: bool = False) -> dict:
    """Train `model` on `train_features` and return a summary of the run.

    The model, tokenizer and data are passed in so the unit tests can use a tiny model offline.
    With resume=True, training continues from the newest checkpoint in the output folder if
    there is one. That is how a full run survives a Colab disconnect: run the same cell again.
    """
    training = cfg.training
    out_dir = Path(training.output_dir)

    # Mixed precision (fp16) makes a CUDA GPU like Colab's T4 faster and lighter on memory.
    # It does not work on a CPU or an Apple chip, so switch it off there instead of crashing.
    fp16 = training.fp16 and torch.cuda.is_available() and not use_cpu
    if training.fp16 and not fp16:
        print("Note: fp16 is on in the config but there is no CUDA GPU here, so training without it.")

    trainer = Trainer(
        model=model,
        args=build_training_arguments(training, fp16, use_cpu),
        train_dataset=train_features,
        data_collator=DefaultDataCollator(),  # stacks examples into batches; they are already padded
        processing_class=tokenizer,           # saved next to the model, so a checkpoint is self-contained
    )

    last_checkpoint = get_last_checkpoint(str(out_dir)) if resume and out_dir.is_dir() else None
    if last_checkpoint:
        print(f"Resuming from {last_checkpoint}")

    started = time.time()
    trainer.train(resume_from_checkpoint=last_checkpoint)
    runtime = time.time() - started
    trainer.save_model(str(out_dir))  # final model + tokenizer, loadable with from_pretrained(out_dir)

    losses = training_losses(trainer.state.log_history)
    checkpoints = sorted((p.name for p in out_dir.glob("checkpoint-*")), key=lambda name: int(name.split("-")[1]))
    summary = {
        "model": cfg.model.checkpoint,
        "output_dir": str(out_dir),
        "train_windows": len(train_features),
        "steps": trainer.state.global_step,
        "device": str(trainer.args.device),
        "fp16": fp16,
        "runtime_seconds": round(runtime, 1),
        "losses": losses,
        "loss_went_down": loss_went_down([loss for _, loss in losses]),
        "checkpoints": checkpoints,
        "final_model_saved": (out_dir / "model.safetensors").exists(),
    }
    summary["passed"] = bool(summary["loss_went_down"] and checkpoints and summary["final_model_saved"])
    (out_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def print_summary(summary: dict) -> None:
    """Print what happened in the run, ending with RESULT: PASS or FAIL."""
    print(f"\nTrained {summary['steps']:,} steps on {summary['train_windows']:,} windows "
          f"in {summary['runtime_seconds']:.0f}s (device: {summary['device']}, fp16: {summary['fp16']})")
    print("Training loss (step: loss):")
    print("  " + "  ".join(f"{step}: {loss:.3f}" for step, loss in summary["losses"]))
    print(f"Loss went down: {'yes' if summary['loss_went_down'] else 'NO'}")
    print(f"Checkpoints in {summary['output_dir']}: {', '.join(summary['checkpoints']) or 'NONE'}")
    print(f"Final model saved: {'yes' if summary['final_model_saved'] else 'NO'}")
    print("\nRESULT:", "PASS" if summary["passed"] else "FAIL")


def main() -> int:
    from datasets import load_dataset

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/distilbert.yaml")
    parser.add_argument("--overrides", default=None, help="second YAML file with the settings to change, e.g. configs/smoke.yaml")
    parser.add_argument("--output-dir", default=None, help="where to save, e.g. a Google Drive folder (replaces training.output_dir)")
    parser.add_argument("--fresh", action="store_true", help="start from step 0 even if the output folder has checkpoints")
    parser.add_argument("--cpu", action="store_true", help="train on the CPU even if a GPU is available")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    if args.output_dir:
        cfg.training.output_dir = args.output_dir
    t = cfg.training
    length = f"{t.max_steps} steps" if t.max_steps > 0 else f"{t.num_epochs} epochs"
    print(f"Model: {cfg.model.checkpoint} | {cfg.data.train_fraction:.0%} of the training data | {length} "
          f"| batch {t.batch_size} | lr {t.learning_rate} | saving to {t.output_dir}\n")

    # The QA head starts from random numbers. Fixing the seed makes two runs give the same result.
    set_seed(t.seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.checkpoint)
    model = AutoModelForQuestionAnswering.from_pretrained(cfg.model.checkpoint)

    raw_train = take_fraction(load_dataset(cfg.data.dataset_name, split="train"), cfg.data.train_fraction, t.seed)
    train_features = make_train_features(raw_train, tokenizer, cfg.preprocessing)
    print(f"{len(raw_train):,} questions -> {len(train_features):,} training windows")

    summary = run_training(model, tokenizer, train_features, cfg, resume=not args.fresh, use_cpu=args.cpu)
    print_summary(summary)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
