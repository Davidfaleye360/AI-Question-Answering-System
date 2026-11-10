"""Tests for src/train.py.

The training test uses a tiny, randomly initialised DistilBERT and the local test tokenizer,
so it runs offline on the CPU in a few seconds. It checks the wiring (data -> Trainer ->
checkpoints -> resume), not that the model learns anything useful.
"""
from __future__ import annotations

import json

import pytest

from conftest import CONTEXT_CURIE, CONTEXT_FAME, ROOT, make_example
from src.config import Config, TrainingConfig, load_config
from src.train import (
    SUMMARY_FILE,
    HUB_IGNORE,
    build_training_arguments,
    loss_went_down,
    make_train_features,
    prepare_hub_repo,
    push_to_hub,
    run_training,
    training_losses,
)


def test_training_losses_skips_the_final_summary_entry():
    history = [
        {"loss": 5.9, "learning_rate": 3e-5, "step": 5},
        {"loss": 4.1, "learning_rate": 1e-5, "step": 10},
        {"train_runtime": 3.0, "train_loss": 5.0, "step": 10},
    ]
    assert training_losses(history) == [(5, 5.9), (10, 4.1)]


@pytest.mark.parametrize("losses,expected", [
    ([5.9, 5.5, 4.8, 4.2, 3.9, 3.7], True),
    ([5.9, 5.2, 4.0, 3.5, 3.6, 4.4], True),    # one bad batch at the end doesn't hide the trend
    ([5.9, 5.9, 6.0, 5.9, 6.0, 6.0], False),   # flat: the model isn't learning
    ([3.0, 4.0], False),
    ([5.9], False),                             # one point can't show a trend
    ([], False),
])
def test_loss_went_down(losses, expected):
    assert loss_went_down(losses) is expected


def test_training_arguments_come_from_the_config():
    training = load_config(ROOT / "configs" / "distilbert.yaml").training
    args = build_training_arguments(training, fp16=False)
    assert args.learning_rate == pytest.approx(training.learning_rate)
    assert args.per_device_train_batch_size == training.batch_size
    assert args.num_train_epochs == training.num_epochs
    assert args.max_steps == -1                       # full run: stop by epochs, not steps
    assert args.warmup_ratio == pytest.approx(training.warmup_ratio)
    assert args.weight_decay == pytest.approx(training.weight_decay)
    assert args.save_steps == training.save_steps and args.save_total_limit == training.save_total_limit
    assert args.seed == training.seed
    assert args.fp16 is False


def _tiny_setup(tokenizer, tmp_path, max_steps):
    """A 1-layer DistilBERT, 8 training examples and a config that trains for a few steps."""
    datasets = pytest.importorskip("datasets")
    from transformers import DistilBertConfig, DistilBertForQuestionAnswering

    rows = [
        make_example("a", "When did Curie win the Chemistry prize?", CONTEXT_CURIE, "1911"),
        make_example("b", "Where was Curie born?", CONTEXT_CURIE, None),
        make_example("c", "When did she rise to fame?", CONTEXT_FAME, "in the late 1990s"),
        make_example("d", "What did Curie win in 1903?", CONTEXT_CURIE, "the Nobel Prize in Physics"),
    ] * 2
    cfg = Config()
    cfg.preprocessing.max_length, cfg.preprocessing.doc_stride = 64, 16
    cfg.training = TrainingConfig(output_dir=str(tmp_path / "run"), batch_size=4, max_steps=max_steps,
                                  logging_steps=1, save_steps=2, fp16=False)
    features = make_train_features(datasets.Dataset.from_list(rows), tokenizer, cfg.preprocessing)
    model = DistilBertForQuestionAnswering(DistilBertConfig(
        vocab_size=tokenizer.vocab_size, dim=32, hidden_dim=64, n_layers=1, n_heads=2,
        max_position_embeddings=cfg.preprocessing.max_length))
    return model, features, cfg


def test_tiny_training_run_saves_checkpoints_model_and_summary(tokenizer, tmp_path):
    model, features, cfg = _tiny_setup(tokenizer, tmp_path, max_steps=4)
    summary = run_training(model, tokenizer, features, cfg, use_cpu=True)

    out = tmp_path / "run"
    assert summary["steps"] == 4
    assert [step for step, _ in summary["losses"]] == [1, 2, 3, 4]
    assert summary["checkpoints"] == ["checkpoint-2", "checkpoint-4"]
    assert summary["final_model_saved"] and (out / "model.safetensors").exists()
    assert (out / "vocab.txt").exists(), "the tokenizer should be saved next to the model"
    assert summary["fp16"] is False and summary["device"] == "cpu"
    assert json.loads((out / SUMMARY_FILE).read_text())["steps"] == 4


def test_training_resumes_from_the_last_checkpoint(tokenizer, tmp_path, capsys):
    model, features, cfg = _tiny_setup(tokenizer, tmp_path, max_steps=4)
    first = run_training(model, tokenizer, features, cfg, use_cpu=True)
    assert "Resuming" not in capsys.readouterr().out

    # Same output folder, two more steps allowed: a new model should pick up at step 4, not step 0.
    model2, features, cfg = _tiny_setup(tokenizer, tmp_path, max_steps=6)
    resumed = run_training(model2, tokenizer, features, cfg, use_cpu=True)
    assert "checkpoint-4" in capsys.readouterr().out
    assert resumed["steps"] == 6
    # Steps 1-4 were not trained again: their losses are the ones recorded by the first run.
    assert resumed["losses"][:4] == first["losses"]
    assert [step for step, _ in resumed["losses"][4:]] == [5, 6]

    # With resume=False the same folder is ignored and training starts again from step 0.
    model3, features, cfg = _tiny_setup(tokenizer, tmp_path, max_steps=2)
    fresh = run_training(model3, tokenizer, features, cfg, resume=False, use_cpu=True)
    assert "Resuming" not in capsys.readouterr().out
    assert fresh["steps"] == 2 and fresh["losses"] != first["losses"][:2]


class FakeHub:
    """Stands in for huggingface_hub.HfApi, so the tests never touch the internet."""

    def __init__(self, fail=False):
        self.fail, self.created, self.uploads = fail, [], []

    def create_repo(self, repo_id, repo_type, exist_ok):
        if self.fail:
            raise PermissionError("401 Unauthorized")
        self.created.append((repo_id, repo_type, exist_ok))
        return f"https://huggingface.co/{repo_id}"

    def upload_folder(self, **kwargs):
        self.uploads.append(kwargs)


def test_prepare_hub_repo_creates_the_model_page_without_failing_if_it_exists():
    hub = FakeHub()
    assert prepare_hub_repo("someone/distilbert-squad2", api=hub) == "https://huggingface.co/someone/distilbert-squad2"
    assert hub.created == [("someone/distilbert-squad2", "model", True)]


def test_prepare_hub_repo_explains_a_missing_id_or_a_bad_token():
    with pytest.raises(ValueError, match="hub_model_id"):
        prepare_hub_repo(None, api=FakeHub())
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        prepare_hub_repo("someone/distilbert-squad2", api=FakeHub(fail=True))


def test_push_to_hub_uploads_the_final_model_but_not_the_checkpoints(tokenizer, tmp_path):
    import fnmatch

    model, features, cfg = _tiny_setup(tokenizer, tmp_path, max_steps=4)
    summary = run_training(model, tokenizer, features, cfg, use_cpu=True)
    out = tmp_path / "run"

    hub = FakeHub()
    url = push_to_hub(out, "someone/distilbert-squad2", summary["steps"], api=hub)
    assert url == "https://huggingface.co/someone/distilbert-squad2"
    (upload,) = hub.uploads
    assert upload["folder_path"] == str(out) and upload["repo_id"] == "someone/distilbert-squad2"

    def uploaded(name):
        return not any(fnmatch.fnmatch(name, pattern) for pattern in upload["ignore_patterns"])

    top_level = [p.name for p in out.iterdir()]
    assert "checkpoint-4" in top_level and not uploaded("checkpoint-4")
    for needed in ("model.safetensors", "config.json", "vocab.txt", SUMMARY_FILE):
        assert needed in top_level and uploaded(needed)
    assert upload["ignore_patterns"] == HUB_IGNORE


def test_the_project_config_names_a_hub_model():
    hub_model_id = load_config(ROOT / "configs" / "distilbert.yaml").model.hub_model_id
    assert hub_model_id and hub_model_id.count("/") == 1
