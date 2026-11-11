"""Answer ONE question about one passage: the path the demo and the speed benchmark use.

    from transformers import AutoModelForQuestionAnswering, AutoTokenizer
    from src.config import load_config
    from src.predict import answer_question

    tokenizer = AutoTokenizer.from_pretrained("Ingsyling/distilbert-squad2")
    model = AutoModelForQuestionAnswering.from_pretrained("Ingsyling/distilbert-squad2")
    prediction = answer_question(model, tokenizer, "Who wrote Hamlet?", passage, load_config())
    prediction.text          # "" means "no answer"
    prediction.candidates    # the top spans with their scores

It runs the same three steps as `src/evaluate.py` (windows -> scores -> answer) and reuses the
same functions, so a question gets the same answer here as in the evaluation. A long passage
simply becomes several overlapping windows, which is also how the long-document mode works.
"""
from __future__ import annotations

import inspect

import numpy as np
import torch

from src.config import Config
from src.postprocess import Prediction, postprocess_predictions
from src.preprocess import prepare_eval_features

PADDING_SCORE = -10000.0   # score for padding positions the model was never shown


def answer_question(model, tokenizer, question: str, context: str, cfg: Config,
                    null_threshold: float | None = None, n_candidates: int = 5,
                    device: torch.device | str = "cpu", window_rule: str = "global",
                    details: dict | None = None) -> Prediction:
    """The model's answer to `question` given `context`, with its top candidate spans.

    null_threshold=None uses the value in the config; the demo's slider passes its own.
    window_rule is how the windows of a long passage are combined (see src/postprocess.py);
    it makes no difference for a passage that fits in one window.
    If `details` is a dict, the number of windows is written into it as details["windows"].
    The model must already be on `device` and in eval mode (see `load_for_inference`).
    """
    pre, post = cfg.preprocessing, cfg.postprocessing
    example = {"id": "question", "question": question, "context": context}
    enc = prepare_eval_features({key: [value] for key, value in example.items()}, tokenizer,
                                max_length=pre.max_length, stride=pre.doc_stride)

    # Only the columns the model's forward() accepts (BERT takes token_type_ids, DistilBERT does not).
    accepted = set(inspect.signature(model.forward).parameters)
    attention = torch.tensor(enc["attention_mask"])
    longest = int(attention.sum(dim=1).max())   # cut off the padding: same scores, much less work
    inputs = {name: torch.tensor(enc[name])[:, :longest].to(device)
              for name in ("input_ids", "attention_mask", "token_type_ids")
              if name in accepted and name in enc}

    with torch.no_grad():
        outputs = model(**inputs)

    n_windows, width = attention.shape
    start_logits = np.full((n_windows, width), PADDING_SCORE, dtype=np.float32)
    end_logits = np.full((n_windows, width), PADDING_SCORE, dtype=np.float32)
    start_logits[:, :longest] = outputs.start_logits.float().cpu().numpy()
    end_logits[:, :longest] = outputs.end_logits.float().cpu().numpy()

    features = [{"example_id": example["id"], "offset_mapping": enc["offset_mapping"][i]} for i in range(n_windows)]
    predictions = postprocess_predictions(
        [example], features, start_logits, end_logits,
        n_best=post.n_best, max_answer_length=post.max_answer_length,
        null_threshold=post.null_threshold if null_threshold is None else null_threshold,
        n_candidates=n_candidates, window_rule=window_rule,
    )
    if details is not None:
        details["windows"] = n_windows
    return predictions[example["id"]]


def load_for_inference(model_name: str, device: torch.device | str = "cpu"):
    """(model, tokenizer) ready for `answer_question`: on the device, dropout switched off."""
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)
    model.eval()
    return model, tokenizer
