"""Turn the model's start/end scores into an answer, or into "no answer".

What the model gives us, for every window of every question:
  * start_logits[k]: how much it thinks the answer STARTS at token k
  * end_logits[k]:   how much it thinks the answer ENDS at token k
A higher number means more likely. The numbers are raw scores ("logits"), not probabilities.

What we do with them:
  1. Score of a span from token s to token e = start_logits[s] + end_logits[e].
  2. In each window, try the `n_best` highest start tokens with the `n_best` highest end
     tokens, and throw away spans that are impossible: outside the passage, ending before
     they start, or longer than `max_answer_length` tokens.
  3. A long passage has several windows. The best span of the question is the best span
     found in any of its windows.
  4. "No answer" has a score too: start_logits[0] + end_logits[0], the [CLS] token, which is
     what unanswerable questions were labeled with in training. We take the LOWEST such score
     over the windows, because one window that sees the answer should be able to overrule
     the windows that don't.
  5. The decision: say "no answer" when
         null_score - best_span_score > tau          (tau = `null_threshold`)
     A larger tau makes the model answer more often, a smaller one makes it abstain more.

Step 5 is kept separate from the rest (`apply_threshold`), so tau can be tuned later
without running the model again.

Steps 3 and 4 compare scores from DIFFERENT windows. That is fine for SQuAD, where 99% of
the passages fit in one window, but a whole article has dozens of windows, and the highest
span score among dozens is high by luck alone. `window_rule="per_window"` is the alternative
for long documents: every window is judged on its own (its no-answer score minus its own
best span), and the window that is most sure it holds the answer wins (see src/longdoc.py).

Adapted from the Hugging Face LLM Course, chapter 7.7, and the official SQuAD 2.0
post-processing in the transformers question-answering examples.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Iterable, Sequence

import numpy as np


@dataclass
class Prediction:
    """Everything we know about the model's answer to one question."""

    text: str                 # final answer after the threshold; "" means "no answer"
    best_span_text: str       # what the model would say if it had to give an answer
    best_span_score: float    # start + end score of that span
    null_score: float         # score of "no answer"
    score_diff: float         # null_score - best_span_score; "no answer" when this is > tau
    candidates: list[dict] = field(default_factory=list)   # top spans: text, score, start_char, end_char

    def to_dict(self) -> dict:
        return asdict(self)


def _is_passage_token(offset) -> bool:
    """False for question, special and padding tokens (prepare_eval_features blanks their offsets)."""
    return offset is not None and offset[0] is not None


def best_spans_in_window(
    start_logits: Sequence[float],
    end_logits: Sequence[float],
    offsets: Sequence,
    n_best: int = 20,
    max_answer_length: int = 30,
) -> list[tuple[float, int, int]]:
    """Valid answer spans in one window as (score, start_char, end_char), in no particular order.

    Trying every start with every end would be 384 x 384 pairs per window. The best span
    almost always uses one of the top few starts and ends, so we only try n_best x n_best.
    """
    start_logits = np.asarray(start_logits)
    end_logits = np.asarray(end_logits)
    top_starts = np.argsort(start_logits)[::-1][:n_best]
    top_ends = np.argsort(end_logits)[::-1][:n_best]

    spans = []
    for s in top_starts:
        if not _is_passage_token(offsets[s]):
            continue
        for e in top_ends:
            if not _is_passage_token(offsets[e]):
                continue
            if e < s or e - s + 1 > max_answer_length:
                continue
            score = float(start_logits[s] + end_logits[e])
            spans.append((score, int(offsets[s][0]), int(offsets[e][1])))
    return spans


def choose_answer(best_span_text: str, score_diff: float, null_threshold: float = 0.0) -> str:
    """The final answer: "" (no answer) when the null score beats the best span by more than tau."""
    return "" if score_diff > null_threshold else best_span_text


def postprocess_predictions(
    examples: Iterable[dict],
    features: Iterable[dict],
    start_logits,
    end_logits,
    n_best: int = 20,
    max_answer_length: int = 30,
    null_threshold: float = 0.0,
    n_candidates: int = 5,
    window_rule: str = "global",
) -> dict[str, Prediction]:
    """Combine the scores of all windows into one Prediction per question.

    examples:  the original questions; needs "id" and "context".
    features:  the windows from `prepare_eval_features`; needs "example_id" and "offset_mapping".
    start_logits, end_logits: one row per window, in the same order as `features`.

    window_rule: "global" (default) takes the best span and the lowest no-answer score over
    all windows. "per_window" takes both from the single window whose best span beats its
    own no-answer score by the most. With one window per question the two are identical.

    Returns {question id: Prediction}. The answer text is cut out of the ORIGINAL passage
    using character offsets, so it keeps the passage's capitalisation and spacing.
    """
    if window_rule not in ("global", "per_window"):
        raise ValueError(f"window_rule must be 'global' or 'per_window', got {window_rule!r}")

    windows_of = defaultdict(list)
    all_offsets = []
    for i, feature in enumerate(features):
        windows_of[feature["example_id"]].append(i)
        all_offsets.append(feature["offset_mapping"])
    if len(all_offsets) != len(start_logits) or len(all_offsets) != len(end_logits):
        raise ValueError(
            f"{len(all_offsets)} windows but {len(start_logits)} start and {len(end_logits)} end score rows"
        )

    predictions = {}
    for example in examples:
        context = example["context"]
        null_score = float("inf")
        best_score_of_span: dict[tuple[int, int], float] = {}
        surest_window = None   # (margin, null score, span score, start_char, end_char) of the most confident window

        for i in windows_of[example["id"]]:
            window_null = float(start_logits[i][0] + end_logits[i][0])
            null_score = min(null_score, window_null)
            spans = best_spans_in_window(start_logits[i], end_logits[i], all_offsets[i], n_best, max_answer_length)
            if spans:
                score, start_char, end_char = max(spans)
                vote = (window_null - score, window_null, score, start_char, end_char)
                if surest_window is None or vote < surest_window:
                    surest_window = vote
            for score, start_char, end_char in spans:
                # Neighbouring windows overlap, so the same span can turn up twice. Keep its best score.
                key = (start_char, end_char)
                if score > best_score_of_span.get(key, float("-inf")):
                    best_score_of_span[key] = score

        ranked = sorted(best_score_of_span.items(), key=lambda item: item[1], reverse=True)
        candidates = [
            {"text": context[start:end], "score": score, "start_char": start, "end_char": end}
            for (start, end), score in ranked[:n_candidates]
        ]

        if candidates:
            best_text, best_score = candidates[0]["text"], candidates[0]["score"]
        else:
            # No valid span at all (should not happen with a real passage): the only option is "no answer".
            best_text, best_score = "", float("-inf")
        score_diff = null_score - best_score

        if window_rule == "per_window" and surest_window is not None:
            score_diff, null_score, best_score, start_char, end_char = surest_window
            best_text = context[start_char:end_char]
            # List the chosen span first, so candidates[0] is always the span behind the answer.
            chosen = {"text": best_text, "score": best_score, "start_char": start_char, "end_char": end_char}
            others = [c for c in candidates if (c["start_char"], c["end_char"]) != (start_char, end_char)]
            candidates = ([chosen] + others)[:n_candidates]

        predictions[example["id"]] = Prediction(
            text=choose_answer(best_text, score_diff, null_threshold),
            best_span_text=best_text,
            best_span_score=best_score,
            null_score=null_score,
            score_diff=score_diff,
            candidates=candidates,
        )
    return predictions


def apply_threshold(predictions: dict[str, Prediction], null_threshold: float) -> dict[str, str]:
    """{question id: answer text} for a different tau, without running the model again.

    Used for the threshold sweep (T7) and the demo's threshold slider.
    """
    return {
        ex_id: choose_answer(p.best_span_text, p.score_diff, null_threshold)
        for ex_id, p in predictions.items()
    }
