"""Everything the demo shows, without any Gradio code (the screen is in app/app.py).

Keeping this apart from the user interface means it can be unit-tested, and it makes the
two steps of the demo visible:

  1. `ask(...)` runs the model once and returns a plain dict with the best span, the
     no-answer score and the top candidates. This is the slow step.
  2. `render(result, tau)` turns that dict into what is on screen for a given threshold.
     Moving the threshold slider only repeats this step, which is why the answer can flip
     between a span and "no answer" instantly: the model's scores stay the same, only our
     decision changes.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

from src.config import Config
from src.postprocess import choose_answer
from src.predict import answer_question, load_for_inference

ANSWER_LABEL = "answer"
REJECTED_LABEL = "best guess (rejected)"
NO_ANSWER_TEXT = "(no answer)"


@dataclass(frozen=True)
class DemoModel:
    label: str            # shown in the model switch
    hub_id: str           # where the weights come from
    default_tau: float    # the best threshold found by the sweep in T7 (results/threshold_sweep.json)
    note: str             # one line under the switch


MODELS = {
    m.label: m for m in (
        DemoModel("DistilBERT (ours)", "Ingsyling/distilbert-squad2", -1.4,
                  "Fine-tuned by us on SQuAD 2.0. 66 M parameters, 67.4 F1 on the dev set."),
        DemoModel("RoBERTa (deepset)", "deepset/roberta-base-squad2", -1.6,
                  "Published baseline. 124 M parameters, 82.9 F1 on the dev set, about half as fast."),
    )
}
DEFAULT_MODEL = "DistilBERT (ours)"

_loaded: dict[str, tuple] = {}


def get_model(label: str, loader=load_for_inference) -> tuple:
    """(model, tokenizer) for a label. Loaded on first use and kept, so switching back is instant."""
    if label not in _loaded:
        _loaded[label] = loader(MODELS[label].hub_id, "cpu")
    return _loaded[label]


def ask(model, tokenizer, question: str, context: str, cfg: Config, n_candidates: int = 5) -> dict:
    """Run the model once. Returns a plain dict (it is kept in the browser session between clicks)."""
    started = time.perf_counter()
    details: dict = {}
    # "per_window" is the rule that held up best on whole articles at an untuned threshold (T10,
    # results/long_documents.json). For a passage that fits in one window it changes nothing.
    prediction = answer_question(model, tokenizer, question.strip(), context, cfg, n_candidates=n_candidates,
                                 window_rule="per_window", details=details)
    return {
        "context": context,
        "best_span": prediction.best_span_text,
        "best_span_score": prediction.best_span_score,
        "null_score": prediction.null_score,
        "score_diff": prediction.score_diff,
        "candidates": prediction.candidates,
        "seconds": time.perf_counter() - started,
        "windows": details["windows"],
    }


def highlight(context: str, start_char: int, end_char: int, label: str) -> list[tuple[str, str | None]]:
    """The passage cut into (text, label) pieces for Gradio's HighlightedText; only the span has a label."""
    pieces = [(context[:start_char], None), (context[start_char:end_char], label), (context[end_char:], None)]
    return [(text, tag) for text, tag in pieces if text]


def confidence_shares(candidates: list[dict], null_score: float) -> list[float]:
    """How the model's confidence is split between the listed spans and "no answer", in percent.

    A softmax over their scores: each score becomes e^score, divided by the sum. The last
    number is the share of "no answer". It only compares the options shown, so it is a guide
    to how sure the model is, not a calibrated probability.
    """
    scores = [c["score"] for c in candidates] + [null_score]
    top = max(scores)
    weights = [math.exp(s - top) for s in scores]   # subtracting the top score avoids overflow
    total = sum(weights)
    return [100.0 * w / total for w in weights]


def render(result: dict | None, tau: float) -> tuple[str, list, list[list], str]:
    """What the screen shows for this result at threshold tau.

    Returns (answer as Markdown, highlighted passage, candidate table rows, explanation).
    """
    if not result:
        return "", [], [], ""

    context, candidates = result["context"], result["candidates"]
    answer = choose_answer(result["best_span"], result["score_diff"], tau)
    shares = confidence_shares(candidates, result["null_score"])
    rows = [[c["text"], round(c["score"], 2), f"{share:.1f}%"] for c, share in zip(candidates, shares)]
    rows.append([NO_ANSWER_TEXT, round(result["null_score"], 2), f"{shares[-1]:.1f}%"])

    diff = result["score_diff"]
    timing = f"Model time: {1000 * result['seconds']:.0f} ms."
    if result.get("windows", 1) > 1:
        timing += f" The passage was too long to read at once, so it was read in {result['windows']} overlapping windows."
    if not candidates:
        return "### No answer found in the passage", [(context, None)], rows, timing

    top = candidates[0]
    if answer:
        headline = f"### {answer}"
        passage = highlight(context, top["start_char"], top["end_char"], ANSWER_LABEL)
        if diff <= 0:
            why = (f"The best span beats \"no answer\" by {-diff:.2f}. It would be rejected "
                   f"only with a threshold below {diff:.2f}.")
        else:
            why = (f"\"No answer\" is ahead by {diff:.2f}, but the threshold of {tau:.2f} allows "
                   f"a lead of up to {tau:.2f}, so the span is given.")
    else:
        headline = "### No answer found in the passage"
        passage = highlight(context, top["start_char"], top["end_char"], REJECTED_LABEL)
        if diff > 0:
            why = (f"\"No answer\" beats the best span (\"{top['text']}\") by {diff:.2f}, more than the "
                   f"threshold of {tau:.2f} allows.")
        else:
            # A negative threshold is a cautious one: the span has to WIN by a margin, and this one doesn't.
            why = (f"The best span (\"{top['text']}\") beats \"no answer\" by only {-diff:.2f}, and the "
                   f"threshold of {tau:.2f} demands a lead of at least {-tau:.2f}.")
        why += f" Raise the threshold to {diff:.2f} or more to see that span as the answer."
    return headline, passage, rows, f"{why} {timing}"


# Passages for the example buttons. The first two are a pair: same passage, one question it can
# answer and one it cannot, which is the point of SQuAD 2.0.
NORMANS = (
    "The Normans were the people who in the 10th and 11th centuries gave their name to Normandy, a region "
    "in France. They were descended from Norse raiders and pirates from Denmark, Iceland and Norway who, "
    "under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through "
    "generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their "
    "descendants would gradually merge with the Carolingian-based cultures of West Francia."
)
OXYGEN = (
    "Oxygen is a chemical element with symbol O and atomic number 8. It is a member of the chalcogen group "
    "on the periodic table and is a highly reactive nonmetal and oxidizing agent that readily forms "
    "compounds with most elements. By mass, oxygen is the third-most abundant element in the universe, "
    "after hydrogen and helium. At standard temperature and pressure, two atoms of the element bind to "
    "form dioxygen, a colorless and odorless diatomic gas with the formula O2."
)
EXAMPLES = [
    [NORMANS, "In what country is Normandy located?"],
    [NORMANS, "Who was the Norse leader?"],
    [NORMANS, "What is France a region of?"],
    [OXYGEN, "What is the atomic number of oxygen?"],
    [OXYGEN, "Which element is the most abundant in the universe by mass?"],
    [OXYGEN, "Who discovered oxygen?"],
]
