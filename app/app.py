"""Gradio demo: ask a question about a passage and see where the model found the answer.

Run it on your own computer (it opens at http://127.0.0.1:7860):

    python app/app.py

Add --share to also get a public https://....gradio.live link that anyone can open. The
link works for as long as this program keeps running (one week at most), because the
model still runs on this computer; the link only forwards visitors to it.

    python app/app.py --share

This file only describes the screen. What the model does is in src/predict.py and what is
shown is decided in src/demo.py, both of which have unit tests.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# In this repo the file lives in app/, on the Hugging Face Space it sits next to src/.
# Either way, make the folder that contains src/ importable.
ROOT = Path(__file__).resolve().parent
if not (ROOT / "src").is_dir():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402

from src.config import load_config  # noqa: E402
from src.demo import ANSWER_LABEL, DEFAULT_MODEL, EXAMPLES, MODELS, REJECTED_LABEL, ask, get_model, render  # noqa: E402

CFG = load_config(ROOT / "configs" / "distilbert.yaml")

INTRO = """
# Question answering on SQuAD 2.0

Paste a passage, ask a question about it, and the model points to the answer **inside the passage**.
If the passage doesn't contain the answer, the model should say so instead of guessing.

CSCI 390 course project by David Falaye and Kosisochukwu Ogbuanya ·
[code](https://github.com/Davidfaleye360/AI-Question-Answering-System) ·
[our model](https://huggingface.co/Ingsyling/distilbert-squad2)
"""

THRESHOLD_HELP = (
    "The model gives a score to its best span and a score to \"no answer\". It abstains when "
    "\"no answer\" leads by more than this threshold. Lower = more cautious, higher = answers more often. "
    "Moving the slider re-decides instantly without running the model again."
)


def run_model(model_label: str, context: str, question: str, tau: float):
    """Button click: run the chosen model, remember its scores, and show the result."""
    if not context.strip() or not question.strip():
        raise gr.Error("Please enter both a passage and a question.")
    model, tokenizer = get_model(model_label)
    result = ask(model, tokenizer, question, context, CFG)
    return (result, *render(result, tau))


def redecide(result: dict | None, tau: float):
    """Slider moved: same scores, new decision."""
    return render(result, tau)


def switch_model(model_label: str):
    """Model changed: show its description, set the slider to its tuned threshold, clear the old answer."""
    chosen = MODELS[model_label]
    return (chosen.note, chosen.default_tau, None, *render(None, chosen.default_tau))


with gr.Blocks(title="SQuAD 2.0 question answering") as demo:
    gr.Markdown(INTRO)
    result_state = gr.State(None)   # the model's scores for the last question, kept per visitor

    with gr.Row():
        with gr.Column(scale=1):
            model_choice = gr.Radio(list(MODELS), value=DEFAULT_MODEL, label="Model")
            model_note = gr.Markdown(MODELS[DEFAULT_MODEL].note)
            context_box = gr.Textbox(label="Passage", lines=10, value=EXAMPLES[0][0],
                                     placeholder="Paste any text. Long texts are read in overlapping windows.")
            question_box = gr.Textbox(label="Question", value=EXAMPLES[0][1])
            tau_slider = gr.Slider(-8, 8, value=MODELS[DEFAULT_MODEL].default_tau, step=0.1,
                                   label="No-answer threshold τ", info=THRESHOLD_HELP)
            ask_button = gr.Button("Answer", variant="primary")
        with gr.Column(scale=1):
            answer_box = gr.Markdown(label="Answer")
            passage_view = gr.HighlightedText(
                label="Where it is in the passage", show_legend=True, combine_adjacent=False,
                color_map={ANSWER_LABEL: "#2a78d6", REJECTED_LABEL: "#9a9a94"})
            explanation = gr.Markdown()
            candidates_table = gr.Dataframe(
                headers=["Candidate", "Score", "Confidence"], datatype=["str", "number", "str"],
                label="Top candidates, and \"no answer\"", interactive=False, wrap=True,
                column_widths=["56%", "16%", "28%"])

    gr.Examples(EXAMPLES, inputs=[context_box, question_box], label="Examples: some of these questions have no answer in the passage")

    outputs = [answer_box, passage_view, candidates_table, explanation]
    ask_inputs = [model_choice, context_box, question_box, tau_slider]
    ask_button.click(run_model, ask_inputs, [result_state, *outputs])
    question_box.submit(run_model, ask_inputs, [result_state, *outputs])
    tau_slider.change(redecide, [result_state, tau_slider], outputs)
    model_choice.change(switch_model, model_choice, [model_note, tau_slider, result_state, *outputs])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Question answering demo")
    parser.add_argument("--share", action="store_true",
                        help="also create a public gradio.live link (works while this program runs)")
    parser.add_argument("--port", type=int, default=None, help="local port (default: 7860)")
    args = parser.parse_args()
    demo.launch(theme=gr.themes.Soft(), share=args.share, server_port=args.port)
