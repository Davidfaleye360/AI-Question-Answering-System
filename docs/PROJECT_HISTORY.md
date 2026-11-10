# Project history

A summary of the planning conversation (Claude Cowork, October–November 2026) that produced this repo,
so work can continue in Claude Code without the original chat. Decisions here are settled unless the
team changes them; update this file and CLAUDE.md when they do.

## The assignment

From Dr. Qi Li's "Project Descriptions: Large Language Model Applications" guide for CSCI 390:

- Pick one of 8 tasks, reproduce an open-source implementation, and present it. Options: text classification,
  **question answering (chosen)**, summarization, translation, text generation, token classification (NER),
  zero-shot classification, table QA.
- Deliverables: GitHub repo (documented, reproducible code, trained models, instructions); technical report
  (2–4 pages: problem and dataset, architecture and training, metrics and analysis, errors and difficulties);
  30-minute interactive presentation (intro 5 min, implementation 5 min, errors and lessons 5 min,
  live demo 10 min, Q&A 5 min). Demos may use Streamlit, Gradio, Flask or a notebook.
- Grading: implementation and code quality 30%, analysis and report 25%, presentation 25%,
  interactive demo 15%, innovation or extension 5%.
- Open question: the guide assumes 10 minutes per member (three people). Ask Dr. Li how a pair splits 30 minutes.

## Why question answering

Compared against NER (easier, less novel), summarization (impressive but GPU-heavy) and text classification
(safest, least impressive). QA won because it's easy to explain, the demo invites the audience to "stump the
model", SQuAD 2.0's unanswerable questions give a real analysis angle, and DistilBERT trains on a free GPU.
Estimated effort: 45–60 hours for the pair over about 4 weeks.

## Team

David Falaye and Kosisochukwu Ogbuanya. Roles: **A = model** (preprocessing, fine-tuning, training
experiments, long-document mode); **B = evaluation and demo** (post-processing, scoring, error analysis,
Gradio app). Both: debugging, documentation, testing, slides, rehearsal. Names not yet assigned to A/B.

## Documents made during planning

These are private claude.ai pages owned by David (share them from each page's Share menu):

- Design doc: https://claude.ai/code/artifact/b2cf1f5c-9865-4ea0-a6d6-c22ffd310453
- Project proposal (merged with the team's own PDF draft): https://claude.ai/code/artifact/75a95cb4-94bd-4e13-8922-ad851db0b610
- Interactive learning primer (12 lessons): https://claude.ai/artifact/5vUttLmPtxkQugFG4cG1Fe
- Project picker (option comparison): https://claude.ai/artifact/T79v6hPKZXAG5JUmcNtKVv

## Key design decisions

**Models**
- Ours: `distilbert-base-uncased` (~66M parameters), fine-tuned on SQuAD 2.0. Expected dev F1 roughly 65–70; target 65+.
- Reference: `deepset/roberta-base-squad2` (~125M), published dev scores EM 79.87 / F1 82.91
  (HasAns F1 84.03 on 5,928 questions, NoAns 81.80 on 5,945). Stretch: fine-tune `roberta-base` ourselves.

**Data**
- `rajpurkar/squad_v2`: train 130,319 questions (about 43.5k unanswerable); dev 11,873 (5,928 answerable,
  5,945 unanswerable). Dev is our test set. Human performance: 86.8 EM / 89.5 F1.

**Preprocessing** (implemented in `src/preprocess.py`)
- `[CLS] question [SEP] passage [SEP]`, `truncation="only_second"`, max_length 384, stride 128,
  offsets via `return_offsets_mapping`. Label (0, 0) = [CLS] for unanswerable questions and for windows
  that don't fully contain the answer. Eval features keep `example_id` and mask non-passage offsets to None.

**Training** (`configs/distilbert.yaml`)
- lr 3e-5, batch 16, 2 epochs, 10% warmup then linear decay, weight decay 0.01, fp16 on a Colab T4,
  seed 42, checkpoint every 2,000 steps (save_total_limit 2) to Google Drive, resume after disconnects,
  push the final model to the Hugging Face Hub. Run order: smoke test (1% of data, ~50 steps) →
  timing run (5%) → full run. Kaggle is the backup GPU.

**Post-processing** (to build in T4)
- Per window: top 20 start and end logits (n_best 20), keep pairs with end ≥ start, both inside the passage,
  span ≤ 30 tokens; score = start logit + end logit; best span across windows.
- Null score = start + end logit at [CLS]. Predict "no answer" if `null − best > τ`. τ = 0 by default;
  sweep τ on dev and pick the best F1. Raising τ makes the model answer more often.

**Evaluation** (T5)
- Official `squad_v2` metric via `evaluate`: EM, F1, HasAns, NoAns. Report τ = 0 and tuned τ, and say τ was
  tuned on dev. Also measure CPU time per question and model size for both models.
- Error analysis on 50+ wrong dev predictions. Categories: boundary error, wrong span, false answer
  (answered an unanswerable question), false refusal, needs reasoning, annotation issue. Also F1 by question
  word and by answer length.

**Demo** (T9)
- Gradio app on a free Hugging Face Space: passage and question boxes, answer highlighted in the passage
  (`HighlightedText`), confidence and top-3 candidates, τ slider, switch between our model and RoBERTa,
  example passages. Live script: easy example → unanswerable question + slider → RoBERTa comparison →
  "stump the model" round from phones via QR code → long-document mode. Backup: recorded video + local notebook.

**Extension (the 5%)**
- Tuned no-answer threshold with a trade-off plot, and long-document QA with overlapping windows
  (test on e.g. the course project guide).

## Proposal (final version)

Title: "An Extractive Question Answering System with DistilBERT and SQuAD 2.0".
- Research questions: how close a small model gets to a larger one; how well it separates answerable from
  unanswerable questions and whether a tuned threshold helps; what errors remain and why.
- Six objectives: fine-tune DistilBERT; evaluate (EM/F1, HasAns/NoAns); compare with RoBERTa (accuracy, CPU
  speed, size); analyze 50+ errors; build the Gradio demo; extend with threshold + long documents.
- Hypotheses: RoBERTa more accurate; DistilBERT faster and smaller; tuning τ raises overall F1; most errors are
  near-miss boundaries or false answers.
- Timeline: W1 study data and model, implement preprocessing, test pipeline on a subset · W2 fine-tune,
  checkpoints, begin evaluation · W3 evaluate both models, error analysis, tune τ, Gradio app ·
  W4 long documents, finish repo and report, slides and rehearsal.

## Implementation log

- **T1 (repo scaffold + data pipeline)**: written and unit-tested in a cloud sandbox (25 offline tests pass;
  an off-by-one bug in label alignment makes 7 of them fail, so they're meaningful). Not yet done:
  `scripts/check_preprocessing.py` on real data (the sandbox couldn't reach huggingface.co), and the first push.
- Library findings (November 2026):
  - transformers 5.x removed the `question-answering` pipeline and replaced `warmup_ratio` with a float
    `warmup_steps`; `Trainer` uses `processing_class=` for the tokenizer. We pinned `transformers>=4.46,<5`.
  - The newest `datasets` needs `huggingface_hub` 1.x+, which transformers 4.x can't use. Installing both in
    one `pip install -r requirements.txt` lets pip pick a compatible `datasets` (5.0.1 worked with hub 0.36.2).
- Notebook `01_explore_data.ipynb` was dry-run on synthetic data; its download/model cells still need a real run in Colab.
- **2026-11-09, moved to Claude Code on David's Mac.** Fresh `.venv` (Python 3.13; transformers 4.57.6,
  datasets 5.0.1, huggingface_hub 0.36.2, torch 2.15.0). 25 unit tests pass. `scripts/check_preprocessing.py
  --n-examples 2000` on real SQuAD 2.0: 2,000 examples → 2,028 windows; 1,310 of 1,314 answerable labels
  decode exactly (99.70%), 4 near matches where the gold answer cuts through a word (`Europe` in `European`,
  `3` in `23`), 0 missing; none of the 686 unanswerable questions got a span. `RESULT: PASS`.
  Local git repo initialised (`main`).
- **2026-11-09, repo moved to David's account.** David only had read access to Kosi's original (empty)
  `Kosiso7-og/AI-Question-Answering-System`, so the project now lives at
  https://github.com/Davidfaleye360/AI-Question-Answering-System, with Kosi (`Kosiso7-og`) invited as a
  collaborator with write access. T1 was pushed to `main` there the same day, and the repo was made public
  so the Colab notebooks can clone it without a login.
- **T2 (explore the data), 2026-11-09.** `notebooks/01_explore_data.ipynb` ran top to bottom on real data on
  David's Mac (CPU), all 14 code cells without errors; not yet run in Colab. Numbers are in
  `results/data_stats.md`: train 130,319 questions (43,498 unanswerable, 33.4%), dev 11,873 (5,945
  unanswerable, 50.1%); passages have a median of 142 tokens and only 0.9% of examples need more than one
  384-token window; dev answers have a median of 2 words. `deepset/roberta-base-squad2` on 50 random dev
  questions scored EM 80.0 / F1 82.7, in line with its published 79.9 / 82.9 (50 questions is a small
  sample, so the close match is partly luck). Notebook fix: the sanity-check cell now calls
  `{sys.executable}` instead of `python`, so it uses the notebook's own Python.
- **T3 (smoke-test training), 2026-11-09.** Wrote `src/train.py` (Hugging Face `Trainer`, resumes from the
  newest checkpoint unless `--fresh`, ends with `RESULT: PASS`/`FAIL`), `configs/smoke.yaml` and
  `notebooks/02_train_colab.ipynb`. Design decisions:
  - Run variants are small YAML files layered on the main config (`load_config(path, overrides=...)`), so the
    smoke test uses the real learning rate, batch size and window size and only changes data fraction, step
    count, logging and saving. New settings: `training.max_steps` and `training.logging_steps`.
  - `fp16` stays `true` in the config and is switched off automatically when there is no CUDA GPU.
  - "Loss went down" compares the average of the first third of the recorded losses with the last third,
    because single batches are noisy.
  - Local result on the Mac's Apple GPU (`mps`, no fp16): 1,303 questions → 1,324 windows; loss 5.86 at
    step 5 (a blind guess among 384 positions is ln 384 ≈ 5.95) down to 3.58 at step 50; `checkpoint-25`,
    `checkpoint-50` and the final model saved; the saved model loads in a `question-answering` pipeline.
    It took 472 s, so anything bigger than the smoke test belongs on Colab.
  - Colab result (2026-11-10, David, Tesla T4, fp16 on): `RESULT: PASS`. Same 1,324 windows; loss 5.860 at
    step 5 down to 3.584 at step 50, almost identical to the Mac run; 50 steps in 30 s (1.7 steps per second,
    about 16 times faster than the Mac). Colab resolved transformers 4.57.6, datasets 4.8.5, torch 2.14.1+cu130.
    What we learned for the full run (T6):
    - The setup cell prints a red pip `ERROR` saying `diffusers` wants a newer `huggingface-hub`. It is
      harmless: Colab preinstalls `diffusers`, we never use it, and transformers 4.x needs the older hub.
    - The TensorFlow lines at the top of the training cell ("Could not find cuda drivers") come from
      TensorFlow being imported in the background, not from PyTorch. The line that counts is
      `device: cuda:0, fp16: True`.
    - Each checkpoint is 797 MB (model plus optimizer state) and the final model is 266 MB, so with
      `save_total_limit: 2` the full run needs about 1.9 GB of Google Drive space.
    - At the smoke test's speed the full run (about 16,500 steps) would take roughly 2.7 hours. Treat that
      as an upper estimate; the first steps of a run are the slowest.
    - After 50 steps the saved model answers the notebook's test question with an empty string (score 0.20),
      which means "no answer". That is expected: it has seen 0.6 of an epoch of 1% of the data.
- 37 offline unit tests now (12 new for training and config overrides), including a 4-step training run of a
  1-layer DistilBERT that checks checkpoints, the saved model and resuming.
- **T4 (post-processing) and T5 (evaluation), 2026-11-10.** Wrote `src/postprocess.py` and `src/evaluate.py`
  with 28 new unit tests (65 in total, still offline and about 5 seconds). Design decisions:
  - Span score = start score + end score. Per window we try the 20 best starts with the 20 best ends and
    drop spans outside the passage, running backwards, or longer than 30 tokens.
  - "No answer" score = start + end score of the first token ([CLS]). For a passage with several windows we
    take the lowest one, so a window that sees the answer can overrule windows that don't. The model
    abstains when `null score - best span score > tau`; tau stays 0.0 until the sweep in T7.
  - That difference (`score_diff`) is saved for every question in `results/<name>_predictions.json`,
    together with the best span, so T7 can sweep tau and T8 can do error analysis without running the
    model again (`apply_threshold` in `src/postprocess.py`).
  - EM and F1 are written out in `src/evaluate.py` (same rules as the official script) because we need a
    score per question for the error analysis and because we have to explain them in the presentation.
    The script also runs the official `squad_v2` metric and stops if the two disagree.
  - The model is run with a plain PyTorch loop instead of `Trainer.predict`: fewer moving parts to explain.
    Each batch is cut at its longest real window, since most passages are far shorter than 384 tokens.
  - Result on the full dev set with `deepset/roberta-base-squad2` (David's Mac, `mps`, 348 s for 12,165
    windows): EM 79.90, F1 82.91 (published 79.9 / 82.9). Answerable: EM 77.99, F1 84.01. Unanswerable:
    81.82. This is the baseline row of the results table, and it shows our preprocessing, post-processing
    and scoring are right before we trust them on our own model.
  - On the first 20 dev questions RoBERTa got 16 exactly right. The 4 misses are the kinds of errors to
    expect in T8: abstaining by a hair on an answerable question ("Rollo", null score ahead by 0.31), a
    span that is too long ("Richard I of Normandy" for "Richard I"), a near-synonym ("Christian" for
    "Catholic"), and answering a trick question (the "country of Normandy").
- **T6 (full training run), code written 2026-11-10; the run itself is still to do.**
  - The model goes to the Hugging Face Hub as `Ingsyling/distilbert-squad2` (David's account), public, so the
    demo Space and anyone checking our results can load it.
  - `python -m src.train ... --push-to-hub` creates the model page BEFORE training starts. A missing or
    read-only token therefore fails in seconds instead of after 2.5 hours on the GPU. The upload happens only
    if the run ends with `RESULT: PASS`, and leaves out the checkpoints (800 MB each, only useful for resuming).
  - The token is never written down: Colab keeps it as a secret named `HF_TOKEN`, the notebook copies it
    into an environment variable, and the Hub library reads it from there. The repo is public.
  - Checkpoints go to Google Drive (`MyDrive/qa-project/distilbert-squad2`) through `--output-dir`. Running
    the training cell again after a disconnect resumes from the newest checkpoint.
  - Notebook 02 now has a Part B (sections 6-8): connect Drive and read the token, train, plot the loss,
    score on the dev set with `src/evaluate.py`.
  - 69 offline unit tests (4 new; the Hub is replaced by a fake object, so no test uses the internet).

## Open questions

- Presentation time for a two-person team (20 or 30 minutes)?
- Who is A and who is B?
- Project deadline and presentation date.
