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
- **T6 result, 2026-11-10 (David, Colab, Tesla T4, fp16).** 130,319 questions became 131,754 windows; 2 epochs
  = 16,470 steps in 2,616 s (44 minutes, about 6.3 steps per second: much faster than the 2.7 hours guessed
  from the smoke test, whose 30 seconds were mostly start-up). Loss 5.81 at step 100, about 1.2 at the end
  of epoch 1, about 0.9 at the end of epoch 2. The visible drop at step 8,300 is the start of the second
  epoch: the model sees the same questions again. `RESULT: PASS`, uploaded to
  https://huggingface.co/Ingsyling/distilbert-squad2 (final model and tokenizer, no checkpoints).
  - Dev set at tau = 0: **EM 64.14, F1 67.38**. Answerable: EM 65.69, F1 72.17. Unanswerable: 62.59.
    The 65 F1 target is met. Scoring the uploaded model on David's Mac gave identical numbers, so the
    upload is the model that was trained.
  - Against RoBERTa (79.90 / 82.91): 15.5 F1 points lower overall. The gap is larger on unanswerable
    questions (19.2 points) than on answerable ones (11.8), so knowing when NOT to answer is where the
    smaller model loses most. That is the first thing to look at in the threshold sweep (T7) and the
    error analysis (T8).
  - Files in `results/`: `distilbert-squad2.json`, `distilbert-squad2_predictions.json` and
    `distilbert-squad2_train_summary.json` (settings and the 164 recorded losses, for the loss-curve figure).
- **T7 (threshold sweep and efficiency), 2026-11-10.** New modules `src/threshold.py`, `src/predict.py` and
  `src/benchmark.py`; 19 new unit tests, 88 in total.
  - The sweep needs no model: it re-decides every question from the `best_span` and `score_diff` saved by
    `src/evaluate.py`. At tau = 0 it reproduces the reported scores exactly (the script checks this).
  - Both models do best with a NEGATIVE tau, which means abstaining more often than the default.
    DistilBERT: best tau -1.38, F1 67.38 -> 68.30, answering 52% of questions instead of 60%. Answerable F1
    falls from 72.2 to 66.3 while unanswerable F1 rises from 62.6 to 70.3. RoBERTa: best tau -1.64,
    82.91 -> 83.50. The gain is small for both: the curve is flat around its peak.
  - Honest number: picking tau on the same questions it is scored on flatters it. We split the 35 dev
    articles into two halves, pick tau on one half and score the other. Cross-validated F1: DistilBERT
    68.08 (the two halves picked -2.29 and -1.37), RoBERTa 83.44 (-1.51 and -1.64). Report 68.08, not 68.30.
    The halves disagree by almost a point of tau for DistilBERT, which again says the peak is flat.
  - `configs/distilbert.yaml` keeps `null_threshold: 0.0`, so the main results table stays at the default.
    The demo slider can start at -1.4 for our model.
  - `src/predict.py` has `answer_question(model, tokenizer, question, context, cfg)`: one question, same
    functions as the evaluation, and a test that both paths give the same answer. The Gradio demo (T9)
    and the long-document mode (T10) will call it.
  - Efficiency, CPU only, one question at a time, 200 random dev questions, median, on David's Mac (Apple
    M2, 4 threads): DistilBERT 66.4 M parameters / 265.5 MB / 31.2 ms; RoBERTa 124.1 M / 496.2 MB / 56.7 ms.
    So ours is 54% of the size and 1.82 times as fast, for 15.5 fewer F1 points. Times will differ on the
    Space's CPU; the ratio should hold roughly.
- **T9 (Gradio demo), written 2026-11-10; deployment still to do.** New: `src/demo.py`, `app/app.py`,
  `app/README.md`, `app/requirements.txt`, `scripts/deploy_space.py`; 14 new unit tests, 102 in total.
  - Split in two on purpose. `src/demo.py` decides what is shown and has no Gradio code, so it is tested;
    `app/app.py` only lays out the screen.
  - The model runs once per question (`ask`) and its scores are kept in the visitor's session. Moving the
    threshold slider only re-decides from those scores (`render`), so the answer flips instantly. That is
    the clearest way to show what tau does in the presentation.
  - When the model abstains, the passage still shows its best guess in grey, with a sentence saying by how
    much "no answer" won and which threshold would flip it.
  - The "confidence" column is a softmax over the five listed spans and "no answer". It is a guide, not a
    calibrated probability, and the app's code says so.
  - Slider defaults are the tuned thresholds from T7: -1.4 for DistilBERT, -1.6 for RoBERTa.
  - Tried in a browser on David's Mac. Good material for the live demo, same Normans passage:
    "What is France a region of?" (no answer in the passage) - our model confidently says "Normandy",
    RoBERTa correctly abstains. And on the oxygen passage, "Which element is the most abundant in the
    universe by mass?": our model abstains at -1.4 and answers "oxygen" (wrong) once the slider passes -1.14.
  - First answer takes about 0.5 s for DistilBERT and 1.1 s for RoBERTa on the Mac, including set-up;
    the model itself needs 31 / 57 ms (T7).
  - Deployment: `scripts/deploy_space.py` copies the app, the five `src/` modules it needs, the config
    and a CPU-only requirements file into the layout a Space expects and uploads it to
    `Ingsyling/squad2-qa-demo`. A test follows the app's imports and fails if a needed module isn't copied.
    A dry run and an import of the copied app both work; the real upload needs David's Hugging Face login.
- **T9 hosting decision, 2026-11-10.** `python scripts/deploy_space.py` failed at creating the Space with
  "402 Payment Required: hosting Gradio and Docker Spaces on free cpu-basic requires a PRO subscription".
  Options were: (A) no permanent hosting, start the demo on demand with a public Gradio link; (B) pay for
  PRO; (C) rewrite the demo as a static Space that runs the model in the browser with JavaScript.
  **David chose A.** C would have thrown away the tested Python path.
  - `python app/app.py --share` prints a `https://....gradio.live` link. Tested from David's Mac: the link
    served the app. The model still runs on the computer that started it; the link forwards visitors
    there, so it stops when the program stops and expires after a week.
  - `notebooks/04_demo_colab.ipynb` starts the same thing in Colab (free CPU is enough), so anyone with a
    Google account can run the demo without installing anything. Not yet run in Colab.
  - `requirements.txt` now pins `gradio>=6.0,<7`: the app passes its theme to `launch()`, which older
    Gradio versions (such as one Colab may have preinstalled) don't accept.
  - The Space card and `scripts/deploy_space.py` stay in the repo; they work for an account with PRO.
  - For the presentation: start the demo beforehand and put that day's link in the QR code (T12). A fixed
    QR code can point at the README's "Try the demo" section instead. Worth asking Dr. Li whether a demo
    run live from Colab meets the "deployed demo" expectation, since the proposal said Hugging Face Space.
- **T8 (error analysis), 2026-11-10; labels read by David and task ticked on 2026-11-11.** New: `src/errors.py`,
  `notebooks/03_error_analysis.ipynb`, 22 new unit tests (124 in total).
  - Automatic error types for every dev question (one each): false answer, missed answer, too long, too
    short, partial overlap, wrong span. DistilBERT has 4,258 errors: 2,224 false answers (52.2%), 1,033
    missed answers (24.3%), 341 too long, 232 too short, 89 partial overlap, 339 wrong span. RoBERTa has
    2,386: 1,081 / 572 / 269 / 228 / 63 / 173. So 76% of our errors are answer / no-answer decisions, and
    most of RoBERTa's advantage (1,143 of its 1,872 fewer errors) is fewer false answers.
  - F1 by question word, answerable questions only (ours / RoBERTa): how many 85.0 / 90.5, when 84.2 / 86.8,
    how much 78.0 / 83.4, which 77.2 / 88.7, who 75.3 / 85.2, where 71.1 / 84.9, what 69.9 / 83.5,
    how 64.2 / 80.4, why 58.7 / 70.1. Numbers and dates are easy; explanations are hard for both models.
  - F1 by gold answer length (ours / RoBERTa): 1 word 73.5 / 83.7, 2 words 71.5 / 84.3, 3-4 words
    73.8 / 86.5, 5-9 words 67.3 / 81.5, 10+ words 50.9 / 72.9.
  - Hand labels: a fixed random sample (seed 42) of 60 of our model's errors, every type represented,
    in `results/error_sample.csv` with the passage around the gold answer and the prediction. Labels were
    proposed during development after reading each one; David and Kosi read them and
    correct any they disagree with (the file has no sign-off column: David's choice, 2026-11-11). Re-running `python -m src.errors` keeps labels already in the file.
    Counts out of 60: swapped detail 14, reversed meaning 10, over-cautious (had the right span) 10,
    boundary but acceptable 10, distractor 7, missed with a wrong best span 5, not stated 2, gold label
    debatable 1, needs reasoning 1.
  - What it means: (1) 24 of 27 sampled false answers are trick questions that reuse the passage's wording
    and change one detail or flip the meaning; the model matches wording. (2) In 10 of 15 missed answers
    the best span was right, so these are threshold cases. (3) All 9 sampled too long / too short answers
    would be accepted by a person, so EM understates quality. (4) Only 1 of 60 needed reasoning.
  - An out-of-dataset example David found in the demo (a fresco restoration passage, "Why did the museum
    board decide to stop using the chemical treatments?") is in the notebook: the reason is never stated
    in one span, and both DistilBERT and RoBERTa return the neighbouring clause instead of abstaining.
    It goes in the report's limitations: extractive models copy one span and match wording.
- **Is fine-tuning doing anything? (David's question, 2026-11-11.)** Yes. On the first 1,000 dev questions
  the untrained `distilbert-base-uncased` (random QA head) scores EM 1.2 / F1 4.6; our fine-tuned model
  scores 64.2 / 67.7 on the same questions. RoBERTa in the demo is a different, larger model fine-tuned by
  deepset, not "our model before training". More data is not available (we used all of SQuAD 2.0), and a
  third epoch would add little; the remaining gap to RoBERTa is model size, which was our trade-off.
- **T10 (long-document mode), 2026-11-11.** New: `src/longdoc.py`, a `window_rule` option in
  `src/postprocess.py`, 9 new unit tests (133 in total).
  - Test design: instead of a few hand-picked questions, rebuild three dev articles from their paragraphs
    (Amazon_rainforest 2,941 tokens, Normans 5,294, Oxygen 6,696) and ask all 907 of their SQuAD questions
    against the whole article. Gold answers are known, so EM and F1 are exact, and the same questions'
    paragraph-level scores are the baseline.
  - F1 at tau 0, paragraph -> whole article with the evaluation's "global" rule: 70.6 -> 54.5,
    60.1 -> 37.3, 59.3 -> 41.5. Answerable F1 holds up better (81.1 -> 80.8, 79.2 -> 65.2, 67.2 -> 62.4)
    and the span is in the right paragraph 94.0 / 73.4 / 72.8% of the time. Unanswerable F1 collapses
    (51.5 -> 6.9, 43.8 -> 13.4, 48.3 -> 12.6).
  - Why: the global rule takes the HIGHEST span score and the LOWEST no-answer score over 12 to 28
    windows per question. With that many windows one of each extreme always turns up, so the model
    almost never abstains.
  - "per_window" rule: each window is judged by its own no-answer score minus its own best span, and
    the most confident window decides. F1 at tau 0: 62.0 / 44.0 / 47.7 (+7.5 / +6.7 / +6.2). It is the
    same as the global rule when there is one window, so the dev-set results are unchanged.
  - Re-tuning tau (optimistic, tuned on the same questions) helps both rules and brings them close:
    global 63.8 / 66.0 / 54.8 at tau -8.6 / -10.3 / -7.8; per-window 64.8 / 60.4 / 51.9 at -2.9 / -8.9 /
    -3.8. So the honest summary is: the single-paragraph threshold does not carry over to long documents;
    without re-tuning, the per-window rule is the safer default. At these thresholds the model abstains
    heavily (answerable F1 28-66), so the gain comes from unanswerable questions.
  - Caveat: gold "unanswerable" labels were written for the paragraph. Some of those questions may be
    answerable from elsewhere in the article, so long-document NoAns scores are a lower bound.
  - Cost: 0.22 / 0.39 / 0.58 s per question on the Mac's GPU, growing with the number of windows.
  - Demo: `src/demo.py` now uses the per-window rule and says how many windows a long passage needed.
- **T13 (repo cleanup), 2026-11-11.** David approved the model card and it was uploaded to the model's
  Hugging Face page the same day (licence Apache 2.0).
  - `requirements-lock.txt` holds the exact versions on David's Mac (Python 3.13.1, transformers 4.57.6,
    datasets 5.0.1, torch 2.15.0, gradio 6.17.3, ...) and records the versions of the Colab training run.
    `requirements.txt` keeps ranges on purpose: pinning torch would replace Colab's GPU build.
  - README: "Getting started" is now followed by "Reproducing our results", eight steps with the command,
    the time it takes and the number to expect. Steps 3 to 7 use the model from the Hub, so the results
    can be reproduced without retraining. Added a Links section and `docs/` to the layout.
  - `docs/MODEL_CARD.md`: usage example (run and checked: answers "France", and "" for a question the
    passage can't answer), results, the comparison table, training settings and limitations taken from
    the error analysis. Its header declares the licence as Apache 2.0, the same as the base model; that
    is David's decision to confirm. It becomes the README of the model's Hugging Face page once approved.
  - The repo itself has no LICENSE file. Without one, others may read the code but have no stated right
    to reuse it. Open question for David and Kosi.
  - Checked: no unused imports in `src/`, `app/`, `scripts/`, `tests/`; 133 tests pass.

## Open questions
- ~~Add a LICENSE file to the repo, and is Apache 2.0 right for the model?~~ Answered 2026-11-11: David
  approved Apache 2.0 for the model and asked for an MIT licence for the code (`LICENSE`, both names).

- ~~Presentation time for a two-person team (20 or 30 minutes)?~~ Answered 2026-11-10: 30 minutes.
- ~~Does a demo started live from Colab count as deployed?~~ Answered 2026-11-10: yes.
- Who is A and who is B?
- Project deadline and presentation date.
