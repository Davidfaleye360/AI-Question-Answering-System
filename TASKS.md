# Project tasks

The project broken into 14 tasks over 4 weeks. Each task ends with something you can check.
Owners follow the proposal: **A** = model (preprocessing, training, long documents),
**B** = evaluation and demo (post-processing, scoring, error analysis, Gradio). Fill in names when you decide.

| Owner | Name |
| --- | --- |
| A | _to decide_ |
| B | _to decide_ |

## Week 1: set up and get the pipeline working

- [x] **T1. Repo scaffold and data pipeline** (A + B) · _checked 2026-11-09 on David's Mac: 25 tests pass, sanity check PASS on 2,000 real examples (99.70% exact, 0 missing). Pushed to GitHub the same day; the Colab re-run happens in T2._
  Folder structure, requirements, config file, preprocessing module with unit tests,
  data exploration notebook, preprocessing sanity-check script.
  _Done when:_ `pytest` passes locally and `scripts/check_preprocessing.py` passes in Colab.
- [x] **T2. Explore the data and try a ready-made model** (B) · _ran top to bottom on David's Mac (CPU) on 2026-11-09, no errors; numbers are in `results/data_stats.md`. RoBERTa on 50 dev questions: EM 80.0, F1 82.7. Not yet run in Colab._
  Run `notebooks/01_explore_data.ipynb` in Colab. Record dataset stats and the RoBERTa baseline on 50 dev questions.
  _Done when:_ the notebook runs top to bottom and the numbers are copied into `results/data_stats.md`.
- [x] **T3. Smoke-test training** (A) · _`RESULT: PASS` in Colab on a Tesla T4 with fp16 on 2026-11-10 (run by David): loss 5.86 → 3.58 in 50 steps, 30 seconds, two checkpoints and the final model saved. It passed on David's Mac the day before too. Code: `src/train.py`, `configs/smoke.yaml`, `notebooks/02_train_colab.ipynb`._
  Train DistilBERT on 1% of the training data for about 50 steps in Colab.
  _Done when:_ the loss goes down and a checkpoint saves without errors.

**Milestone: pipeline works end to end on a small sample.**

## Week 2: train and start evaluating

- [x] **T4. Post-processing** (B) · _done on David's Mac on 2026-11-10: 10 unit tests pass, and RoBERTa's answers on the first 20 dev questions look right (16 of 20 exact; run `python -m src.evaluate --model deepset/roberta-base-squad2 --limit 20 --show 20 --no-save` to see them)._
  `src/postprocess.py`: turn start/end scores into answer text, with n-best spans, max answer length and the no-answer threshold τ.
  _Done when:_ unit tests pass and the RoBERTa checkpoint's predictions look right on 20 dev questions.
- [x] **T5. Evaluation script** (B) · _done on David's Mac on 2026-11-10: RoBERTa scores 79.90 EM / 82.91 F1 on all 11,873 dev questions (published: 79.9 / 82.9), and our numbers equal the official `squad_v2` metric's. Results are in `results/roberta-base-squad2.json`._
  `src/evaluate.py`: EM, F1, HasAns and NoAns with the official `squad_v2` metric; writes `results/*.json`.
  _Done when:_ RoBERTa scores close to its published 79.9 EM / 82.9 F1 on the dev set.
- [x] **T6. Full training run** (A) · _done in Colab on a Tesla T4 on 2026-11-10 (run by David): 16,470 steps in 44 minutes, loss 5.81 → about 0.9, `RESULT: PASS`. The model is at https://huggingface.co/Ingsyling/distilbert-squad2. Dev set: **EM 64.14, F1 67.38** (answerable F1 72.17, unanswerable 62.59), above the 65 F1 target. Scoring the Hub model on David's Mac gave the same numbers._
  `src/train.py` + `notebooks/02_train_colab.ipynb`: 2 epochs on all of SQuAD 2.0, checkpoints to Google Drive, final model pushed to the Hugging Face Hub.
  _Done when:_ the model is on the Hub and its dev F1 is recorded.

**Milestone: model trained and scored.**

## Week 3: compare, analyze, build the demo

- [x] **T7. Threshold sweep and efficiency comparison** (A) · _done on David's Mac on 2026-11-10. Best τ for our model is about −1.4: F1 67.38 → 68.08 cross-validated (68.30 if tuned on the whole dev set). Ours: 266 MB, 31 ms per question on CPU. RoBERTa: 496 MB, 57 ms. Figure: `results/threshold_sweep.png`._
  Sweep τ on the dev set and plot F1 against it. Measure CPU time per question and model size for both models.
- [x] **T8. Error analysis** (B) · _done on 2026-11-11: `src/errors.py` and `notebooks/03_error_analysis.ipynb`. Error types for all 11,873 questions (76% of our model's errors are answer / no-answer decisions), F1 by question word and answer length, and 60 sampled errors labelled by hand in `results/error_sample.csv`; the labels were proposed during development and read by David._
  `notebooks/03_error_analysis.ipynb`: label at least 50 wrong predictions by type, break F1 down by question word and answer length.
- [x] **T9. Gradio demo** (B) · _done on 2026-11-10. `python app/app.py` runs it locally; `--share` gives a public `gradio.live` link (tested from David's Mac). **Not on a Hugging Face Space:** Hugging Face now requires a paid PRO plan for Gradio Spaces, so we start the demo on demand instead, from `notebooks/04_demo_colab.ipynb` or a laptop (David's decision). Still to check: notebook 04 has not been run in Colab yet._
  `app/app.py`: passage and question inputs, highlighted answer, top candidates, threshold slider, model switch. Deploy to a Hugging Face Space.

**Milestone: demo online.**

## Week 4: extension, report, presentation

- [ ] **T10. Long-document mode** (A)
  Answer questions over long texts using overlapping windows. Test on 3 long documents.
- [ ] **T11. Technical report, 2–4 pages** (A writes method and training, B writes results and analysis)
- [ ] **T12. Slides and rehearsal** (A + B)
  Include a QR code to the demo. Record a 2-minute backup video of the demo.
- [ ] **T13. Final repo cleanup** (A + B)
  README results table, pinned versions, reproduction steps, links to the model and the Space.
- [ ] **T14. Presentation** (A + B)

**Milestone: presentation.**
