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
- [ ] **T3. Smoke-test training** (A)
  Train DistilBERT on 1% of the training data for about 50 steps in Colab.
  _Done when:_ the loss goes down and a checkpoint saves without errors.

**Milestone: pipeline works end to end on a small sample.**

## Week 2: train and start evaluating

- [ ] **T4. Post-processing** (B)
  `src/postprocess.py`: turn start/end scores into answer text, with n-best spans, max answer length and the no-answer threshold τ.
  _Done when:_ unit tests pass and the RoBERTa checkpoint's predictions look right on 20 dev questions.
- [ ] **T5. Evaluation script** (B)
  `src/evaluate.py`: EM, F1, HasAns and NoAns with the official `squad_v2` metric; writes `results/*.json`.
  _Done when:_ RoBERTa scores close to its published 79.9 EM / 82.9 F1 on the dev set.
- [ ] **T6. Full training run** (A)
  `src/train.py` + `notebooks/02_train_colab.ipynb`: 2 epochs on all of SQuAD 2.0, checkpoints to Google Drive, final model pushed to the Hugging Face Hub.
  _Done when:_ the model is on the Hub and its dev F1 is recorded.

**Milestone: model trained and scored.**

## Week 3: compare, analyze, build the demo

- [ ] **T7. Threshold sweep and efficiency comparison** (A)
  Sweep τ on the dev set and plot F1 against it. Measure CPU time per question and model size for both models.
- [ ] **T8. Error analysis** (B)
  `notebooks/03_error_analysis.ipynb`: label at least 50 wrong predictions by type, break F1 down by question word and answer length.
- [ ] **T9. Gradio demo** (B)
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
