# AI Question Answering System

An extractive question answering system for CSCI 390 (Natural Language Processing and Applications, Dr. Qi Li).
Given a passage and a question, the model highlights the words in the passage that answer it, or says
**"no answer"** when the passage doesn't contain one.

We fine-tune [`distilbert-base-uncased`](https://huggingface.co/distilbert/distilbert-base-uncased) on
[SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) and compare it with the published
[`deepset/roberta-base-squad2`](https://huggingface.co/deepset/roberta-base-squad2).

**Team:** David Falaye, Kosisochukwu Ogbuanya

## Status

| Week | Milestone | Status |
| --- | --- | --- |
| 1 | Pipeline works on a small sample | In progress |
| 2 | Model trained and scored | Not started |
| 3 | Demo online | Not started |
| 4 | Report and presentation | Not started |

See [TASKS.md](TASKS.md) for the full task list.

## Results

_To be filled in after training (week 2)._

| Model | EM | F1 | HasAns F1 | NoAns F1 | Size | CPU time / question |
| --- | --- | --- | --- | --- | --- | --- |
| DistilBERT (ours) | | | | | | |
| RoBERTa (deepset) | | | | | | |

## Repository layout

```text
AI-Question-Answering-System/
├── README.md
├── TASKS.md                    # project task list
├── requirements.txt
├── configs/
│   ├── distilbert.yaml         # every setting in one place
│   └── smoke.yaml              # the few settings the 50-step smoke test changes
├── src/
│   ├── config.py               # loads the YAML config
│   ├── data.py                 # loads SQuAD 2.0
│   ├── preprocess.py           # tokenization, windows, start/end labels
│   ├── explore.py              # dataset statistics for the exploration notebook
│   └── train.py                # fine-tuning with the Hugging Face Trainer
├── scripts/
│   └── check_preprocessing.py  # sanity check: labels decode back to the right answers
├── notebooks/
│   ├── 01_explore_data.ipynb   # dataset stats + a ready-made model as a baseline
│   └── 02_train_colab.ipynb    # training in Colab (smoke test now, full run later)
├── tests/                      # unit tests (no internet or GPU needed)
├── pytest.ini
└── results/
    └── data_stats.md           # dataset statistics from notebook 01 (more results added later)
```

Coming next: `src/postprocess.py`, `src/evaluate.py`, `notebooks/03_error_analysis.ipynb`, `app/app.py`.

## Getting started

### In Google Colab (recommended: free GPU)

1. Open a notebook from this repo in Colab: **File → Open notebook → GitHub**, paste the repo URL.
2. **Runtime → Change runtime type → T4 GPU.**
3. Run the first cell. It clones the repo and installs the requirements.

### On your own computer

```bash
git clone https://github.com/Davidfaleye360/AI-Question-Answering-System.git
cd AI-Question-Answering-System
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the unit tests (no internet or GPU needed):

```bash
pytest -q
```

Check preprocessing on real SQuAD data (downloads the dataset and tokenizer, about 50 MB):

```bash
python scripts/check_preprocessing.py --n-examples 2000
```

Run the training smoke test: 50 steps on 1% of the data. It works without a CUDA GPU (about 8 minutes on
an M2 MacBook with 8 GB of memory) and writes about 1.8 GB to `outputs/smoke`. It ends with `RESULT: PASS`
if the loss went down and the model was saved:

```bash
python -m src.train --config configs/distilbert.yaml --overrides configs/smoke.yaml --fresh
```

## How the pipeline works

1. **Load** SQuAD 2.0 from the Hugging Face Hub (`rajpurkar/squad_v2`).
2. **Preprocess** (`src/preprocess.py`): tokenize question and passage together, split long passages into
   overlapping windows (384 tokens, 128 overlap), and convert each answer's character position into
   start and end token labels. Unanswerable questions, and windows that don't contain the whole answer,
   are labeled `(0, 0)`, which points at the `[CLS]` token.
3. **Train** (`src/train.py`): fine-tune DistilBERT with the Hugging Face `Trainer`, saving checkpoints
   along the way so a run can continue after a Colab disconnect. The full run comes in week 2.
4. **Post-process** start and end scores into answer text, with a tunable no-answer threshold (week 2).
5. **Evaluate** with the official SQuAD 2.0 metric: Exact Match and F1 (week 2).

## Configuration

All settings live in [`configs/distilbert.yaml`](configs/distilbert.yaml), so experiments are easy to
reproduce and compare. Don't hard-code numbers in notebooks; read them from the config.

## References

- Rajpurkar, Jia & Liang (2018). [Know What You Don't Know: Unanswerable Questions for SQuAD](https://arxiv.org/abs/1806.03822).
- Sanh et al. (2019). [DistilBERT, a distilled version of BERT](https://arxiv.org/abs/1910.01108).
- Hugging Face. [LLM Course, chapter 7.7: Question answering](https://huggingface.co/learn/llm-course/chapter7/7).
