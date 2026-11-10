---
title: SQuAD 2.0 Question Answering
emoji: 🔎
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.17.3
python_version: "3.12"
app_file: app.py
pinned: false
models:
  - Ingsyling/distilbert-squad2
  - deepset/roberta-base-squad2
datasets:
  - rajpurkar/squad_v2
---

# Question answering on SQuAD 2.0

Paste a passage, ask a question, and the model points to the answer inside the passage, or says that
the passage doesn't contain one.

- **DistilBERT (ours):** fine-tuned by us on SQuAD 2.0, 67.4 F1 on the dev set.
  Model: [Ingsyling/distilbert-squad2](https://huggingface.co/Ingsyling/distilbert-squad2)
- **RoBERTa (deepset):** a published baseline, 82.9 F1, about twice the size and half the speed.

CSCI 390 course project by David Falaye and Kosisochukwu Ogbuanya.
Code, training details and results: https://github.com/Davidfaleye360/AI-Question-Answering-System

This Space is built from that repository by `scripts/deploy_space.py`. Edit the files there, not here.
