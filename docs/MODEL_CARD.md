---
language: en
license: apache-2.0
library_name: transformers
pipeline_tag: question-answering
base_model: distilbert/distilbert-base-uncased
datasets:
  - rajpurkar/squad_v2
metrics:
  - exact_match
  - f1
tags:
  - question-answering
  - extractive-qa
  - squad_v2
  - distilbert
model-index:
  - name: distilbert-squad2
    results:
      - task:
          type: question-answering
          name: Extractive question answering
        dataset:
          name: SQuAD 2.0
          type: rajpurkar/squad_v2
          split: validation
        metrics:
          - type: exact_match
            value: 64.14
            name: Exact Match
          - type: f1
            value: 67.38
            name: F1
---

# DistilBERT fine-tuned on SQuAD 2.0

`distilbert-base-uncased` fine-tuned for **extractive question answering** on SQuAD 2.0. Given a passage and
a question, it marks the words in the passage that answer the question, or says that the passage does not
contain an answer.

A course project for CSCI 390 (Natural Language Processing and Applications) by David Falaye and
Kosisochukwu Ogbuanya. Code, training notebook, evaluation and error analysis:
https://github.com/Davidfaleye360/AI-Question-Answering-System

## How to use it

```python
from transformers import pipeline

qa = pipeline("question-answering", model="Ingsyling/distilbert-squad2")
result = qa(
    question="In what country is Normandy located?",
    context="The Normans gave their name to Normandy, a region in France.",
    handle_impossible_answer=True,   # allow "no answer"; the answer is then an empty string
)
print(result["answer"])   # France
```

This needs `transformers` 4.x (version 5 removed the `question-answering` pipeline). The project's own
code (`src/predict.py` in the repository) does the same without the pipeline and also returns the top
candidates and the no-answer score.

## Results

SQuAD 2.0 dev set, 11,873 questions, scored with the official `squad_v2` metric at the default no-answer
threshold.

| | Exact Match | F1 | Questions |
| --- | --- | --- | --- |
| All | 64.14 | 67.38 | 11,873 |
| Answerable | 65.69 | 72.17 | 5,928 |
| Unanswerable | 62.59 | 62.59 | 5,945 |

For comparison, measured with the same code:

| Model | Exact Match | F1 | Parameters | CPU time per question |
| --- | --- | --- | --- | --- |
| This model | 64.14 | 67.38 | 66 M | 31 ms |
| `deepset/roberta-base-squad2` | 79.90 | 82.91 | 124 M | 57 ms |
| `distilbert-base-uncased` before fine-tuning (first 1,000 questions) | 1.2 | 4.6 | 66 M | |

CPU time is the median over 200 dev questions on an Apple M2 laptop. Tuning the no-answer threshold raises
this model's F1 to 68.08 (cross-validated over dev articles).

## Training

| Setting | Value |
| --- | --- |
| Data | SQuAD 2.0 training set, 130,319 questions (131,754 windows) |
| Window | 384 tokens, 128 tokens of overlap |
| Epochs | 2 (16,470 steps) |
| Batch size | 16 |
| Learning rate | 3e-5, linear warm-up over the first 10% of steps, then linear decay |
| Weight decay | 0.01 |
| Precision | mixed (fp16) |
| Seed | 42 |
| Hardware and time | one Tesla T4 (Google Colab), 44 minutes |

Unanswerable questions, and windows that do not contain the whole answer, are labelled with the `[CLS]`
position. The training loss fell from 5.81 to about 0.9.

## Limitations

- **It copies, it does not reason.** The answer is always one continuous piece of the passage. Questions
  whose answer has to be put together from several sentences, and most "why" questions, are beyond it.
- **It is too willing to answer.** 52% of its errors on the dev set are answers to unanswerable questions,
  typically questions that reuse the passage's wording and change one detail or reverse the meaning.
- **"Why" and "how" questions and long answers are weak**: F1 is 59 for "why" questions and 51 for answers
  of ten or more words, against 85 for "how many".
- **Long documents need care.** Reading a whole article in overlapping windows lowers F1 by 16 to 23 points
  with this model's default settings, mainly because it stops abstaining.
- **English Wikipedia only.** It was trained on Wikipedia paragraphs, uncased, and has not been tested on
  other kinds of text or other languages.
- Do not use it where a wrong answer has consequences (medical, legal, financial) without a person checking.

## Licences

The base model is released under Apache 2.0. SQuAD 2.0 is released under CC BY-SA 4.0.
