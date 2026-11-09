"""Turn SQuAD examples into model inputs ("features") with start/end labels.

Why this is needed:
  * The model reads one sequence: [CLS] question [SEP] passage [SEP].
  * It has a maximum length (384 tokens here), so long passages are split into
    overlapping windows. One example can become several features.
  * SQuAD gives the answer as a CHARACTER position. The model predicts TOKEN
    positions. The tokenizer's offset mapping (each token's start/end character)
    lets us convert between the two.
  * Unanswerable questions, and windows that don't contain the whole answer,
    are labeled (0, 0): position 0 is [CLS], which means "no answer".

Adapted from the Hugging Face LLM Course, chapter 7.7 (question answering).
"""
from __future__ import annotations

from typing import Any

# Sequence ids returned by the tokenizer: 0 = question, 1 = passage, None = special token.
QUESTION_SEQ_ID = 0
CONTEXT_SEQ_ID = 1


def _tokenize(examples: dict[str, list], tokenizer, max_length: int, stride: int):
    """Tokenize question + passage pairs into overlapping, padded windows."""
    # Some SQuAD questions start with spaces, which waste tokens and can break truncation.
    questions = [q.strip() for q in examples["question"]]
    return tokenizer(
        questions,
        examples["context"],
        truncation="only_second",        # only ever cut the passage, never the question
        max_length=max_length,
        stride=stride,                   # tokens shared by neighbouring windows
        return_overflowing_tokens=True,  # long passages -> several windows
        return_offsets_mapping=True,     # (start_char, end_char) for every token
        padding="max_length",
    )


def _context_token_range(sequence_ids: list) -> tuple[int, int]:
    """First and last token index of the passage inside one window."""
    start = sequence_ids.index(CONTEXT_SEQ_ID)
    end = len(sequence_ids) - 1 - sequence_ids[::-1].index(CONTEXT_SEQ_ID)
    return start, end


def answer_char_span(answers: dict) -> tuple[int, int] | None:
    """(start_char, end_char) of the first gold answer, or None if unanswerable."""
    if len(answers["answer_start"]) == 0:
        return None
    start_char = answers["answer_start"][0]
    return start_char, start_char + len(answers["text"][0])


def char_span_to_token_span(
    offsets: list[tuple[int, int]],
    sequence_ids: list,
    start_char: int,
    end_char: int,
) -> tuple[int, int]:
    """Convert an answer's character span to token positions within one window.

    Returns (0, 0) if the answer is not fully inside this window.
    """
    ctx_start, ctx_end = _context_token_range(sequence_ids)

    # Answer not completely inside this window -> label it "no answer here".
    if offsets[ctx_start][0] > start_char or offsets[ctx_end][1] < end_char:
        return 0, 0

    # Walk forward to the first token that starts after start_char; the answer starts one before it.
    idx = ctx_start
    while idx <= ctx_end and offsets[idx][0] <= start_char:
        idx += 1
    start_token = idx - 1

    # Walk backward to the last token that ends before end_char; the answer ends one after it.
    idx = ctx_end
    while idx >= ctx_start and offsets[idx][1] >= end_char:
        idx -= 1
    end_token = idx + 1

    return start_token, end_token


def prepare_train_features(
    examples: dict[str, list],
    tokenizer,
    max_length: int = 384,
    stride: int = 128,
    keep_mapping: bool = False,
) -> Any:
    """Batched function for `Dataset.map`: adds start_positions and end_positions.

    Example:
        train = raw["train"].map(
            prepare_train_features, batched=True,
            fn_kwargs={"tokenizer": tok, "max_length": 384, "stride": 128},
            remove_columns=raw["train"].column_names)

    keep_mapping=True also returns offset_mapping and overflow_to_sample_mapping,
    which the sanity checks use to decode labels. Leave it False for training.
    """
    enc = _tokenize(examples, tokenizer, max_length, stride)
    sample_map = enc.pop("overflow_to_sample_mapping")
    offset_mapping = enc.pop("offset_mapping")

    start_positions, end_positions = [], []
    for i, offsets in enumerate(offset_mapping):
        answers = examples["answers"][sample_map[i]]
        span = answer_char_span(answers)
        if span is None:  # unanswerable question
            start_positions.append(0)
            end_positions.append(0)
            continue
        s, e = char_span_to_token_span(offsets, enc.sequence_ids(i), *span)
        start_positions.append(s)
        end_positions.append(e)

    enc["start_positions"] = start_positions
    enc["end_positions"] = end_positions
    if keep_mapping:
        enc["offset_mapping"] = offset_mapping
        enc["overflow_to_sample_mapping"] = sample_map
    return enc


def prepare_eval_features(
    examples: dict[str, list],
    tokenizer,
    max_length: int = 384,
    stride: int = 128,
) -> Any:
    """Batched function for `Dataset.map` on the dev set.

    Keeps what post-processing needs to turn predictions back into text:
      * example_id: which question each window came from
      * offset_mapping: character offsets, set to None for non-passage tokens
        so a predicted span can never land in the question or on special tokens.
    """
    enc = _tokenize(examples, tokenizer, max_length, stride)
    sample_map = enc.pop("overflow_to_sample_mapping")

    example_ids = []
    for i in range(len(enc["input_ids"])):
        example_ids.append(examples["id"][sample_map[i]])
        seq_ids = enc.sequence_ids(i)
        enc["offset_mapping"][i] = [
            offset if seq_ids[k] == CONTEXT_SEQ_ID else None
            for k, offset in enumerate(enc["offset_mapping"][i])
        ]

    enc["example_id"] = example_ids
    return enc


def decode_label(feature_offsets, context: str, start: int, end: int) -> str:
    """Text a (start, end) label points to, sliced from the original passage.

    Returns "" for the (0, 0) "no answer" label. Used by the sanity checks.
    """
    if start == 0 and end == 0:
        return ""
    return context[feature_offsets[start][0]: feature_offsets[end][1]]
