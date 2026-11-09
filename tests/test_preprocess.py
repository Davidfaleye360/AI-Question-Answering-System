"""Tests for src/preprocess.py: do labels point at the right tokens?"""
from __future__ import annotations

import pytest

from conftest import CONTEXT_CURIE, CONTEXT_FAME, FILLER_WORDS, make_example, to_batch
from src.preprocess import (
    char_span_to_token_span,
    decode_label,
    prepare_eval_features,
    prepare_train_features,
)


def _features(tokenizer, rows, max_length=64, stride=16):
    return prepare_train_features(to_batch(rows), tokenizer, max_length, stride, keep_mapping=True)


def _decoded(feats, rows, i):
    ex = feats["overflow_to_sample_mapping"][i]
    return decode_label(feats["offset_mapping"][i], rows[ex]["context"],
                        feats["start_positions"][i], feats["end_positions"][i])


def test_answerable_label_decodes_to_answer(tokenizer):
    rows = [make_example("q1", "When did Curie win the Chemistry prize?", CONTEXT_CURIE, "1911")]
    feats = _features(tokenizer, rows)
    assert len(feats["input_ids"]) == 1
    assert _decoded(feats, rows, 0) == "1911"


def test_multi_word_answer(tokenizer):
    rows = [make_example("q2", "What did Curie win in 1903?", CONTEXT_CURIE, "the Nobel Prize in Physics")]
    feats = _features(tokenizer, rows)
    assert _decoded(feats, rows, 0) == "the Nobel Prize in Physics"


def test_unanswerable_is_labeled_cls(tokenizer):
    rows = [make_example("q3", "Where was Curie born?", CONTEXT_CURIE, None)]
    feats = _features(tokenizer, rows)
    assert feats["start_positions"] == [0]
    assert feats["end_positions"] == [0]
    assert tokenizer.convert_ids_to_tokens(feats["input_ids"][0][0]) == "[CLS]"


def test_question_whitespace_is_stripped(tokenizer):
    rows = [make_example("q4", "   When did Curie win the Chemistry prize?  ", CONTEXT_CURIE, "1911")]
    feats = _features(tokenizer, rows)
    assert _decoded(feats, rows, 0) == "1911"


def test_answer_inside_a_word_maps_to_the_covering_token(tokenizer):
    # "1990s" is not in the vocabulary, so it becomes "1990" + "##s". A gold answer of
    # "1990" must label just the "1990" piece.
    assert tokenizer.tokenize("1990s") == ["1990", "##s"]
    rows = [make_example("q5", "In what decade?", CONTEXT_FAME, "1990")]
    feats = _features(tokenizer, rows)
    assert _decoded(feats, rows, 0) == "1990"


def test_labels_point_inside_the_passage(tokenizer):
    rows = [make_example("q6", "When did she rise to fame?", CONTEXT_FAME, "in the late 1990s")]
    feats = _features(tokenizer, rows)
    s, e = feats["start_positions"][0], feats["end_positions"][0]
    seq = feats.sequence_ids(0)
    assert seq[s] == 1 and seq[e] == 1 and s <= e


def test_long_passage_is_split_and_only_windows_with_the_whole_answer_are_labeled(tokenizer):
    answer = "the golden gate bridge"
    words = FILLER_WORDS[:90] + answer.split() + FILLER_WORDS[90:110]
    context = " ".join(words) + "."
    rows = [make_example("q7", "What is near the water?", context, answer)]
    max_length, stride = 32, 8
    feats = _features(tokenizer, rows, max_length=max_length, stride=stride)

    n = len(feats["input_ids"])
    assert n > 1, "a long passage should produce several windows"
    assert all(len(ids) == max_length for ids in feats["input_ids"])

    a_start = context.index(answer)
    a_end = a_start + len(answer)
    labeled = 0
    for i in range(n):
        seq = feats.sequence_ids(i)
        ctx = [k for k, sid in enumerate(seq) if sid == 1]
        offs = feats["offset_mapping"][i]
        window_has_answer = offs[ctx[0]][0] <= a_start and offs[ctx[-1]][1] >= a_end
        if window_has_answer:
            labeled += 1
            assert _decoded(feats, rows, i) == answer
        else:
            assert (feats["start_positions"][i], feats["end_positions"][i]) == (0, 0)
    assert labeled >= 1


def test_windows_overlap_by_stride(tokenizer):
    context = " ".join(FILLER_WORDS[:100]) + "."
    rows = [make_example("q8", "What is near the water?", context, None)]
    stride = 8
    feats = _features(tokenizer, rows, max_length=32, stride=stride)
    first = [t for t, s in zip(feats["input_ids"][0], feats.sequence_ids(0)) if s == 1]
    second = [t for t, s in zip(feats["input_ids"][1], feats.sequence_ids(1)) if s == 1]
    assert first[-stride:] == second[:stride]


def test_char_span_outside_window_returns_zero(tokenizer):
    enc = tokenizer("Where?", CONTEXT_CURIE, return_offsets_mapping=True)
    beyond = len(CONTEXT_CURIE) + 5
    assert char_span_to_token_span(enc["offset_mapping"], enc.sequence_ids(0), beyond, beyond + 3) == (0, 0)


def test_batch_with_mixed_examples_keeps_alignment(tokenizer):
    rows = [
        make_example("a", "When did Curie win the Chemistry prize?", CONTEXT_CURIE, "1911"),
        make_example("b", "Where was Curie born?", CONTEXT_CURIE, None),
        make_example("c", "When did she rise to fame?", CONTEXT_FAME, "in the late 1990s"),
    ]
    feats = _features(tokenizer, rows)
    got = [_decoded(feats, rows, i) for i in range(len(feats["input_ids"]))]
    assert got == ["1911", "", "in the late 1990s"]


def test_eval_features_keep_ids_and_mask_non_passage_offsets(tokenizer):
    context = " ".join(FILLER_WORDS[:100]) + "."
    rows = [
        make_example("long", "What is near the water?", context, None),
        make_example("short", "Where was Curie born?", CONTEXT_CURIE, None),
    ]
    feats = prepare_eval_features(to_batch(rows), tokenizer, max_length=32, stride=8)
    ids = feats["example_id"]
    assert ids[0] == "long" and ids[-1] == "short" and ids.count("long") > 1
    for i in range(len(feats["input_ids"])):
        seq = feats.sequence_ids(i)
        for k, off in enumerate(feats["offset_mapping"][i]):
            if seq[k] == 1:
                assert off is not None
            else:
                assert off is None
    assert "start_positions" not in feats


def test_dataset_map_integration(tokenizer):
    datasets = pytest.importorskip("datasets")
    rows = [
        make_example("a", "When did Curie win the Chemistry prize?", CONTEXT_CURIE, "1911"),
        make_example("b", "Where was Curie born?", CONTEXT_CURIE, None),
    ]
    ds = datasets.Dataset.from_list(rows)
    out = ds.map(prepare_train_features, batched=True, remove_columns=ds.column_names,
                 fn_kwargs={"tokenizer": tokenizer, "max_length": 64, "stride": 16})
    assert set(out.column_names) >= {"input_ids", "attention_mask", "start_positions", "end_positions"}
    assert "offset_mapping" not in out.column_names
    assert out["start_positions"][1] == 0 and out["end_positions"][1] == 0
