# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from data_designer_retrieval_sdg.chunking import text_to_sentence_chunks


def test_text_to_sentence_chunks_avoids_double_punctuation():
    text = "Hello world. How are you? I am fine! Thanks for asking"
    chunks = text_to_sentence_chunks(text, sentences_per_chunk=5)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "Hello world. How are you? I am fine! Thanks for asking."


def test_text_to_sentence_chunks_does_not_append_period_when_terminated():
    text = "Hello world. How are you?"
    chunks = text_to_sentence_chunks(text, sentences_per_chunk=5)
    assert chunks[0]["text"] == "Hello world. How are you?"


def test_text_to_sentence_chunks_multiple_chunks():
    text = "First sentence. Second sentence? Third sentence! Fourth sentence. Fifth sentence? Sixth sentence"
    chunks = text_to_sentence_chunks(text, sentences_per_chunk=3)
    assert len(chunks) == 2
