# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional embedding-token budgets for directory text preparation."""

from __future__ import annotations

import json
from bisect import bisect_right
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

from tokenizers.models import BPE
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from data_designer_retrieval_sdg.multimodal.source_models import SourceDirectoryConfig
from data_designer_retrieval_sdg.text_qa.chunking import text_to_sentence_chunks


@dataclass(frozen=True)
class TextTokenBudget:
    """Count untruncated text with a Hugging Face embedding tokenizer."""

    encoder: PreTrainedTokenizerBase
    maximum: int
    reserved: int
    tokenizer_sha256: str

    def count(self, text: str) -> int:
        """Count text, configured special tokens, and downstream overhead.

        Args:
            text: Exact text to be emitted as a retrieval unit.

        Returns:
            Total tokens charged against the configured maximum.
        """
        return len(self.encoder.encode(text, add_special_tokens=True, truncation=False, padding=False)) + self.reserved

    def metadata(self) -> dict[str, str]:
        """Describe the tokenizer used to prepare sources for resume validation.

        Returns:
            Hash of the resolved tokenizer snapshot and tokenizer library versions.
        """
        return {
            "tokenizer_sha256": self.tokenizer_sha256,
            "tokenizers_version": version("tokenizers"),
            "transformers_version": version("transformers"),
        }


def tokenizer_fingerprint(encoder: PreTrainedTokenizerBase) -> str:
    """Hash a saved snapshot of the loaded tokenizer and its configuration.

    Args:
        encoder: Resolved tokenizer with padding and truncation disabled.

    Returns:
        SHA-256 digest covering the tokenizer assets and wrapper configuration.
    """
    with TemporaryDirectory() as directory:
        root = Path(directory)
        encoder.save_pretrained(root)
        hashes = {
            path.relative_to(root).as_posix(): sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }
    return sha256(json.dumps(hashes, sort_keys=True).encode("utf-8")).hexdigest()


def load_text_token_budget(config: SourceDirectoryConfig) -> TextTokenBudget | None:
    """Load a fast tokenizer from a Hugging Face model ID or local directory.

    Args:
        config: Validated source-directory settings.

    Returns:
        Token budget, or None when token limits are disabled.

    Raises:
        ValueError: The tokenizer cannot be loaded, lacks a fast backend, or
            enables BPE dropout.
    """
    if config.max_tokens_per_chunk is None:
        return None
    try:
        encoder = AutoTokenizer.from_pretrained(config.tokenizer, use_fast=True, trust_remote_code=False)
        if not encoder.is_fast:
            raise ValueError("a fast tokenizer is required for original-text offsets")
        backend = encoder.backend_tokenizer
        if isinstance(backend.model, BPE) and backend.model.dropout not in (None, 0.0):
            raise ValueError("BPE dropout must be null or 0 for deterministic token limits and resume")
        backend.no_truncation()
        backend.no_padding()
        fingerprint = tokenizer_fingerprint(encoder)
    except Exception as exc:
        raise ValueError(f"Cannot load source_directory.tokenizer {config.tokenizer}: {exc}") from exc
    return TextTokenBudget(encoder, config.max_tokens_per_chunk, config.reserved_tokens, fingerprint)


def split_sentence(text: str, budget: TextTokenBudget, path: Path) -> list[str]:
    """Split an oversized sentence using verified original-text substrings.

    Token offsets suggest boundaries; each actual substring is encoded again.
    Decoding token slices could change normalized text or corrupt Unicode, so
    all output is sliced from the original string instead. Offset candidates
    are computed once to avoid encoding the entire remainder on every split.

    Args:
        text: One sentence that exceeds the budget.
        budget: Token counter and cap, including overhead.
        path: Source filename for error reporting.

    Returns:
        Ordered nonempty pieces that each fit the budget.

    Raises:
        ValueError: The budget cannot fit a nonempty text prefix.
    """
    offsets = budget.encoder(
        text, add_special_tokens=False, truncation=False, padding=False, return_offsets_mapping=True
    )["offset_mapping"]
    ends = sorted({end for _, end in offsets if end > 0})
    chunks = []
    start = 0
    while start < len(text):
        if text[start].isspace():
            start += 1
            continue
        first = bisect_right(ends, start)
        candidates = ends[first : first + budget.maximum - budget.reserved]
        # A single Unicode character remains a candidate when token offsets
        # span larger strings. Never emit a partial UTF-8 byte sequence.
        candidates = sorted({start + 1, *candidates}, reverse=True)
        for end in candidates:
            piece = text[start:end].strip()
            if piece and budget.count(piece) <= budget.maximum:
                chunks.append(piece)
                start = end
                break
        else:
            raise ValueError(
                f"Cannot split source text {path} within max_tokens_per_chunk={budget.maximum}: "
                "no nonempty prefix fits including special and reserved tokens; increase the budget"
            )
    return chunks


def prepare_text_chunks(
    text: str, config: SourceDirectoryConfig, budget: TextTokenBudget | None, path: Path
) -> list[str]:
    """Pack source text within each configured sentence and token limit.

    Args:
        text: Decoded source file.
        config: Validated directory settings.
        budget: Optional embedding-token counter, loaded once per preparation.
        path: Source filename for error reporting.

    Returns:
        Ordered text units satisfying sentence and optional token limits.

    Raises:
        ValueError: A whole file or an indivisible sentence exceeds the cap.
    """
    if config.text_chunking == "none":
        chunk = text.strip()
        if budget is not None and (count := budget.count(chunk)) > budget.maximum:
            raise ValueError(
                f"Source text {path} uses {count} tokens including special and reserved tokens, exceeding "
                f"max_tokens_per_chunk={budget.maximum}; use text_chunking: sentences or increase the budget"
            )
        return [chunk]
    chunks, pending = [], []
    for record in text_to_sentence_chunks(text, sentences_per_chunk=1):
        sentence = record["text"]
        if budget is not None and (count := budget.count(sentence)) > budget.maximum:
            if not config.split_oversized_sentences:
                raise ValueError(
                    f"Sentence in source text {path} uses {count} tokens including special and reserved tokens, "
                    f"exceeding max_tokens_per_chunk={budget.maximum}; increase the budget or explicitly set "
                    "split_oversized_sentences: true"
                )
            if pending:
                chunks.append(" ".join(pending))
                pending = []
            chunks.extend(split_sentence(sentence, budget, path))
            continue
        sentence_limit = config.sentences_per_chunk is not None and len(pending) >= config.sentences_per_chunk
        token_limit = budget is not None and budget.count(" ".join([*pending, sentence])) > budget.maximum
        if pending and (sentence_limit or token_limit):
            chunks.append(" ".join(pending))
            pending = []
        pending.append(sentence)
    if pending:
        chunks.append(" ".join(pending))
    return chunks
