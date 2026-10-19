# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Directory token bounds preserve defaults, source text, and resume identity."""

import json
import re
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError
from test_multimodal_sdg import ScriptedInference
from test_source_directory import UnexpectedInference, directory_config
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from data_designer_retrieval_sdg.multimodal import SourceDirectoryConfig, run_multimodal_sdg
from data_designer_retrieval_sdg.multimodal.source_directory import load_directory_sources


def save_embedding_tokenizer(tmp_path: Path, *, byte_level: bool = False, specials: bool = True) -> str:
    """Save a deterministic local tokenizer for public source-loader tests.

    Args:
        tmp_path: Temporary asset directory, outside the source corpus.
        byte_level: Use UTF-8 byte tokens instead of whitespace-delimited words.
        specials: Add two postprocessor special tokens.

    Returns:
        Path to the saved Hugging Face tokenizer directory.
    """
    if byte_level:
        vocab = {char: index for index, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
        encoder = Tokenizer(models.BPE(vocab=vocab, merges=[]))
        encoder.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    else:
        encoder = Tokenizer(models.WordLevel(vocab={"[UNK]": 0}, unk_token="[UNK]"))
        encoder.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    if specials:
        encoder.add_special_tokens(["[CLS]", "[SEP]"])
        encoder.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]",
            special_tokens=[(token, encoder.token_to_id(token)) for token in ("[CLS]", "[SEP]")],
        )
    path = tmp_path / "embedding-tokenizer"
    PreTrainedTokenizerFast(tokenizer_object=encoder).save_pretrained(path)
    return str(path)


def save_bpe_tokenizer(tmp_path: Path, dropout: float | None) -> str:
    """Save a byte-level tokenizer with merges and explicit BPE dropout.

    Args:
        tmp_path: Temporary tokenizer directory, outside the source corpus.
        dropout: BPE merge-dropout probability, or None to disable dropout.

    Returns:
        Path to the saved Hugging Face tokenizer directory.
    """
    vocab = {char: index for index, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    vocab.update({piece: len(vocab) + index for index, piece in enumerate(("aa", "aaaa", "aaaaaaaa"))})
    encoder = Tokenizer(models.BPE(vocab=vocab, merges=[("a", "a"), ("aa", "aa"), ("aaaa", "aaaa")], dropout=dropout))
    encoder.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    path = tmp_path / "embedding-tokenizer"
    PreTrainedTokenizerFast(tokenizer_object=encoder).save_pretrained(path)
    return str(path)


def text_config(tmp_path: Path, text: str, **settings) -> SourceDirectoryConfig:
    """Configure one source file with a local embedding tokenizer.

    Args:
        tmp_path: Temporary test directory.
        text: Original source text.
        **settings: Overrides for token and sentence limits.

    Returns:
        Validated directory configuration.
    """
    directory = tmp_path / "documents"
    directory.mkdir()
    (directory / "manual.txt").write_bytes(text.encode("utf-8"))
    tokenizer = save_embedding_tokenizer(tmp_path)
    return SourceDirectoryConfig.model_validate(
        {"path": directory, "max_tokens_per_chunk": 7, "tokenizer": tokenizer, "reserved_tokens": 1} | settings
    )


@pytest.mark.parametrize(
    "settings",
    [
        {"max_tokens_per_chunk": 0},
        {"tokenizer": ""},
        {"tokenizer": None},
        {"max_tokens_per_chunk": None, "reserved_tokens": 1},
        {"reserved_tokens": -1},
        {"max_tokens_per_chunk": None, "split_oversized_sentences": True},
        {"max_tokens_per_chunk": 8, "tokenizer": "./tokenizer", "reserved_tokens": 8},
        {
            "max_tokens_per_chunk": 8,
            "tokenizer": "./tokenizer",
            "text_chunking": "none",
            "split_oversized_sentences": True,
        },
    ],
)
def test_incomplete_or_invalid_budget_settings_fail(tmp_path, settings):
    with pytest.raises(ValidationError):
        SourceDirectoryConfig(path=tmp_path, **settings)


def test_defaults_use_nemotron_and_only_the_1024_token_cap(tmp_path, monkeypatch):
    corpus = text_config(tmp_path, "First sentence. " * 700)
    encoder = AutoTokenizer.from_pretrained(corpus.tokenizer)
    loader = Mock(return_value=encoder)
    monkeypatch.setattr("data_designer_retrieval_sdg.multimodal.text_preparation.AutoTokenizer.from_pretrained", loader)
    config = SourceDirectoryConfig(path=corpus.path)
    sources, _ = load_directory_sources(config)
    assert config.max_tokens_per_chunk == 1024
    assert config.sentences_per_chunk is None
    loader.assert_called_once_with("nvidia/Nemotron-3.5-Embed-1B-BF16", use_fast=True, trust_remote_code=False)
    assert [len(encoder.encode(source.text)) for source in sources] == [1024, 380]


@pytest.mark.parametrize(
    ("sentences", "tokens", "groups"),
    [
        (1, 100, [1, 1, 1]),
        (2, 100, [2, 1]),
        (5, 5, [1, 1, 1]),
        (None, 7, [2, 1]),
        (None, None, [3]),
        (2, None, [2, 1]),
    ],
)
def test_sentence_and_token_caps_apply_independently(tmp_path, sentences, tokens, groups):
    config = text_config(
        tmp_path,
        "First sentence. Second sentence! Third sentence?",
        sentences_per_chunk=sentences,
        max_tokens_per_chunk=tokens,
        reserved_tokens=1 if tokens else 0,
    )
    sources, _ = load_directory_sources(config)
    assert [len(source.text.split()) // 2 for source in sources] == groups
    assert " ".join(source.text for source in sources) == "First sentence. Second sentence! Third sentence?"


def test_slow_tokenizer_is_rejected_with_actionable_error(tmp_path, monkeypatch):
    config = text_config(tmp_path, "Valid sentence.")
    loader = Mock(return_value=SimpleNamespace(is_fast=False))
    monkeypatch.setattr("data_designer_retrieval_sdg.multimodal.text_preparation.AutoTokenizer.from_pretrained", loader)
    with pytest.raises(ValueError, match="fast tokenizer is required"):
        load_directory_sources(config)


def test_images_only_do_not_load_default_embedding_tokenizer(tmp_path, monkeypatch):
    config = directory_config(tmp_path)
    for path in config.source_directory.path.glob("*.*"):
        if path.suffix != ".png":
            path.unlink()
    monkeypatch.setattr("data_designer_retrieval_sdg.multimodal.text_preparation.AutoTokenizer", UnexpectedInference)
    sources, manifest = load_directory_sources(SourceDirectoryConfig(path=config.source_directory.path))
    assert len(sources) == 1 and sources[0].images
    assert "source_preparation" not in manifest


def test_exact_limit_includes_special_and_reserved_tokens(tmp_path):
    config = text_config(tmp_path, "First sentence. Second sentence! Third sentence?")
    sources, _ = load_directory_sources(config)
    assert [source.text for source in sources] == ["First sentence. Second sentence!", "Third sentence?"]
    encoder = Tokenizer.from_file(str(Path(config.tokenizer) / "tokenizer.json"))
    assert [len(encoder.encode(source.text).ids) + 1 for source in sources] == [7, 5]
    assert [source.unit_id for source in sources] == ["manual.txt#chunk-000001", "manual.txt#chunk-000002"]


@pytest.mark.parametrize("sentences_per_chunk", [1, 2, 5])
def test_large_cap_preserves_shared_sentence_boundaries(tmp_path, sentences_per_chunk):
    config = text_config(
        tmp_path,
        "Dr. Smith opens the valve. Pressure rises!\n\nCheck the gauge? Close the valve. Stop safely.",
        max_tokens_per_chunk=1000,
        sentences_per_chunk=sentences_per_chunk,
    )
    bounded, bounded_manifest = load_directory_sources(config)
    uncapped = SourceDirectoryConfig(
        path=config.path, max_tokens_per_chunk=None, sentences_per_chunk=sentences_per_chunk
    )
    default, default_manifest = load_directory_sources(uncapped)
    assert bounded == default
    assert bounded_manifest["files"] == default_manifest["files"]


def test_counts_joined_text_instead_of_summing_sentence_counts(tmp_path):
    config = text_config(tmp_path, "Hi! Go!", max_tokens_per_chunk=6, reserved_tokens=0)
    save_embedding_tokenizer(tmp_path, byte_level=True, specials=False)
    encoder = Tokenizer.from_file(str(Path(config.tokenizer) / "tokenizer.json"))
    assert len(encoder.encode("Hi!").ids) + len(encoder.encode("Go!").ids) == 6
    assert len(encoder.encode("Hi! Go!").ids) == 7
    sources, _ = load_directory_sources(config)
    assert [source.text for source in sources] == ["Hi!", "Go!"]


def test_saved_tokenizer_padding_and_truncation_do_not_hide_overflow(tmp_path):
    config = text_config(tmp_path, "One two three four five six seven.")
    encoder = Tokenizer.from_file(str(Path(config.tokenizer) / "tokenizer.json"))
    encoder.enable_truncation(max_length=4)
    encoder.enable_padding(length=100)
    encoder.save(str(Path(config.tokenizer) / "tokenizer.json"))
    with pytest.raises(ValueError, match=r"manual.txt uses 10 tokens.*split_oversized_sentences"):
        load_directory_sources(config)
    (config.path / "manual.txt").write_text("Small sentence.")
    sources, _ = load_directory_sources(config)
    assert [source.text for source in sources] == ["Small sentence."]


@pytest.mark.parametrize("dropout", [0.5, 1.0])
@pytest.mark.parametrize("mode", ["sentences", "none"])
def test_bpe_dropout_rejected_before_inference_and_run_creation(tmp_path, monkeypatch, dropout, mode):
    config = directory_config(tmp_path)
    config.source_directory = SourceDirectoryConfig(
        path=config.source_directory.path,
        text_chunking=mode,
        max_tokens_per_chunk=1000,
        tokenizer=save_bpe_tokenizer(tmp_path, dropout),
    )
    monkeypatch.setattr("data_designer_retrieval_sdg.multimodal.workflow.DataDesignerInference", UnexpectedInference)
    with pytest.raises(ValueError, match=r"BPE dropout.*null or 0") as error:
        run_multimodal_sdg(config)
    assert str(config.source_directory.tokenizer) in str(error.value)
    assert not config.output_dir.exists()


@pytest.mark.parametrize("dropout", [None, 0.0])
def test_disabled_bpe_dropout_preserves_token_cap_and_repeatable_sources(tmp_path, dropout):
    text = "aaaaaaaa " * 64
    config = text_config(tmp_path, text, max_tokens_per_chunk=32, reserved_tokens=0, split_oversized_sentences=True)
    save_bpe_tokenizer(tmp_path, dropout)
    sources, manifest = load_directory_sources(config)
    repeated, repeated_manifest = load_directory_sources(config)
    encoder = Tokenizer.from_file(str(Path(config.tokenizer) / "tokenizer.json"))
    assert len(sources) > 1
    assert sources == repeated and manifest == repeated_manifest
    assert all(len(encoder.encode(source.text).ids) <= 32 for source in sources)
    assert re.sub(r"\s", "", "".join(source.text for source in sources)) == re.sub(r"\s", "", text)


@pytest.mark.parametrize("byte_level", [False, True])
def test_explicit_splitting_preserves_unicode_text_without_decoding(tmp_path, byte_level):
    text = "Start here. " + "CAFÉ e\u0301 中文 🙂 Straße evidence " * 3 + "ends here. Finish safely."
    config = text_config(tmp_path, text, max_tokens_per_chunk=15, split_oversized_sentences=True)
    save_embedding_tokenizer(tmp_path, byte_level=byte_level)
    encoder = Tokenizer.from_file(str(Path(config.tokenizer) / "tokenizer.json"))
    encoder.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase()])
    encoder.save(str(Path(config.tokenizer) / "tokenizer.json"))
    sources, _ = load_directory_sources(config)
    assert len(sources) > 1
    assert re.sub(r"\s", "", "".join(source.text for source in sources)) == re.sub(r"\s", "", text)
    assert all(len(encoder.encode(source.text).ids) + config.reserved_tokens <= 15 for source in sources)
    assert all(source.text.strip() for source in sources)


@pytest.mark.parametrize("cap", [1, 2, 3])
def test_unrepresentable_character_fails_without_corruption(tmp_path, cap):
    config = text_config(tmp_path, "🙂", max_tokens_per_chunk=cap, reserved_tokens=0, split_oversized_sentences=True)
    save_embedding_tokenizer(tmp_path, byte_level=True, specials=False)
    with pytest.raises(ValueError, match=r"Cannot split source text .*manual.txt.*no nonempty prefix fits"):
        load_directory_sources(config)


def test_budget_too_small_for_special_tokens_fails(tmp_path):
    config = text_config(tmp_path, "Words here.", max_tokens_per_chunk=2, reserved_tokens=0)
    with pytest.raises(ValueError, match="exceeding max_tokens_per_chunk=2"):
        load_directory_sources(config)


def test_whole_file_budget_validates_without_changing_formatting(tmp_path, monkeypatch):
    text = "  Heading\r\n\r\nColumn A\tColumn B\n"
    config = text_config(tmp_path, text, text_chunking="none", max_tokens_per_chunk=8)
    monkeypatch.setattr(
        "data_designer_retrieval_sdg.multimodal.text_preparation.text_to_sentence_chunks", UnexpectedInference
    )
    sources, _ = load_directory_sources(config)
    assert [source.text for source in sources] == [text.strip()]
    with pytest.raises(ValueError, match=r"manual.txt uses 8 tokens.*text_chunking: sentences"):
        load_directory_sources(config.model_copy(update={"max_tokens_per_chunk": 7}))


@pytest.mark.parametrize("content", [None, "not JSON", "{}", "\ufffd"])
def test_missing_or_invalid_tokenizer_has_actionable_error(tmp_path, content):
    config = text_config(tmp_path, "Valid source.")
    if content is None:
        (Path(config.tokenizer) / "tokenizer.json").unlink()
    else:
        (Path(config.tokenizer) / "tokenizer.json").write_text(content)
    with pytest.raises(ValueError, match=r"source_directory.tokenizer .*embedding-tokenizer"):
        load_directory_sources(config)


def test_uncapped_preparation_does_not_load_embedding_tokenizer(tmp_path, monkeypatch):
    config = text_config(tmp_path, "Uncapped sentence.")
    monkeypatch.setattr("data_designer_retrieval_sdg.multimodal.text_preparation.AutoTokenizer", UnexpectedInference)
    sources, _ = load_directory_sources(SourceDirectoryConfig(path=config.path, max_tokens_per_chunk=None))
    assert [source.text for source in sources] == ["Uncapped sentence."]


def test_oversized_sentence_rejected_before_inference_and_run_creation(tmp_path, monkeypatch):
    config = directory_config(tmp_path)
    config.source_directory = SourceDirectoryConfig(
        path=config.source_directory.path,
        max_tokens_per_chunk=3,
        tokenizer=save_embedding_tokenizer(tmp_path),
    )
    monkeypatch.setattr("data_designer_retrieval_sdg.multimodal.workflow.DataDesignerInference", UnexpectedInference)
    with pytest.raises(ValueError, match="split_oversized_sentences"):
        run_multimodal_sdg(config)
    assert not config.output_dir.exists()


@pytest.mark.parametrize(
    "change", ["tokenizer_backend", "tokenizer_config", "cap", "sentences", "reserve", "split", "versions"]
)
def test_resume_binds_tokenizer_snapshot_and_budget_even_if_units_unchanged(tmp_path, monkeypatch, change):
    config = directory_config(tmp_path)
    config.source_directory = SourceDirectoryConfig(
        path=config.source_directory.path,
        max_tokens_per_chunk=100,
        tokenizer=save_embedding_tokenizer(tmp_path),
    )
    monkeypatch.setattr("data_designer_retrieval_sdg.multimodal.workflow.DataDesignerInference", ScriptedInference)
    handoff = run_multimodal_sdg(config)
    manifest = json.loads((config.output_dir / "inputs/source_manifest.json").read_text())
    identity = json.loads((config.output_dir / "identity.json").read_text())
    assert manifest["source_preparation"] == identity["source_preparation"]
    assert re.fullmatch(r"[0-9a-f]{64}", manifest["source_preparation"]["tokenizer_sha256"])
    assert manifest["source_preparation"]["tokenizers_version"] == version("tokenizers")
    assert manifest["source_preparation"]["transformers_version"] == version("transformers")
    original, _ = load_directory_sources(config.source_directory)
    monkeypatch.setattr("data_designer_retrieval_sdg.multimodal.workflow.DataDesignerInference", UnexpectedInference)
    assert run_multimodal_sdg(config.model_copy(update={"resume": True})) == handoff
    if change == "tokenizer_backend":
        path = Path(config.source_directory.tokenizer) / "tokenizer.json"
        encoder = Tokenizer.from_file(str(path))
        encoder.add_tokens(["unused_added_token"])
        encoder.save(str(path))
    elif change == "tokenizer_config":
        path = Path(config.source_directory.tokenizer) / "tokenizer_config.json"
        settings = json.loads(path.read_text())
        settings["model_max_length"] = 2048
        path.write_text(json.dumps(settings))
    elif change == "versions":
        monkeypatch.setattr("data_designer_retrieval_sdg.multimodal.text_preparation.version", changed_version)
    else:
        settings = {
            "cap": {"max_tokens_per_chunk": 99},
            "sentences": {"sentences_per_chunk": 1},
            "reserve": {"reserved_tokens": 1},
            "split": {"split_oversized_sentences": True},
        }[change]
        config.source_directory = config.source_directory.model_copy(update=settings)
    unchanged, _ = load_directory_sources(config.source_directory)
    assert unchanged == original
    with pytest.raises(ValueError, match="Artifact changed"):
        run_multimodal_sdg(config.model_copy(update={"resume": True}))


def changed_version(distribution: str) -> str:
    """Simulate tokenizer dependency drift.

    Args:
        distribution: Distribution queried by preparation metadata.

    Returns:
        A version different from the installed distribution.
    """
    return version(distribution) + ".changed"
