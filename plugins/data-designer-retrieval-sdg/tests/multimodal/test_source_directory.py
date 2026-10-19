# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Directory input preserves canonical source, export, and resume contracts."""

import json
import re
import shutil
from pathlib import Path

import pytest
from data_designer.engine.secret_resolver import PlaintextResolver
from PIL import Image
from pydantic import ValidationError
from test_multimodal_sdg import ScriptedInference, fixture_config
from transformers import PreTrainedTokenizerFast

from data_designer_retrieval_sdg.multimodal import MultimodalSDGConfig, SourceDirectoryConfig, run_multimodal_sdg
from data_designer_retrieval_sdg.multimodal.source_directory import load_directory_sources
from data_designer_retrieval_sdg.multimodal.source_file import load_retrieval_sources
from data_designer_retrieval_sdg.text_qa.seed_reader import DocumentChunkerSeedReader
from data_designer_retrieval_sdg.text_qa.seed_source import DocumentChunkerSeedSource


def directory_config(tmp_path: Path) -> MultimodalSDGConfig:
    """Configure a small mixed collection with no explicit context manifest."""
    config = fixture_config(tmp_path)
    directory = tmp_path / "documents"
    directory.mkdir()
    (directory / "manual.txt").write_text("Valve A opens at 20 bar.", encoding="utf-8")
    (directory / "other.md").write_text("Valve B opens at 30 bar.", encoding="utf-8")
    Image.new("RGB", (24, 24), "white").save(directory / "figure.png")
    return MultimodalSDGConfig.model_validate(
        config.model_dump()
        | {
            "sources_file": None,
            "contexts_file": None,
            "source_directory": {"path": directory, "max_tokens_per_chunk": None},
        }
    )


class UnexpectedInference:
    """Fail when validation or a completed resume incorrectly reaches inference."""

    def __init__(self, *args, **kwargs):
        pytest.fail("Inference must not be constructed")


@pytest.mark.parametrize("both", [False, True])
def test_exactly_one_input_is_required(tmp_path, both):
    config = fixture_config(tmp_path).model_dump()
    if both:
        config["source_directory"] = {"path": tmp_path}
    else:
        config.pop("sources_file")
    with pytest.raises(ValidationError, match="exactly one"):
        MultimodalSDGConfig.model_validate(config)


@pytest.mark.parametrize(
    "changes",
    [
        {"file_extensions": []},
        {"file_extensions": [".pdf"]},
        {"file_pattern": "sub/*.txt"},
        {"file_pattern": ""},
        {"sentences_per_chunk": 0},
        {"text_chunking": "characters"},
        {"text_chunk_chars": 4000},
        {"num_files": 0},
        {"unexpected": True},
    ],
)
def test_directory_settings_reject_invalid_options(tmp_path, changes):
    with pytest.raises(ValidationError):
        SourceDirectoryConfig(max_tokens_per_chunk=None, path=tmp_path, **changes)


def test_discovery_order_filtering_limits_and_relative_identity(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / ".hidden").mkdir()
    for relative in ("z.txt", "nested/a.txt", "a.TXT", ".secret.txt", ".hidden/file.txt"):
        (tmp_path / relative).write_text("Original source text.", encoding="utf-8")
    (tmp_path / "unsupported.pdf").write_bytes(b"Not a text file")
    config = SourceDirectoryConfig(
        max_tokens_per_chunk=None, path=tmp_path, file_extensions=[".TXT", ".txt"], num_files=2
    )
    sources, manifest = load_directory_sources(config)
    assert config.file_extensions == [".txt"]
    assert [record["path"] for record in manifest["files"]] == ["a.TXT", "nested/a.txt"]
    assert [source.document_id for source in sources] == ["a.TXT", "nested/a.txt"]
    assert [source.unit_id for source in sources] == ["a.TXT#chunk-000001", "nested/a.txt#chunk-000001"]
    flat, _ = load_directory_sources(config.model_copy(update={"recursive": False, "file_pattern": "z*"}))
    assert [source.document_id for source in flat] == ["z.txt"]


def test_ids_and_input_hashes_survive_root_relocation(tmp_path):
    config = directory_config(tmp_path)
    (config.source_directory.path / "manual#chunk-000001.txt").write_text("Separate document.")
    original, manifest = load_directory_sources(config.source_directory)
    relocated = tmp_path / "moved"
    shutil.copytree(config.source_directory.path, relocated)
    moved, moved_manifest = load_directory_sources(config.source_directory.model_copy(update={"path": relocated}))
    assert manifest == moved_manifest
    assert [s.model_dump(exclude={"images"}) for s in original] == [s.model_dump(exclude={"images"}) for s in moved]
    assert len({source.unit_id for source in original}) == len(original)
    assert any("%23" in source.document_id for source in original)


@pytest.mark.parametrize("extension", ["txt", "md"])
@pytest.mark.parametrize("sentences_per_chunk", [1, 3, 5])
def test_directory_passages_match_text_qa_reader(tmp_path, extension, sentences_per_chunk):
    text = (
        "Dr. Smith checked the valve. It opens at 20 bar. Is that safe?\n\n"
        "Yes, within the rated range! The gauge reads 19.5 bar. Reset it after use."
    )
    path = tmp_path / f"manual.{extension}"
    path.write_text(text, encoding="utf-8")
    sources, _ = load_directory_sources(
        SourceDirectoryConfig(max_tokens_per_chunk=None, path=tmp_path, sentences_per_chunk=sentences_per_chunk)
    )
    reader = DocumentChunkerSeedReader()
    reader.attach(
        DocumentChunkerSeedSource(path=str(tmp_path), sentences_per_chunk=sentences_per_chunk),
        PlaintextResolver(),
    )
    batches = reader.create_batch_reader(batch_size=1, index_range=None, shuffle=False)
    row = batches.read_next_batch().to_pandas().iloc[0]
    assert [source.text for source in sources] == [chunk["text"] for chunk in row["chunks"]]
    assert {source.document_id for source in sources} == {row["source_id"]}
    assert [source.unit_id for source in sources] == [
        f"{row['source_id']}#chunk-{chunk['chunk_id']:06d}" for chunk in row["chunks"]
    ]


def test_explicit_sentence_chunks_cross_paragraph_boundaries_without_overlap(tmp_path):
    (tmp_path / "manual.txt").write_text(
        "First sentence. Second sentence?\n\nThird sentence! Fourth sentence. Fifth sentence? Sixth sentence"
    )
    sources, _ = load_directory_sources(
        SourceDirectoryConfig(max_tokens_per_chunk=None, path=tmp_path, sentences_per_chunk=5)
    )
    assert [source.text for source in sources] == [
        "First sentence. Second sentence? Third sentence! Fourth sentence. Fifth sentence?",
        "Sixth sentence",
    ]


def test_sentence_chunks_do_not_silently_fall_back_to_character_splitting(tmp_path):
    text = "A long sentence " + "with evidence " * 400 + "ends here."
    (tmp_path / "manual.txt").write_text(text)
    sources, _ = load_directory_sources(
        SourceDirectoryConfig(max_tokens_per_chunk=None, path=tmp_path, sentences_per_chunk=1)
    )
    assert len(sources) == 1
    assert sources[0].text == text
    assert len(sources[0].text) > 4000


def unexpected_sentence_chunking(text: str, sentences_per_chunk: int) -> None:
    """Reject sentence tokenization when whole-file preparation was selected.

    Args:
        text: Source text that must bypass sentence tokenization.
        sentences_per_chunk: Sentence limit that must not be applied.
    """
    pytest.fail("Whole-file preparation must not invoke sentence tokenization")


def test_no_chunking_preserves_internal_formatting_and_does_not_tokenize(tmp_path, monkeypatch):
    text = "  Heading\r\n\r\nColumn A\tColumn B\r\n" + "Long prepared evidence. " * 400 + "\n"
    (tmp_path / "prepared.md").write_bytes(text.encode("utf-8"))
    monkeypatch.setattr(
        "data_designer_retrieval_sdg.multimodal.text_preparation.text_to_sentence_chunks",
        unexpected_sentence_chunking,
    )
    sources, manifest = load_directory_sources(
        SourceDirectoryConfig(max_tokens_per_chunk=None, path=tmp_path, text_chunking="none")
    )
    assert len(sources) == 1
    assert sources[0].text == text.strip()
    assert sources[0].unit_id == "prepared.md#chunk-000001"
    assert manifest["files"][0]["unit_ids"] == [sources[0].unit_id]


def test_text_units_share_document_and_preserve_language(tmp_path):
    text = "First paragraph has evidence.\n\nSecond paragraph has more evidence."
    (tmp_path / "manual.md").write_bytes(b"\xef\xbb\xbf" + text.encode())
    sources, manifest = load_directory_sources(
        SourceDirectoryConfig(max_tokens_per_chunk=None, path=tmp_path, sentences_per_chunk=1, language="en")
    )
    assert len(sources) > 1
    assert {source.document_id for source in sources} == {"manual.md"}
    assert {source.language for source in sources} == {"en"}
    assert all(source.page_number is None and source.source_uri == "manual.md" for source in sources)
    assert re.sub(r"\s", "", "".join(source.text for source in sources)) == re.sub(r"\s", "", text)
    assert manifest["files"][0]["unit_ids"] == [source.unit_id for source in sources]


@pytest.mark.parametrize("extension", ["png", "jpg", "jpeg", "webp"])
def test_static_images_are_independent_units_without_inferred_pairing(tmp_path, extension):
    path = tmp_path / f"page-16.{extension}"
    Image.new("RGB", (24, 24), "white").save(path)
    (tmp_path / "page-16.txt").write_text("A separately retrievable passage.")
    sources, _ = load_directory_sources(SourceDirectoryConfig(max_tokens_per_chunk=None, path=tmp_path))
    image = next(source for source in sources if source.images)
    text = next(source for source in sources if source.text)
    assert image.images == [str(path)] and image.text == ""
    assert image.document_id != text.document_id
    assert image.page_number is None


@pytest.mark.parametrize(
    ("filename", "content", "message"),
    [
        ("empty.txt", b" \n", "nonwhitespace"),
        ("bad.txt", b"\xff", "UTF-8"),
        ("bad.png", b"broken", "Invalid source image"),
    ],
)
def test_invalid_selected_files_fail_with_path(tmp_path, filename, content, message):
    (tmp_path / filename).write_bytes(content)
    with pytest.raises(ValueError, match=message) as error:
        load_directory_sources(SourceDirectoryConfig(max_tokens_per_chunk=None, path=tmp_path))
    assert filename in str(error.value)


def test_animated_images_are_rejected(tmp_path):
    first = Image.new("RGB", (24, 24), "white")
    second = Image.new("RGB", (24, 24), "black")
    first.save(tmp_path / "animated.png", save_all=True, append_images=[second], duration=100)
    with pytest.raises(ValueError, match="static"):
        load_directory_sources(SourceDirectoryConfig(max_tokens_per_chunk=None, path=tmp_path))


def test_missing_and_empty_directories_fail(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        load_directory_sources(SourceDirectoryConfig(max_tokens_per_chunk=None, path=tmp_path / "missing"))
    with pytest.raises(ValueError, match="No supported files"):
        load_directory_sources(SourceDirectoryConfig(max_tokens_per_chunk=None, path=tmp_path))


def test_symlink_files_are_rejected_and_directory_links_are_not_followed(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "a.txt").write_text("Evidence.")
    (tmp_path / "loop").symlink_to(tmp_path, target_is_directory=True)
    sources, _ = load_directory_sources(SourceDirectoryConfig(max_tokens_per_chunk=None, path=tmp_path))
    assert len(sources) == 1
    (tmp_path / "alias.txt").symlink_to(real / "a.txt")
    with pytest.raises(ValueError, match="symbolic link"):
        load_directory_sources(SourceDirectoryConfig(max_tokens_per_chunk=None, path=tmp_path))


def test_output_cannot_be_discovered_as_input(tmp_path, monkeypatch):
    config = directory_config(tmp_path)
    config.output_dir = config.source_directory.path / "run"
    monkeypatch.setattr("data_designer_retrieval_sdg.multimodal.workflow.DataDesignerInference", UnexpectedInference)
    with pytest.raises(ValueError, match="outside source_directory"):
        run_multimodal_sdg(config)
    assert not config.output_dir.exists()


def test_invalid_directory_input_fails_before_run_initialization(tmp_path, monkeypatch):
    config = directory_config(tmp_path)
    (config.source_directory.path / "manual.txt").write_bytes(b"\xff")
    monkeypatch.setattr("data_designer_retrieval_sdg.multimodal.workflow.DataDesignerInference", UnexpectedInference)
    with pytest.raises(ValueError, match="UTF-8"):
        run_multimodal_sdg(config)
    assert not config.output_dir.exists()


@pytest.mark.parametrize("with_images", [False, True])
def test_prepared_sources_round_trip_and_completed_resume(tmp_path, monkeypatch, with_images):
    config = directory_config(tmp_path)
    if not with_images:
        (config.source_directory.path / "figure.png").unlink()
    monkeypatch.setattr("data_designer_retrieval_sdg.multimodal.workflow.DataDesignerInference", ScriptedInference)
    handoff = run_multimodal_sdg(config)
    prepared = config.output_dir / "inputs/sources.jsonl"
    sources = load_retrieval_sources(prepared)
    assert len(sources) == (3 if with_images else 2)
    assert (config.output_dir / "inputs/assets").exists() == with_images
    assert all(Path(image).is_relative_to(config.output_dir / "inputs/assets") for s in sources for image in s.images)
    manifest = json.loads((config.output_dir / "inputs/source_manifest.json").read_text())
    identity = json.loads((config.output_dir / "identity.json").read_text())
    assert manifest["files"] == identity["source_files"]
    assert [row["path"] for row in manifest["files"]] == (
        ["figure.png", "manual.txt", "other.md"] if with_images else ["manual.txt", "other.md"]
    )
    exported = json.loads((handoff.parent / "run_manifest.json").read_text())
    assert "source_directory" not in exported["policy"]["generation_config"]
    monkeypatch.setattr("data_designer_retrieval_sdg.multimodal.workflow.DataDesignerInference", UnexpectedInference)
    assert run_multimodal_sdg(config.model_copy(update={"resume": True})) == handoff


@pytest.mark.parametrize("change", ["add", "delete", "rename", "text", "image", "settings", "chunking_mode"])
def test_directory_input_drift_rejects_resume_before_inference(tmp_path, monkeypatch, change):
    config = directory_config(tmp_path)
    monkeypatch.setattr("data_designer_retrieval_sdg.multimodal.workflow.DataDesignerInference", ScriptedInference)
    run_multimodal_sdg(config)
    directory = config.source_directory.path
    if change == "add":
        (directory / "new.txt").write_text("New source.")
    elif change == "delete":
        (directory / "other.md").unlink()
    elif change == "rename":
        (directory / "other.md").rename(directory / "renamed.md")
    elif change == "text":
        # Canonical text stays identical after trimming; the raw hash must detect this.
        path = directory / "manual.txt"
        path.write_text(path.read_text() + "\n")
    elif change == "image":
        Image.new("RGB", (24, 24), "black").save(directory / "figure.png")
    elif change == "settings":
        config.source_directory = config.source_directory.model_copy(update={"sentences_per_chunk": 1})
    else:
        config.source_directory = config.source_directory.model_copy(update={"text_chunking": "none"})
    monkeypatch.setattr("data_designer_retrieval_sdg.multimodal.workflow.DataDesignerInference", UnexpectedInference)
    with pytest.raises(ValueError, match="Artifact changed"):
        run_multimodal_sdg(config.model_copy(update={"resume": True}))


@pytest.mark.parametrize(
    ("policy", "artifact"),
    [
        ("error", "inputs/sources.jsonl"),
        ("error", "inputs/source_manifest.json"),
        ("skip", "inputs/sources.jsonl"),
        ("skip", "inputs/source_manifest.json"),
        ("skip", "generation_failures.json"),
    ],
)
def test_completed_directory_artifacts_are_checked_on_resume(tmp_path, monkeypatch, policy, artifact):
    config = directory_config(tmp_path).model_copy(update={"failed_record_policy": policy})
    monkeypatch.setattr(ScriptedInference, "failures", {}, raising=False)
    monkeypatch.setattr("data_designer_retrieval_sdg.multimodal.workflow.DataDesignerInference", ScriptedInference)
    handoff = run_multimodal_sdg(config)
    monkeypatch.setattr("data_designer_retrieval_sdg.multimodal.workflow.DataDesignerInference", UnexpectedInference)
    assert run_multimodal_sdg(config.model_copy(update={"resume": True})) == handoff
    (config.output_dir / artifact).write_text("changed")
    with pytest.raises(ValueError, match="integrity"):
        run_multimodal_sdg(config.model_copy(update={"resume": True}))


@pytest.mark.parametrize("mode", ["sentences", "none", "tokens"])
def test_directory_and_equivalent_jsonl_produce_same_bundle(tmp_path, monkeypatch, mode):
    config = directory_config(tmp_path)
    original, _ = load_directory_sources(config.source_directory)
    tokenizer_dir = tmp_path / "embedding-tokenizer"
    if mode == "tokens":
        PreTrainedTokenizerFast(tokenizer_file=str(config.generator.tokenizer_file)).save_pretrained(tokenizer_dir)
    settings = (
        {
            "max_tokens_per_chunk": 16,
            "tokenizer": str(tokenizer_dir),
            "split_oversized_sentences": True,
        }
        if mode == "tokens"
        else {"text_chunking": mode}
    )
    config.source_directory = SourceDirectoryConfig.model_validate(config.source_directory.model_dump() | settings)
    sources, _ = load_directory_sources(config.source_directory)
    if mode == "tokens":
        assert len(sources) > len(original)
    assert [source for source in sources if source.images] == [source for source in original if source.images]
    source_file = tmp_path / "canonical.jsonl"
    source_file.write_text("".join(source.model_dump_json() + "\n" for source in sources))
    monkeypatch.setattr("data_designer_retrieval_sdg.multimodal.workflow.DataDesignerInference", ScriptedInference)
    directory_handoff = run_multimodal_sdg(config)
    jsonl_config = MultimodalSDGConfig.model_validate(
        config.model_dump()
        | {"source_directory": None, "sources_file": source_file, "output_dir": tmp_path / "jsonl-run"}
    )
    jsonl_handoff = run_multimodal_sdg(jsonl_config)
    for filename in ("retrieval_units.jsonl", "candidate_diagnostics.jsonl", "report.json", "split_manifest.json"):
        assert (directory_handoff.parent / filename).read_bytes() == (jsonl_handoff.parent / filename).read_bytes()
