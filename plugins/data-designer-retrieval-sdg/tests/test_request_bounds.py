# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concrete rendering and deployment budgets, with invented offline model responses."""

from functools import partial

import pytest
from data_designer.engine.column_generators.utils.prompt_renderer import PromptType, RecordBasedPromptRenderer
from data_designer.engine.processing.ginja import environment
from test_multimodal_sdg import fixture_config

from data_designer_retrieval_sdg.multimodal.bounds import RequestTooLarge, check_request, fit_contexts, tokenizer
from data_designer_retrieval_sdg.multimodal.inference import DataDesignerInference, Request
from data_designer_retrieval_sdg.multimodal.models import ContextSummary, GenerationContext, QuerySlot
from data_designer_retrieval_sdg.multimodal.planning import bounded_contexts
from data_designer_retrieval_sdg.multimodal.storage import digest
from data_designer_retrieval_sdg.multimodal.summary_stages import summary_requests
from data_designer_retrieval_sdg.multimodal.workflow import (
    bounded_query_batches,
    generation_requests,
    judgment_requests,
)
from data_designer_retrieval_sdg.retrieval.models import RetrievalSource
from data_designer_retrieval_sdg.structured import RetrievalStructuredResponseRecipe


def test_dd_boundary_is_the_rendered_prompt_not_source_text(tmp_path):
    config = fixture_config(tmp_path)
    config.generator = config.generator.model_copy(update={"context_window_tokens": None})
    request = Request("x" * environment.MAX_RENDERED_LEN, ContextSummary, "generator")
    check_request(request, config)
    recipe = RetrievalStructuredResponseRecipe(json_schema=ContextSummary.model_json_schema(), pruning=False)
    renderer = RecordBasedPromptRenderer(recipe)
    rendered = renderer.render(
        prompt_template="{{ request_text }}", record={"request_text": request.text}, prompt_type=PromptType.USER_PROMPT
    )
    assert rendered == recipe.apply_recipe_to_user_prompt(request.text)
    assert len(rendered) > environment.MAX_RENDERED_LEN  # Schema is added after DD's rendering check.
    with pytest.raises(RequestTooLarge, match="DD limit"):
        check_request(Request(request.text + "x", ContextSummary, "generator"), config)


def test_large_fitting_context_keeps_all_evidence_and_identity(tmp_path):
    config = fixture_config(tmp_path)
    sources = [RetrievalSource(unit_id=str(i), document_id="doc", text="x" * 60000) for i in range(2)]
    original = [s.model_dump() for s in sources]
    context = GenerationContext(context_id="intact", unit_ids=["0", "1"])
    assert config.max_context_chars is None
    assert bounded_contexts([context], sources, config) == [context]
    builder = partial(generation_requests, sources={s.unit_id: s for s in sources}, config=config)
    assert fit_contexts([context], builder, config) == [context]
    assert [s.model_dump() for s in sources] == original


def test_instruction_and_enrichment_overhead_split_whole_units(tmp_path):
    config = fixture_config(tmp_path)
    sources = {str(i): RetrievalSource(unit_id=str(i), document_id="doc", text="x" * 100000) for i in range(4)}
    visual = {key: {"has_visual_content": True, "description": "v" * 30000} for key in sources}
    context = GenerationContext(context_id="long", unit_ids=list(sources))
    builder = partial(summary_requests, sources=sources, visual=visual, descriptions={("doc", "source"): "d" * 1000})
    children = fit_contexts([context], builder, config)
    assert len(children) == 2
    assert [key for child in children for key in child.unit_ids] == list(sources)
    for child in children:
        check_request(builder(child)[0], config)


def test_indivisible_unit_fails_without_truncating(tmp_path):
    config = fixture_config(tmp_path)
    sources = {"x": RetrievalSource(unit_id="x", document_id="doc", text="x" * environment.MAX_RENDERED_LEN)}
    context = GenerationContext(context_id="one", unit_ids=["x"])
    with pytest.raises(RequestTooLarge, match="prepare smaller canonical units"):
        fit_contexts([context], partial(generation_requests, sources=sources, config=config), config)
    assert len(sources["x"].text) == environment.MAX_RENDERED_LEN


def test_model_budget_counts_schema_utf8_images_overhead_and_output(tmp_path):
    config = fixture_config(tmp_path)
    model = config.generator
    request = Request("é😀{{ literal }}", ContextSummary, "generator", (tmp_path / "page.png",) * 2)
    recipe = RetrievalStructuredResponseRecipe(json_schema=ContextSummary.model_json_schema(), pruning=False)
    text = recipe.apply_recipe_to_user_prompt(request.text)
    count = len(
        tokenizer(str(model.tokenizer_file), digest(model.tokenizer_file)).encode(text, add_special_tokens=False).ids
    )
    required = count + 2 * model.image_tokens_per_image + model.request_overhead_tokens + model.max_tokens
    config.generator = model.model_copy(update={"context_window_tokens": required})
    check_request(request, config)
    config.generator = model.model_copy(update={"context_window_tokens": required - 1})
    with pytest.raises(RequestTooLarge, match="including images, overhead and output"):
        check_request(request, config)


@pytest.mark.parametrize("field", ["tokenizer_file", "request_overhead_tokens", "image_tokens_per_image"])
def test_explicit_model_budget_requires_consistent_metadata(tmp_path, field):
    config = fixture_config(tmp_path)
    config.generator = config.generator.model_copy(update={field: None})
    inference = DataDesignerInference(tmp_path / "unused", config)
    with pytest.raises(ValueError, match="Configure"):
        inference.generate([Request("short", ContextSummary, "generator", (tmp_path / "page.png",))])
    assert not inference.root.exists()


def test_judge_model_budget_is_independent(tmp_path):
    config = fixture_config(tmp_path)
    config.judge = config.judge.model_copy(update={"context_window_tokens": 9000})
    check_request(Request("x " * 1000, ContextSummary, "generator"), config)
    with pytest.raises(RequestTooLarge, match="judge request"):
        check_request(Request("x " * 1000, ContextSummary, "judge"), config)


class QueryInference:
    """Return small queries while preserving every attempted concrete request."""

    def __init__(self):
        self.requests = []

    def generate(self, requests):
        self.requests.extend(requests)
        return [
            request.schema(
                queries=[
                    QuerySlot(slot=i, query="Which specification applies?", evidence_modality="text")
                    for i in range(request.schema.model_fields["queries"].metadata[0].min_length)
                ]
            )
            for request in requests
        ]


def test_repeated_judge_evidence_replans_queries_on_children(tmp_path):
    config = fixture_config(tmp_path)
    sources = {str(i): RetrievalSource(unit_id=str(i), document_id="doc", text="x" * 140000) for i in range(2)}
    context = GenerationContext(context_id="parent", unit_ids=list(sources))
    row = {"context": context.model_dump(), "slots": {"queries": []}}
    outcomes = [row]
    check_request(generation_requests(context, sources, config)[0], config)
    slot = QuerySlot(slot=0, query="Which specification applies?", evidence_modality="text")
    with pytest.raises(RequestTooLarge):
        check_request(judgment_requests(context, slot, sources)[2], config)
    inference = QueryInference()
    selected, batches = bounded_query_batches([row], outcomes, sources, config, inference)
    assert row["selection_reason"] == "request_size_replanned"
    assert row["slots"]["queries"]  # Superseded response evidence is retained.
    assert [key for child in selected for key in child["context"]["unit_ids"]] == list(sources)
    assert len(selected) == len(batches) == 2
    assert len(inference.requests) == 3  # Original, then two regenerated child queries.
    for child, batch in zip(selected, batches, strict=True):
        context = GenerationContext.model_validate(child["context"])
        for slot in batch.queries:
            for request in judgment_requests(context, slot, sources):
                check_request(request, config)


def test_tokenizer_truncation_cannot_hide_overflow_and_drift_changes_cache(tmp_path):
    from tokenizers import Tokenizer

    config = fixture_config(tmp_path)
    model = config.generator
    request = Request("x " * 1000, ContextSummary, "generator")
    inference = DataDesignerInference(tmp_path / "unused", config)
    original_key = inference.request_key(request)
    encoder = Tokenizer.from_file(str(model.tokenizer_file))
    encoder.enable_truncation(max_length=1)
    encoder.save(str(model.tokenizer_file))
    assert inference.request_key(request) != original_key
    config.generator = model.model_copy(update={"context_window_tokens": 9000})
    with pytest.raises(RequestTooLarge):
        check_request(request, config)


def test_combination_rendering_splits_complete_summary_memberships(tmp_path):
    from data_designer_retrieval_sdg.multimodal.summary_stages import combination_requests

    config = fixture_config(tmp_path)
    rows = [{"summary": {"summary": "s" * 180000}} for _ in range(3)]
    context = GenerationContext(context_id="combined", unit_ids=["0", "1", "2"])
    builder = partial(combination_requests, rows=rows)
    children = fit_contexts([context], builder, config)
    assert [key for child in children for key in child.unit_ids] == context.unit_ids
    assert len(children) == 2
    for child in children:
        check_request(builder(child)[0], config)


def test_combined_summary_workflow_submits_the_checked_request(tmp_path, monkeypatch):
    from test_multimodal_sections import SectionInference

    from data_designer_retrieval_sdg.multimodal.summary_stages import plan_summaries
    from data_designer_retrieval_sdg.retrieval.source_file import load_retrieval_sources

    config = fixture_config(tmp_path)
    sources = load_retrieval_sources(config.sources_file)
    contexts = [GenerationContext(context_id=str(i), unit_ids=[sources[i].unit_id]) for i in range(2)]
    monkeypatch.setattr(
        "data_designer_retrieval_sdg.multimodal.summary_stages.semantic_combinations", lambda *args: [[0, 1]]
    )
    inference = SectionInference()
    rows = plan_summaries(sources, contexts, config, inference)
    assert len(rows) == 3
    assert rows[-1]["combined"]
    assert rows[-1]["context"]["unit_ids"] == [sources[0].unit_id, sources[1].unit_id]
    for request in inference.requests:
        check_request(request, config)


def test_incomplete_opt_in_budget_does_not_create_an_unresumable_run(tmp_path):
    from data_designer_retrieval_sdg.multimodal.workflow import run_multimodal_sdg

    config = fixture_config(tmp_path)
    config.generator = config.generator.model_copy(update={"tokenizer_file": None})
    with pytest.raises(ValueError, match="Configure"):
        run_multimodal_sdg(config)
    assert not config.output_dir.exists()


def without_model_budgets(config):
    from data_designer_retrieval_sdg.multimodal.models import ModelSettings

    model = ModelSettings(model="operator/vlm", endpoint="https://example.invalid/v1", credential_env="TEST_SDG_KEY")
    return config.model_copy(update={"generator": model, "judge": model})


def test_default_live_inference_needs_no_local_model_metadata(tmp_path):
    from test_multimodal_sdg import MissingInference

    config = without_model_budgets(fixture_config(tmp_path))
    inference = MissingInference(tmp_path / "inference", config)
    with pytest.raises(RuntimeError, match="Missing structured"):
        inference.generate([Request("short", ContextSummary, "generator", (tmp_path / "page.png",))])
    assert inference.calls == config.missing_response_attempts  # Reached the transport, with normal retry semantics.
    with pytest.raises(RequestTooLarge, match="DD limit"):
        inference.generate([Request("x" * (environment.MAX_RENDERED_LEN + 1), ContextSummary, "generator")])
    assert inference.calls == config.missing_response_attempts  # Oversized prompt never reaches transport.


def test_default_workflow_runs_without_tokenizer_or_allowances(tmp_path, monkeypatch):
    from test_multimodal_sdg import ScriptedInference

    from data_designer_retrieval_sdg.multimodal.workflow import run_multimodal_sdg

    config = without_model_budgets(fixture_config(tmp_path))
    monkeypatch.setattr("data_designer_retrieval_sdg.multimodal.workflow.DataDesignerInference", ScriptedInference)
    result = run_multimodal_sdg(config)
    assert result.is_file()
