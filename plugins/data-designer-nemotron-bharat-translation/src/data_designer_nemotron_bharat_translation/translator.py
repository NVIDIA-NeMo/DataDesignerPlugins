# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared single-prompt translation logic for ``document-translation``, with optional windowing.

Every plugin translates a span of text with one combined prompt -- ``text``,
``source_language_description``, and ``target_language_description`` all in
a single user turn, no system message, no per-row column access -- via
:class:`Translator`, rather than maintaining its own copy of the
render-parse logic. This is also why translation defaults to the same
:data:`GENERIC_TRANSLATION_PROMPT`/:data:`BODHAN_TRANSLATION_PROMPT` pair
(see ``prompts.py``), and takes BCP-47 ``source_language``/``target_language``
codes resolved to natural-language descriptions via
``DocumentTranslationConfig.language_descriptions``.

``translate``/``atranslate`` always return ``(translation, window_traces)``.
When ``Translator`` is constructed with ``enable_windowing=True``, the
input is transparently split into overlapping token-bounded windows,
each window is translated independently (carrying its neighbours for
context), then each window's own non-overlapping segment is extracted and
the segments are stacked back into one document -- ``translation`` is that
merged document, and (when ``keep_window_traces`` is also set) the window
traces are a list of one dict per window: ``{"window": ...,
"translated_window": ..., "extracted_window": ...,
"extracted_translated_window": ...}`` (``"window"`` is the source text,
``"translated_window"`` is that window's raw translation before
extraction, ``"extracted_window"``/``"extracted_translated_window"`` are
its own non-overlapping source/translated segment after extraction --
both ``None`` for a window that owns no chunks and so contributes nothing
to the merged document). The window traces are ``None`` when
``enable_windowing`` is not set, or
when ``keep_window_traces`` is ``False``. This is how long documents that
degrade whole-document translation quality get translated without
changing the shape of what a caller (e.g. ``document-translation``'s
translate-evaluate-rewrite loop) does with the result.
"""

from __future__ import annotations

import asyncio
import functools
from collections.abc import Iterator

from data_designer.engine.models.recipes.response_recipes import TextResponseRecipe
from jinja2 import Template
from tokenizers import Tokenizer

from data_designer_nemotron_bharat_translation.prompts import MERGE_PROMPT


class Translator:
    """Renders a single combined translation prompt and parses the model's plain-text response.

    Whole-document translation degrades badly past a few thousand tokens for
    sentence-level translation models (they omit, splice, or loop), while
    per-chunk translation with no context loses pronoun, gender, and
    register agreement across chunk boundaries. When ``enable_windowing`` is
    set, this sits between the two: the document is split into overlapping
    token-bounded windows (``window_tokens``/``window_overlap``/
    ``window_separator``, measured with ``window_tokenizer_model``), each
    window is translated inside its neighbours' context, then each window's
    own non-overlapping segment is extracted independently -- by
    ``merge_model``, rendering ``merge_prompt`` -- and the segments are
    stacked. ``merge_model`` is required when ``enable_windowing`` is set,
    since ``Translator`` itself has no access to a model registry to resolve
    it from an alias. ``keep_window_traces`` (default ``True``) controls
    whether the per-window trace data is kept and returned -- set it to
    ``False`` to drop it (e.g. to avoid the extra memory/output size) once
    it isn't needed.
    """

    def __init__(
        self,
        template: str,
        enable_windowing: bool = False,
        window_separator: str = "\n",
        window_tokenizer_model: str = "google/gemma-4-31b-it",
        window_tokens: int = 2048,
        window_overlap: int = 1,
        window_max_concurrent: int = 8,
        keep_window_traces: bool = True,
        merge_model=None,
        merge_prompt: str = MERGE_PROMPT,
    ) -> None:
        self._template = Template(template)
        self._recipe = TextResponseRecipe()
        self._enable_windowing = enable_windowing
        if enable_windowing:
            if merge_model is None:
                raise ValueError("merge_model is required when enable_windowing is True.")
            self._merge_model = merge_model
            self._merge_template = Template(merge_prompt)
            self._window_separator = window_separator
            self._window_tokenizer_model = window_tokenizer_model
            self._window_tokens = window_tokens
            self._window_overlap = window_overlap
            self._window_max_concurrent = window_max_concurrent
            self._keep_window_traces = keep_window_traces

    @functools.cached_property
    def _tokenizer(self) -> Tokenizer:
        return Tokenizer.from_pretrained(self._window_tokenizer_model)

    def render(self, text: str, source_language_description: str, target_language_description: str) -> str:
        user_prompt = self._template.render(
            text=text,
            source_language_description=source_language_description,
            target_language_description=target_language_description,
        )
        return self._recipe.apply_recipe_to_user_prompt(user_prompt)

    def translate(
        self,
        model,
        text: str,
        source_language_description: str,
        target_language_description: str,
        max_correction_steps: int,
        record: dict | None = None,
    ) -> tuple[str, list[dict[str, str | None]] | None]:
        """Translates ``text``. Returns ``(translation, window_traces)``.

        ``window_traces`` is a list of one dict per window -- ``"window"``,
        ``"translated_window"``, ``"extracted_window"``,
        ``"extracted_translated_window"`` (see the module docstring) -- or
        ``None`` when ``enable_windowing`` is not set, or when
        ``keep_window_traces`` is ``False``.
        """
        if not self._enable_windowing:
            prompt = self.render(text, source_language_description, target_language_description)
            result, _trace = model.generate(
                prompt, parser=self._recipe.parse, max_correction_steps=max_correction_steps
            )
            return result, None
        return self._translate_windowed(
            model, text, source_language_description, target_language_description, max_correction_steps, record or {}
        )

    async def atranslate(
        self,
        model,
        text: str,
        source_language_description: str,
        target_language_description: str,
        max_correction_steps: int,
        record: dict | None = None,
    ) -> tuple[str, list[dict[str, str | None]] | None]:
        """Async version of :meth:`translate`. See its docstring for the return value."""
        if not self._enable_windowing:
            prompt = self.render(text, source_language_description, target_language_description)
            result, _trace = await model.agenerate(
                prompt, parser=self._recipe.parse, max_correction_steps=max_correction_steps
            )
            return result, None
        return await self._atranslate_windowed(
            model, text, source_language_description, target_language_description, max_correction_steps, record or {}
        )

    # ---------------------------------------------------------------- windowing

    @staticmethod
    def build_spans(chunk_tokens: list[int], window_tokens: int, overlap: int) -> list[tuple[int, int]]:
        """Computes ``[start, end)`` chunk indices of each window, sized by a token budget.

        Each window grows chunk by chunk while the running total stays
        within ``window_tokens``, then the next window starts ``overlap``
        chunks before this one ended, so consecutive windows share that
        many chunks. Two invariants:

        - At least 2 chunks per window, even when that busts the budget --
          a one-chunk window carries no neighbouring context, which is the
          whole point of windowing.
        - Progress: the next window always starts at least one chunk
          further on, so a large ``overlap`` cannot stall the walk.

        Args:
            chunk_tokens: Token count of each chunk, in document order.
            window_tokens: Max tokens per window.
            overlap: Chunks shared between consecutive windows.

        Returns:
            A list of ``(start, end)`` chunk-index spans.
        """
        n = len(chunk_tokens)
        spans: list[tuple[int, int]] = []
        start = 0
        while start < n:
            end, total = start, 0
            while end < n and (end - start < 2 or total + chunk_tokens[end] <= window_tokens):
                total += chunk_tokens[end]
                end += 1
            end = max(end, min(start + 2, n))
            if end - start < 2 and n >= 2:
                start = end - 2  # last window ran out of document -- extend it backwards instead
            spans.append((start, end))
            if end >= n:
                break
            start = max(start + 1, end - overlap)
        return spans

    def _chunk_tokens(self, chunks: list[str]) -> list[int]:
        if not chunks:
            return []
        encoded = self._tokenizer.encode_batch(chunks, add_special_tokens=False)
        return [len(e.ids) for e in encoded]

    def _split(self, text: str) -> tuple[list[str], list[tuple[int, int]], list[str]]:
        chunks = (text or "").split(self._window_separator)
        spans = self.build_spans(self._chunk_tokens(chunks), self._window_tokens, self._window_overlap)
        windows = [self._window_separator.join(chunks[lo:hi]) for lo, hi in spans]
        return chunks, spans, windows

    def _translate_window(
        self,
        model,
        window_text: str,
        source_language_description: str,
        target_language_description: str,
        max_correction_steps: int,
        index: int,
        count: int,
    ) -> str:
        """Translates one window. Blank windows return ""; a model failure raises."""
        if not window_text.strip():
            return ""
        prompt = self.render(window_text, source_language_description, target_language_description)
        result, _trace = model.generate(prompt, parser=self._recipe.parse, max_correction_steps=max_correction_steps)
        return self._require(result, f"window {index + 1}/{count}")

    async def _atranslate_window(
        self,
        model,
        window_text: str,
        source_language_description: str,
        target_language_description: str,
        max_correction_steps: int,
        index: int,
        count: int,
    ) -> str:
        if not window_text.strip():
            return ""
        prompt = self.render(window_text, source_language_description, target_language_description)
        result, _trace = await model.agenerate(
            prompt, parser=self._recipe.parse, max_correction_steps=max_correction_steps
        )
        return self._require(result, f"window {index + 1}/{count}")

    async def _atranslate_window_limited(
        self,
        semaphore: asyncio.Semaphore,
        model,
        window_text: str,
        source_language_description: str,
        target_language_description: str,
        max_correction_steps: int,
        index: int,
        count: int,
    ) -> str:
        async with semaphore:
            return await self._atranslate_window(
                model,
                window_text,
                source_language_description,
                target_language_description,
                max_correction_steps,
                index,
                count,
            )

    def _translate_windowed(
        self,
        model,
        text: str,
        source_language_description: str,
        target_language_description: str,
        max_correction_steps: int,
        record: dict,
    ) -> tuple[str, list[dict[str, str | None]] | None]:
        """Sync path: windows are translated one at a time. Prefer the async path."""
        chunks, spans, windows = self._split(text)
        translations = [
            self._translate_window(
                model,
                w,
                source_language_description,
                target_language_description,
                max_correction_steps,
                i,
                len(windows),
            )
            for i, w in enumerate(windows)
        ]
        merged, _steps, extracted_windows, extracted_translations = self._merge_all(
            chunks, spans, translations, record, source_language_description, target_language_description
        )
        if not self._keep_window_traces:
            return merged, None
        return merged, self._zip_window_traces(windows, translations, extracted_windows, extracted_translations)

    async def _atranslate_windowed(
        self,
        model,
        text: str,
        source_language_description: str,
        target_language_description: str,
        max_correction_steps: int,
        record: dict,
    ) -> tuple[str, list[dict[str, str | None]] | None]:
        """Async path: a document's windows are translated concurrently, then their segments extracted concurrently."""
        chunks, spans, windows = self._split(text)
        semaphore = asyncio.Semaphore(self._window_max_concurrent)
        translations = list(
            await asyncio.gather(
                *(
                    self._atranslate_window_limited(
                        semaphore,
                        model,
                        w,
                        source_language_description,
                        target_language_description,
                        max_correction_steps,
                        i,
                        len(windows),
                    )
                    for i, w in enumerate(windows)
                )
            )
        )
        merged, _steps, extracted_windows, extracted_translations = await self._amerge_all(
            chunks, spans, translations, record, source_language_description, target_language_description
        )
        if not self._keep_window_traces:
            return merged, None
        return merged, self._zip_window_traces(windows, translations, extracted_windows, extracted_translations)

    @staticmethod
    def _zip_window_traces(
        windows: list[str],
        translations: list[str],
        extracted_windows: list[str | None],
        extracted_translations: list[str | None],
    ) -> list[dict[str, str | None]]:
        return [
            {"window": w, "translated_window": t, "extracted_window": ew, "extracted_translated_window": et}
            for w, t, ew, et in zip(windows, translations, extracted_windows, extracted_translations, strict=True)
        ]

    # ---------------------------------------------------------------- merge (local segment extraction)

    def _owned_range(self, spans: list[tuple[int, int]], i: int) -> tuple[int, int]:
        """The ``[start, end)`` chunk range window ``i`` owns.

        The first window keeps its leading chunk (no previous window owns
        it); every other window starts where the previous window ended --
        so the ranges tile the document with no gaps or overlaps -- and
        each window keeps its trailing (overlap) chunk.
        """
        lo = spans[i][0] if i == 0 else spans[i - 1][1]
        return lo, spans[i][1]

    def _merge_calls(
        self, chunks: list[str], spans: list[tuple[int, int]], translations: list[str]
    ) -> Iterator[tuple[int, str, str, str]]:
        """Yields ``(i, source_text, windows_block, segment_source)`` for each window that owns >= 1 chunk."""
        sep = self._window_separator
        n = len(spans)
        for i in range(n):
            lo, hi = self._owned_range(spans, i)
            if hi <= lo:
                continue
            shown = [k for k in (i - 1, i, i + 1) if 0 <= k < n]
            src_lo, src_hi = spans[shown[0]][0], spans[shown[-1]][1]
            block = "\n\n".join(
                f"## Window {pos}{'  <- the portion to return is within this window' if k == i else ''}\n"
                f"{translations[k]}"
                for pos, k in enumerate(shown, 1)
            )
            yield i, sep.join(chunks[src_lo:src_hi]), block, sep.join(chunks[lo:hi])

    def _merge_prompt(
        self,
        source_text: str,
        windows_block: str,
        segment_source: str,
        record: dict,
        source_language_description: str,
        target_language_description: str,
    ) -> str:
        context = {
            **record,
            "source_text": source_text,
            "windows_block": windows_block,
            "segment_source": segment_source,
            "source_language": source_language_description,
            "target_language": target_language_description,
        }
        return self._merge_template.render(**context)

    def _merge_all(
        self,
        chunks,
        spans,
        translations,
        record,
        source_language_description: str,
        target_language_description: str,
    ) -> tuple[str, int, list[str | None], list[str | None]]:
        if not translations:
            return "", 0, [], []
        if len(translations) == 1 and not translations[0].strip():
            # _translate_window returns "" for a whitespace-only window, so a blank
            # single-window document arrives here as [""]. Nothing to extract or correct.
            return "", 0, [None], [None]
        return self._merge(
            chunks, spans, translations, record, source_language_description, target_language_description
        )

    async def _amerge_all(
        self,
        chunks,
        spans,
        translations,
        record,
        source_language_description: str,
        target_language_description: str,
    ) -> tuple[str, int, list[str | None], list[str | None]]:
        if not translations:
            return "", 0, [], []
        if len(translations) == 1 and not translations[0].strip():
            return "", 0, [None], [None]
        return await self._amerge(
            chunks, spans, translations, record, source_language_description, target_language_description
        )

    def _merge(
        self, chunks, spans, translations, record, source_language_description: str, target_language_description: str
    ) -> tuple[str, int, list[str | None], list[str | None]]:
        """Sync: extracts each window's owned segment, then stacks them in document order."""
        extracted_windows: list[str | None] = [None] * len(spans)
        extracted_translations: list[str | None] = [None] * len(spans)
        segments = []
        for i, source_text, block, segment_source in self._merge_calls(chunks, spans, translations):
            out, _trace = self._merge_model.generate(
                self._merge_prompt(
                    source_text, block, segment_source, record, source_language_description, target_language_description
                )
            )
            segment = self._require(out, f"segment {i + 1}")
            extracted_windows[i] = segment_source
            extracted_translations[i] = segment
            segments.append(segment)
        return self._window_separator.join(segments), len(segments), extracted_windows, extracted_translations

    async def _amerge(
        self, chunks, spans, translations, record, source_language_description: str, target_language_description: str
    ) -> tuple[str, int, list[str | None], list[str | None]]:
        """Async: the per-window segment extractions are independent, so they run concurrently.

        ``asyncio.gather`` preserves order, so the stack stays in document order.
        """
        calls = list(self._merge_calls(chunks, spans, translations))
        semaphore = asyncio.Semaphore(self._window_max_concurrent)
        segments = list(
            await asyncio.gather(
                *(
                    self._aextract_segment(
                        semaphore,
                        i,
                        source_text,
                        block,
                        segment_source,
                        record,
                        source_language_description,
                        target_language_description,
                    )
                    for i, source_text, block, segment_source in calls
                )
            )
        )
        extracted_windows: list[str | None] = [None] * len(spans)
        extracted_translations: list[str | None] = [None] * len(spans)
        for (i, _source_text, _block, segment_source), segment in zip(calls, segments, strict=True):
            extracted_windows[i] = segment_source
            extracted_translations[i] = segment
        return self._window_separator.join(segments), len(segments), extracted_windows, extracted_translations

    async def _aextract_segment(
        self,
        semaphore: asyncio.Semaphore,
        i: int,
        source_text: str,
        block: str,
        segment_source: str,
        record: dict,
        source_language_description: str,
        target_language_description: str,
    ) -> str:
        async with semaphore:
            out, _trace = await self._merge_model.agenerate(
                self._merge_prompt(
                    source_text, block, segment_source, record, source_language_description, target_language_description
                )
            )
            return self._require(out, f"segment {i + 1}")

    def _require(self, text: object, what: str) -> str:
        """Model output must be a non-empty string. Anything else fails the row loudly.

        Strips leading/trailing repeats of the literal ``window_separator``
        string (``"\\n"`` by default) to clean up stray blank-line padding
        around the model's response, before ``_merge``/``_amerge`` join
        segments back into a document.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{what} returned no usable text ({text!r})")
        return self._strip_window_separator(text)

    def _strip_window_separator(self, text: str) -> str:
        """Strips leading/trailing repeats of the literal ``window_separator`` string.

        Unlike ``str.strip(window_separator)``, which strips any character
        in ``window_separator`` (a set, not a substring match), this only
        removes exact occurrences of the separator itself.
        """
        sep = self._window_separator
        if not sep:
            return text
        while text.startswith(sep):
            text = text[len(sep) :]
        while text.endswith(sep):
            text = text[: -len(sep)]
        return text
