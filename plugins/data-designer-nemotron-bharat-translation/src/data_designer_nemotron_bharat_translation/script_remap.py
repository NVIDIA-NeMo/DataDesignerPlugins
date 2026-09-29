# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic post-translation script remapping.

A translation into ``target_language`` sometimes needs to be re-expressed
in a different script than the one the translation model produced it in --
e.g. ``bodhan-ai/indic-translate`` only translates Manipuri into Bengali
script, but some pipelines need the final output in Meetei Mayek. Since
that conversion is a deterministic graphemic mapping, not a translation, it
runs as a plain function over the finished text rather than another model
call (see ``bengali_meitei_mapping.py``).

:data:`SCRIPT_REMAPPERS` maps ``(source_script, target_script)`` ISO 15924
code pairs to the remapper for that pair. Currently only Bengali -> Meetei
Mayek (``"Beng"`` -> ``"Mtei"``) is supported.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from data_designer_nemotron_bharat_translation.bengali_meitei_mapping import transliterate

#: Maps (source_script, target_script) ISO 15924 code pairs to their deterministic remapper.
SCRIPT_REMAPPERS: dict[tuple[str, str], Callable[[str], str]] = {
    ("Beng", "Mtei"): transliterate,
}

#: Spans the translation prompt promises to preserve byte-for-byte (see
#: GENERIC_TRANSLATION_PROMPT's "protected spans" rule): fenced code blocks,
#: inline code, and URLs. A single capturing group so re.split() interleaves
#: the protected spans themselves (odd indices) with the surrounding prose
#: (even indices) -- see remap_script().
_PROTECTED_SPAN_PATTERN = re.compile(r"(```.*?```|`[^`\n]*?`|https?://\S+)", re.DOTALL)


def can_remap_script(source_script: str, target_script: str) -> bool:
    """Whether a deterministic remapper exists for ``source_script`` -> ``target_script``."""
    return (source_script, target_script) in SCRIPT_REMAPPERS


def remap_script(text: str, source_script: str, target_script: str, *, preserve_protected_spans: bool = True) -> str:
    """Deterministically remaps ``text`` from ``source_script`` to ``target_script``.

    Args:
        text: Source-script text to remap.
        source_script: The script ``text`` is currently written in (ISO 15924, e.g. ``"Beng"``).
        target_script: The script to remap ``text`` into (ISO 15924, e.g. ``"Mtei"``).
        preserve_protected_spans: When ``True`` (the default), fenced code
            blocks, inline code, and URLs are left untouched -- only the
            surrounding prose is remapped. Character-level remappers like
            ``bengali_meitei_mapping.transliterate`` have no notion of code
            vs. prose, so without this, source-script characters that
            happen to appear inside a code literal or URL (e.g. a Bengali
            string literal in a code sample) would be silently remapped
            too, corrupting content the translation step promised to
            preserve byte-for-byte.

    Returns:
        ``text`` remapped into ``target_script``.

    Raises:
        ValueError: If no remapper exists for ``(source_script, target_script)``.
    """
    try:
        remapper = SCRIPT_REMAPPERS[(source_script, target_script)]
    except KeyError as exc:
        supported = ", ".join(f"{src!r} -> {dst!r}" for src, dst in sorted(SCRIPT_REMAPPERS))
        raise ValueError(
            f"No script remapper for {source_script!r} -> {target_script!r}. Supported: {supported}."
        ) from exc
    if not preserve_protected_spans:
        return remapper(text)
    parts = _PROTECTED_SPAN_PATTERN.split(text)
    return "".join(part if i % 2 else remapper(part) for i, part in enumerate(parts))
