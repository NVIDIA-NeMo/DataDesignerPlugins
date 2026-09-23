# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structure-aware sections with whole-unit identity and checked source coverage."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict

from data_designer_retrieval_sdg.multimodal import prompts
from data_designer_retrieval_sdg.multimodal.inference import Request
from data_designer_retrieval_sdg.multimodal.models import GenerationContext, MultimodalSDGConfig, TOCConfirmation
from data_designer_retrieval_sdg.multimodal.planning import automatic_contexts, context_chars
from data_designer_retrieval_sdg.multimodal.storage import fingerprint, write_json
from data_designer_retrieval_sdg.retrieval.models import RetrievalSource


def structural_lines(text: str) -> list[str]:
    """Return source lines outside fenced code and HTML comments; never modify sources."""
    lines, fence = [], None
    for line in re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL).splitlines():
        marker = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if marker:
            token = marker.group(1)
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
        elif fence is None:
            lines.append(line.rstrip())
    return lines


def normalized_title(text: str) -> str:
    """Normalize exact title matches across numbering, accents and markdown syntax."""
    text = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"^\s*[#|*\s]*\d+(?:\.\d+)*[.)]?\s+", "", text)
    text = "".join(c for c in unicodedata.normalize("NFKD", text.casefold()) if not unicodedata.combining(c))
    return " ".join(re.findall(r"[^\W_]+", text))


def heading_titles(text: str) -> list[str]:
    """Read ATX headings without cutting the canonical unit containing a heading."""
    return [
        normalized_title(match.group(1))
        for line in structural_lines(text)
        if (match := re.match(r"^ {0,3}#{1,5}\s+(.+?)(?:\s+#+)?$", line))
    ]


def toc_entries(text: str) -> list[dict]:
    """Read title/trailing-Arabic-page rows; preserve source order for validation.

    Unsupported Roman labels and ambiguous rows remain ordinary source content.
    A numbered list alone is not a TOC: candidate detection and cached confirmation
    are separate gates, followed by title/offset validation within the document.
    """
    entries = []
    for line in structural_lines(text):
        line = re.sub(r"[|…]", " ", line)
        line = re.sub(r"\.{2,}", " ", line).strip()
        match = re.match(r"^(.+?)\s+(-?\d+)\s*$", line)
        if match and len(match.group(2)) <= 9 and any(char.isalpha() for char in match.group(1)):
            entries.append({"title": match.group(1).strip(), "page": int(match.group(2))})
    return entries


def toc_candidate(text: str) -> bool:
    """Find likely TOCs, leaving final classification to the cached judge call."""
    count = len(toc_entries(text))
    marker = re.search(
        r"\b(?:table of contents|contents|table des matieres|sommaire|inhaltsverzeichnis|indice)\b",
        normalized_title(" ".join(structural_lines(text))),
    )
    return count >= 9 or (count >= 3 and marker is not None)


def confirm_tocs(sources, config, inference) -> dict[str, dict]:
    """Confirm bounded candidate units through the existing schema/cache/retry path."""
    candidates = [source for source in sources if toc_candidate(source.text)]
    by_id = {s.unit_id: s for s in sources}
    bounded = [s for s in candidates if context_chars([s.unit_id], by_id) <= config.max_context_chars]
    responses = (
        inference.generate(
            [
                Request(prompts.render("check_toc", page=s.text, language=s.language), TOCConfirmation, "judge")
                for s in bounded
            ]
        )
        if bounded
        else []
    )
    result = {s.unit_id: response.model_dump() for s, response in zip(bounded, responses, strict=True)}
    for source in candidates:
        if source.unit_id not in result:
            result[source.unit_id] = {
                "has_table_of_contents": False,
                "explanation": "Candidate exceeds the text bound; no truncated confirmation request was made.",
            }
    return result


def title_locations(sources: list[RetrievalSource], excluded: set[str]) -> dict[str, set[int]]:
    """Index short source lines for exact TOC-title anchors, excluding TOC candidates."""
    locations = defaultdict(set)
    for index, source in enumerate(sources):
        if source.unit_id in excluded:
            continue
        for line in structural_lines(source.text):
            if len(line) <= 200:
                title = normalized_title(line)
                if title:
                    locations[title].add(index)
    return locations


def page_coordinates(sources: list[RetrievalSource]) -> dict[int, int]:
    """Use explicit, strictly increasing page metadata; never infer pages from unit IDs."""
    pages = [source.page_number for source in sources]
    if any(page is None for page in pages) or any(a >= b for a, b in zip(pages, pages[1:])):
        return {}
    return {page: index for index, page in enumerate(pages)}


def ordered_toc_entries(sources, confirmations) -> tuple[list[dict], list[dict]]:
    """Read unique entries in source order and reject nonpositive printed labels."""
    valid, rejected, seen = [], [], set()
    for source in sources:
        if not confirmations.get(source.unit_id, {}).get("has_table_of_contents", False):
            continue
        for entry in toc_entries(source.text):
            entry = {**entry, "toc_unit_id": source.unit_id}
            identity = (normalized_title(entry["title"]), entry["page"])
            if identity in seen:
                continue
            seen.add(identity)
            if entry["page"] <= 0:
                rejected.append({**entry, "reason": "nonpositive_page"})
                continue
            valid.append(entry)
    return valid, rejected


def toc_boundaries(sources, confirmations, candidates) -> tuple[dict[int, list[str]], dict]:
    """Map confirmed TOC entries using exact titles or a corroborated page offset.

    Two distinct printed page labels must agree on an offset before numeric
    projection is allowed. Disagreement disables projection; uniquely matched
    source titles remain usable. Unmapped, out-of-range and backward entries
    are rejected rather than clamped or sorted into plausible boundaries.
    """
    entries, rejected = ordered_toc_entries(sources, confirmations)
    if not entries:
        return defaultdict(list), {"offset": None, "anchor_page_count": 0, "accepted": [], "rejected": rejected}
    locations = title_locations(sources, candidates)
    coordinates = page_coordinates(sources)
    anchors = {}
    for entry in entries:
        matches = locations.get(normalized_title(entry["title"]), set())
        if len(matches) == 1 and coordinates:
            index = next(iter(matches))
            anchors.setdefault(entry["page"], set()).add(sources[index].page_number - entry["page"])
    offsets = set().union(*anchors.values()) if anchors else set()
    offset = next(iter(offsets)) if len(anchors) >= 2 and len(offsets) == 1 else None
    accepted, boundaries, previous, previous_page = [], defaultdict(list), -1, 0
    for entry in entries:
        matches = locations.get(normalized_title(entry["title"]), set())
        direct = next(iter(matches)) if len(matches) == 1 else None
        projected = coordinates.get(entry["page"] + offset) if offset is not None else None
        index = direct if direct is not None else projected
        reason = None
        if direct is not None and projected is not None and direct != projected:
            reason = "conflicting_title_and_offset"
        elif index is None:
            reason = "out_of_range_or_unmapped_page" if offset is not None else "unverified_page_offset"
        elif index < previous or entry["page"] < previous_page:
            reason = "backward_source_boundary"
        if reason:
            rejected.append({**entry, "reason": reason})
            continue
        previous = index
        previous_page = entry["page"]
        boundaries[index].append("toc_title" if direct is not None else "toc_offset")
        accepted.append({**entry, "unit_id": sources[index].unit_id, "method": boundaries[index][-1]})
    return boundaries, {"offset": offset, "anchor_page_count": len(anchors), "accepted": accepted, "rejected": rejected}


def document_boundaries(sources, confirmations) -> tuple[dict[int, list[str]], dict]:
    """Combine nonrepeated markdown headings and validated TOC boundaries."""
    candidates = {s.unit_id for s in sources if toc_candidate(s.text)}
    excluded = {key for key in candidates if confirmations.get(key, {}).get("has_table_of_contents", True)}
    titles = [heading_titles(s.text) if s.unit_id not in excluded else [] for s in sources]
    counts = Counter(title for group in titles for title in set(group))
    boundaries, audit = toc_boundaries(sources, confirmations, excluded)
    for index, group in enumerate(titles):
        if any(title and counts[title] == 1 for title in group):
            boundaries[index].append("heading")
    audit["candidate_unit_ids"] = sorted(candidates)
    audit["boundaries"] = [
        {"unit_id": sources[i].unit_id, "index": i, "evidence": evidence} for i, evidence in sorted(boundaries.items())
    ]
    return boundaries, audit


def section_ranges(length: int, boundaries: set[int], size: int) -> list[tuple[int, int]]:
    """Prefer structure near the target size, bound sections, and always flush the tail.

    Structural ends in [size, 2*size] take priority, followed by an earlier
    structural end. Gaps without usable boundaries use fixed-size sections.
    All intervals are contiguous, nonempty and half-open; no text is rebuilt.
    """
    output, start = [], 0
    ordered = sorted(i for i in boundaries if 0 < i < length)
    while start < length:
        near = [i for i in ordered if start + size <= i <= start + 2 * size]
        early = [i for i in ordered if start < i < start + size]
        end = near[0] if near else early[-1] if early else min(start + size, length)
        output.append((start, end))
        start = end
    return output


def verify_section_coverage(contexts, sources, maximum: int) -> dict:
    """Fail before inference if section memberships lose, repeat or reorder any unit."""
    expected = defaultdict(list)
    by_id = {source.unit_id: source for source in sources}
    for source in sources:
        expected[(source.document_id, source.language)].append(source.unit_id)
    actual = defaultdict(list)
    for context in contexts:
        if not context.unit_ids or len(context.unit_ids) > maximum or any(k not in by_id for k in context.unit_ids):
            raise ValueError("Invalid section membership or size")
        groups = {(by_id[k].document_id, by_id[k].language) for k in context.unit_ids}
        if len(groups) != 1 or context.language != next(iter(groups))[1]:
            raise ValueError("Section crosses document/language boundaries")
        actual[next(iter(groups))].extend(context.unit_ids)
    if dict(actual) != dict(expected):
        raise ValueError("Section coverage must preserve every source unit exactly once in reading order")
    return {
        "input_units": len(sources),
        "covered_units": sum(map(len, actual.values())),
        "exact_ordered_coverage": True,
    }


def build_sections(sources, config, confirmations=None) -> tuple[list[GenerationContext], dict]:
    """Plan whole-unit structural sections and return a reproducible boundary audit.

    Args:
        sources: Canonical units in caller-provided reading order.
        config: Section target size and output settings.
        confirmations: Recorded TOC classifications by source identity. Missing
            confirmations never authorize numeric TOC projections.

    Returns:
        Sections and an audit of boundaries, rejected TOC entries and coverage.

    Raises:
        ValueError: Memberships do not cover each input unit exactly once in order.
    """
    confirmations = confirmations or {}
    by_id = {s.unit_id: s for s in sources}
    output, documents = [], []
    for document in automatic_contexts(sources, "document"):
        members = [by_id[k] for k in document.unit_ids]
        boundaries, audit = document_boundaries(members, confirmations)
        audit["sections"] = []
        for start, end in section_ranges(len(members), set(boundaries), config.section_size):
            ids = document.unit_ids[start:end]
            audit["sections"].append(
                {
                    "start": start,
                    "end": end,
                    "end_reason": "structural"
                    if end in boundaries
                    else "document_end"
                    if end == len(members)
                    else "fixed_size",
                }
            )
            output.append(
                GenerationContext(
                    context_id="section_" + fingerprint([document.context_id, ids])[:32],
                    unit_ids=ids,
                    language=document.language,
                )
            )
        documents.append({"document_id": members[0].document_id, "language": document.language, **audit})
    coverage = verify_section_coverage(output, sources, 2 * config.section_size)
    return output, {
        "policy": "whole-unit-structure-v1",
        "section_size": config.section_size,
        "max_section_units": 2 * config.section_size,
        "toc_confirmations": confirmations,
        "documents": documents,
        "coverage": coverage,
    }


def section_contexts(
    sources: list[RetrievalSource], config: MultimodalSDGConfig, inference=None
) -> list[GenerationContext]:
    """Confirm TOCs through cached inference when supplied, then check complete coverage.

    Without inference, only headings contribute structural evidence. Explicit
    caller contexts bypass this automatic planner in the workflow. Confirmation
    failures propagate through the existing retry policy, never become approvals.
    """
    confirmations = confirm_tocs(sources, config, inference) if inference is not None else {}
    contexts, audit = build_sections(sources, config, confirmations)
    if inference is not None:
        write_json(config.output_dir / "planning/section_boundaries.json", audit)
    return contexts
