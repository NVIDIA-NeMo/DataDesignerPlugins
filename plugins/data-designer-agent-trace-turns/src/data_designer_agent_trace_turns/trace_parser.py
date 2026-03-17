# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from data_designer.engine.processing.utils import deserialize_json_values

CONVERSATION_KEYS = frozenset({"chat", "conversation", "conversations", "dialog", "dialogue", "thread", "threads"})
TURN_COLLECTION_KEYS = frozenset({"events", "items", "messages", "steps", "turns"})
ROLE_KEYS = ("role", "sender_role", "author_role")
SPEAKER_KEYS = ("speaker", "sender", "participant", "actor", "name")
ITEM_KIND_KEYS = ("kind", "message_type", "event_type")
TEXT_KEYS = (
    "content",
    "text",
    "message",
    "value",
    "body",
    "output_text",
    "input_text",
    "prompt",
    "response",
    "output",
    "input",
)
KNOWN_ROLE_VALUES = {"assistant", "developer", "model", "system", "tool", "user"}


@dataclass(frozen=True)
class ExtractedTraceTurn:
    """A normalized conversational step extracted from a trace payload.

    Attributes:
        conversation_index: Zero-based conversation index within the trace.
        turn_index: Zero-based turn index within the conversation.
        text: Primary text representation extracted from the turn.
        role: Role for the turn, when available.
        speaker: Speaker label for the turn, when available.
        turn_kind: Kind of conversational item, such as `message` or `turn`.
        trace_path: JSONPath-like path pointing to the extracted turn.
        raw_turn_json: Compact JSON representation of the extracted turn payload.
    """

    conversation_index: int
    turn_index: int
    text: str | None
    role: str | None
    speaker: str | None
    turn_kind: str
    trace_path: str
    raw_turn_json: str


@dataclass(frozen=True)
class _TraceTurnCandidate:
    conversation_key: str
    text: str | None
    role: str | None
    speaker: str | None
    turn_kind: str
    trace_path: str
    raw_turn_json: str


def extract_trace_turns(trace_value: Any) -> list[ExtractedTraceTurn]:
    """Extract normalized conversational turns from an agent trace payload.

    Args:
        trace_value: Raw trace payload. Supported values include Python dict/list
            objects, JSON strings, and Data Designer `__trace` values.

    Returns:
        A list of normalized turn records in source order.
    """

    payload = _normalize_trace_value(trace_value)
    candidates = _extract_turn_candidates(payload, path="$", conversation_key="$", item_kind_hint=None)
    return _finalize_candidates(candidates)


def _normalize_trace_value(trace_value: Any) -> Any:
    return deserialize_json_values(_to_python_value(trace_value))


def _to_python_value(value: Any) -> Any:
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _to_python_value(value.model_dump(mode="python"))
    if hasattr(value, "dict") and callable(value.dict):
        return _to_python_value(value.dict())
    if isinstance(value, Mapping):
        return {key: _to_python_value(item) for key, item in value.items()}
    if _is_sequence(value):
        return [_to_python_value(item) for item in value]
    return value


def _extract_turn_candidates(
    node: Any,
    *,
    path: str,
    conversation_key: str,
    item_kind_hint: str | None,
) -> list[_TraceTurnCandidate]:
    if node is None:
        return []
    if isinstance(node, Mapping):
        return _extract_from_mapping(node, path=path, conversation_key=conversation_key, item_kind_hint=item_kind_hint)
    if _is_sequence(node):
        return _extract_from_sequence(node, path=path, conversation_key=conversation_key, item_kind_hint=item_kind_hint)
    return _extract_from_scalar(node, path=path, conversation_key=conversation_key, item_kind_hint=item_kind_hint)


def _extract_from_mapping(
    node: Mapping[str, Any],
    *,
    path: str,
    conversation_key: str,
    item_kind_hint: str | None,
) -> list[_TraceTurnCandidate]:
    explicit_results: list[_TraceTurnCandidate] = []
    for key, value in node.items():
        child_path = _append_path(path, key)
        if key in CONVERSATION_KEYS and _is_container(value):
            explicit_results.extend(_extract_conversation_group(value, child_path))
        elif key in TURN_COLLECTION_KEYS and _is_sequence(value):
            collection_kind = _singularize_item_kind(key)
            for index, item in enumerate(value):
                explicit_results.extend(
                    _extract_turn_candidates(
                        item,
                        path=f"{child_path}[{index}]",
                        conversation_key=conversation_key,
                        item_kind_hint=collection_kind,
                    )
                )
    if explicit_results:
        return explicit_results
    if _looks_like_turn(node):
        return [_build_candidate(node, path=path, conversation_key=conversation_key, item_kind_hint=item_kind_hint)]

    generic_results: list[_TraceTurnCandidate] = []
    for key, value in node.items():
        if _is_container(value):
            generic_results.extend(
                _extract_turn_candidates(
                    value,
                    path=_append_path(path, key),
                    conversation_key=conversation_key,
                    item_kind_hint=item_kind_hint,
                )
            )
    return generic_results


def _extract_conversation_group(node: Any, path: str) -> list[_TraceTurnCandidate]:
    if _is_sequence(node):
        results: list[_TraceTurnCandidate] = []
        for index, item in enumerate(node):
            conversation_path = f"{path}[{index}]"
            results.extend(
                _extract_turn_candidates(
                    item,
                    path=conversation_path,
                    conversation_key=conversation_path,
                    item_kind_hint=None,
                )
            )
        return results
    return _extract_turn_candidates(node, path=path, conversation_key=path, item_kind_hint=None)


def _extract_from_sequence(
    node: Sequence[Any],
    *,
    path: str,
    conversation_key: str,
    item_kind_hint: str | None,
) -> list[_TraceTurnCandidate]:
    inferred_item_kind = item_kind_hint or _infer_sequence_item_kind(node)
    results: list[_TraceTurnCandidate] = []
    for index, item in enumerate(node):
        results.extend(
            _extract_turn_candidates(
                item,
                path=f"{path}[{index}]",
                conversation_key=conversation_key,
                item_kind_hint=inferred_item_kind,
            )
        )
    return results


def _extract_from_scalar(
    node: Any,
    *,
    path: str,
    conversation_key: str,
    item_kind_hint: str | None,
) -> list[_TraceTurnCandidate]:
    text = _extract_text_from_value(node)
    if text is None or (item_kind_hint is None and path != "$"):
        return []
    return [
        _TraceTurnCandidate(
            conversation_key=conversation_key,
            text=text,
            role=None,
            speaker=None,
            turn_kind=item_kind_hint or "turn",
            trace_path=path,
            raw_turn_json=_compact_json(node),
        )
    ]


def _looks_like_turn(node: Mapping[str, Any]) -> bool:
    if any(key in node and _is_sequence(node[key]) for key in TURN_COLLECTION_KEYS):
        return False

    has_role = _extract_role(node) is not None
    has_speaker = _extract_explicit_speaker(node) is not None
    has_tool_data = bool(node.get("tool_calls")) or node.get("tool_call_id") is not None
    has_reasoning = _extract_text_from_value(node.get("reasoning_content")) is not None
    has_text = _extract_text(node) is not None

    return (has_role or has_speaker or has_tool_data or has_reasoning) and has_text


def _build_candidate(
    node: Mapping[str, Any],
    *,
    path: str,
    conversation_key: str,
    item_kind_hint: str | None,
) -> _TraceTurnCandidate:
    return _TraceTurnCandidate(
        conversation_key=conversation_key,
        text=_extract_text(node),
        role=_extract_role(node),
        speaker=_extract_speaker(node),
        turn_kind=_extract_turn_kind(node, item_kind_hint),
        trace_path=path,
        raw_turn_json=_compact_json(node),
    )


def _extract_role(node: Mapping[str, Any]) -> str | None:
    role = _first_text(node, ROLE_KEYS)
    if role is not None:
        return role.lower()

    author = node.get("author")
    if isinstance(author, Mapping):
        author_role = _first_text(author, ("role",))
        if author_role is not None:
            return author_role.lower()
    if isinstance(author, str) and author.lower() in KNOWN_ROLE_VALUES:
        return author.lower()
    return None


def _extract_speaker(node: Mapping[str, Any]) -> str | None:
    speaker = _extract_explicit_speaker(node)
    if speaker is not None:
        return speaker
    return _extract_role(node)


def _extract_explicit_speaker(node: Mapping[str, Any]) -> str | None:
    speaker = _first_text(node, SPEAKER_KEYS)
    if speaker is not None:
        return speaker

    author = node.get("author")
    if isinstance(author, Mapping):
        return _first_text(author, ("name", "display_name", "id"))
    if isinstance(author, str):
        return author
    return None


def _extract_turn_kind(node: Mapping[str, Any], item_kind_hint: str | None) -> str:
    kind = _first_text(node, ITEM_KIND_KEYS)
    if kind is None:
        raw_type = _first_text(node, ("type",))
        if raw_type in {"event", "message", "step", "tool_call", "turn"}:
            kind = raw_type
    return kind or item_kind_hint or "turn"


def _extract_text(node: Mapping[str, Any]) -> str | None:
    for key in TEXT_KEYS:
        if key not in node:
            continue
        text = _extract_text_from_value(node[key])
        if text is not None:
            return text

    reasoning_text = _extract_text_from_value(node.get("reasoning_content"))
    if reasoning_text is not None:
        return reasoning_text

    tool_calls = node.get("tool_calls")
    if tool_calls:
        return _compact_json(tool_calls)
    return None


def _extract_text_from_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, (bool, int, float)):
        return str(value)
    if isinstance(value, Mapping):
        for key in ("text", "content", "message", "value", "body", "reasoning_content"):
            if key in value:
                nested_text = _extract_text_from_value(value[key])
                if nested_text is not None:
                    return nested_text
        for key in ("parts", "segments", "items", "function"):
            if key in value:
                nested_text = _extract_text_from_value(value[key])
                if nested_text is not None:
                    return nested_text
        if "arguments" in value or "name" in value:
            return _compact_json(value)
        if value:
            return _compact_json(value)
        return None
    if _is_sequence(value):
        parts = [part for item in value if (part := _extract_text_from_value(item)) is not None]
        if not parts:
            return None
        return "\n\n".join(parts)
    return str(value)


def _first_text(node: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        value = node.get(key)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped
    return None


def _infer_sequence_item_kind(node: Sequence[Any]) -> str | None:
    if not node:
        return None
    first_item = node[0]
    if isinstance(first_item, Mapping) and _extract_role(first_item) is not None:
        return "message"
    return None


def _finalize_candidates(candidates: list[_TraceTurnCandidate]) -> list[ExtractedTraceTurn]:
    conversation_indices: dict[str, int] = {}
    turn_counts: defaultdict[str, int] = defaultdict(int)
    turns: list[ExtractedTraceTurn] = []

    for candidate in candidates:
        if candidate.conversation_key not in conversation_indices:
            conversation_indices[candidate.conversation_key] = len(conversation_indices)
        conversation_index = conversation_indices[candidate.conversation_key]
        turn_index = turn_counts[candidate.conversation_key]
        turn_counts[candidate.conversation_key] += 1
        turns.append(
            ExtractedTraceTurn(
                conversation_index=conversation_index,
                turn_index=turn_index,
                text=candidate.text,
                role=candidate.role,
                speaker=candidate.speaker,
                turn_kind=candidate.turn_kind,
                trace_path=candidate.trace_path,
                raw_turn_json=candidate.raw_turn_json,
            )
        )

    return turns


def _append_path(path: str, key: str) -> str:
    if path == "$":
        return f"$.{key}"
    return f"{path}.{key}"


def _singularize_item_kind(value: str) -> str:
    if value.endswith("ies"):
        return f"{value[:-3]}y"
    if value.endswith("s"):
        return value[:-1]
    return value


def _compact_json(value: Any) -> str:
    return json.dumps(_make_jsonable(value), ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _make_jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _make_jsonable(value.model_dump(mode="python"))
    if hasattr(value, "dict") and callable(value.dict):
        return _make_jsonable(value.dict())
    if isinstance(value, Mapping):
        return {str(key): _make_jsonable(item) for key, item in value.items()}
    if _is_sequence(value):
        return [_make_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _is_container(value: Any) -> bool:
    return isinstance(value, Mapping) or _is_sequence(value)


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str))
