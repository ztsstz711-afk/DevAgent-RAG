from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    question: str
    route: str
    error_info: dict[str, Any]
    documents: list[dict[str, Any]]
    evidence_assessment: dict[str, Any]
    valid_documents: list[dict[str, Any]]
    code_snippets: list[dict[str, Any]]
    answer: str
    answer_backend: str
    answer_fallback_reason: str | None
    quality: dict[str, Any]
    tool_trace: list[dict[str, Any]]
