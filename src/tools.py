from __future__ import annotations

from typing import Any

from .error_parser import parse_error
from .quality_checker import check_quality
from .retriever import TfidfRetriever


def doc_search_tool(query: str, retriever: TfidfRetriever, top_k: int = 4, min_score: float = 0.01) -> list[dict]:
    return retriever.search(query, top_k=top_k, min_score=min_score)


def error_parser_tool(text: str) -> dict[str, Any]:
    return parse_error(text)


def code_snippet_finder_tool(
    query: str, retriever: TfidfRetriever, top_k: int = 2, preferred_sources: list[str] | None = None
) -> list[dict]:
    candidates = retriever.search(query, top_k=max(top_k * 4, top_k), min_score=0.0, code_only=True)
    source_rank = {source: rank for rank, source in enumerate(dict.fromkeys(preferred_sources or []))}
    candidates.sort(key=lambda item: (source_rank.get(item["source"], 10_000), -item["score"]))
    return candidates[:top_k]


def quality_check_tool(
    answer: str,
    min_citations: int = 1,
    no_evidence: bool = False,
    evidence_issues: list[str] | None = None,
) -> dict:
    return check_quality(
        answer,
        min_citations=min_citations,
        no_evidence=no_evidence,
        evidence_issues=evidence_issues,
    )
