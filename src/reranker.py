"""Lightweight optional lexical reranker.

This module is intentionally dependency-free. LexicalReranker is optionally used
after retrieval when reranker.enabled is true, while preserving original document
fields.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


Document = Dict[str, Any]


class LexicalReranker:
    """Rerank documents with query/document lexical overlap and error signals."""

    ERROR_PATTERNS: Sequence[Tuple[str, Sequence[str]]] = (
        ("module_not_found", ("modulenotfounderror", "importerror", "no module named", "module not found")),
        ("cuda_oom", ("cuda", "out of memory", "oom", "cuda out of memory")),
        ("runtime_error", ("runtimeerror", "runtime error")),
        ("openai_api_key", ("openai_api_key", "api key", "authentication", "missing key")),
    )

    def rerank(self, query: str, documents: Sequence[Document], top_k: Optional[int] = None) -> List[Document]:
        """Return documents sorted by lexical relevance.

        The input document dictionaries are not mutated. If ``top_k`` is None,
        all reranked documents are returned.
        """
        if not documents:
            return []

        query_text = query or ""
        query_tokens = self._tokens(query_text)
        query_errors = self._matched_error_types(query_text)

        scored = []
        for index, document in enumerate(documents):
            score = self._score_document(query_tokens, query_errors, document)
            scored.append((score, index, document))

        scored.sort(key=lambda item: (-item[0], item[1]))
        reranked = [document for _, _, document in scored]
        if top_k is None:
            return reranked
        return reranked[: max(0, top_k)]

    def _score_document(self, query_tokens: set[str], query_errors: set[str], document: Document) -> float:
        doc_text = self._document_text(document)
        doc_tokens = self._tokens(doc_text)

        overlap_score = self._overlap_score(query_tokens, doc_tokens)
        error_score = self._error_score(query_errors, doc_text)
        original_score = self._original_score(document)

        return overlap_score + error_score + original_score

    def _document_text(self, document: Document) -> str:
        fields = (
            document.get("content"),
            document.get("text"),
            document.get("title"),
            document.get("section_title"),
            document.get("source"),
            document.get("source_path"),
            document.get("product"),
        )
        return " ".join(str(value) for value in fields if value)

    def _tokens(self, text: str) -> set[str]:
        return {token for token in re.findall(r"[a-zA-Z0-9_]+", text.lower()) if len(token) > 1}

    def _matched_error_types(self, text: str) -> set[str]:
        lowered = (text or "").lower()
        matches = set()
        for error_type, patterns in self.ERROR_PATTERNS:
            if any(pattern in lowered for pattern in patterns):
                matches.add(error_type)
        return matches

    def _error_score(self, query_errors: set[str], document_text: str) -> float:
        if not query_errors:
            return 0.0

        doc_errors = self._matched_error_types(document_text)
        if not doc_errors:
            return 0.0

        return 2.0 * len(query_errors & doc_errors)

    def _overlap_score(self, query_tokens: set[str], doc_tokens: set[str]) -> float:
        if not query_tokens or not doc_tokens:
            return 0.0

        overlap = query_tokens & doc_tokens
        precision_like = len(overlap) / max(1, len(query_tokens))
        breadth = math.log1p(len(overlap))
        return precision_like + breadth

    def _original_score(self, document: Document) -> float:
        try:
            raw_score = float(document.get("score", 0.0))
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(raw_score, 1.0)) * 0.25


def rerank(query: str, documents: Sequence[Document], top_k: Optional[int] = None) -> List[Document]:
    """Convenience wrapper using the default lexical reranker."""
    return LexicalReranker().rerank(query, documents, top_k=top_k)
