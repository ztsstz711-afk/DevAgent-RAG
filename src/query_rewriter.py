"""Rule-based query rewriting and keyword extraction.

The rewriter is intentionally lightweight and dependency-free. It is not wired
into the main RAG graph yet; callers can use its structured output to inspect or
later improve retrieval queries.
"""

from __future__ import annotations

import re
from typing import Any


LIBRARY_ALIASES = {
    "openai": ("openai",),
    "langchain": ("langchain",),
    "langgraph": ("langgraph",),
    "pytorch": ("pytorch",),
    "torch": ("torch",),
    "transformers": ("transformers",),
    "huggingface": ("huggingface", "hugging face"),
    "vllm": ("vllm",),
    "llamafactory": ("llamafactory", "llama-factory", "llama factory"),
}

API_NAME_RE = re.compile(r"\b[A-Z][A-Za-z0-9_]*(?:\.[A-Z][A-Za-z0-9_]*)?\b|\b[a-z_]+_pretrained\b")
MODULE_NOT_FOUND_RE = re.compile(r"(?:ModuleNotFoundError|ImportError).*?No module named ['\"]?([^'\"\s]+)", re.I | re.S)
ERROR_TYPE_ALIASES = {
    "module_not_found": "ModuleNotFoundError",
    "cuda_oom": "CUDA out of memory",
    "openai_api_key_missing": "OPENAI_API_KEY missing",
    "runtime_error": "RuntimeError",
}


class QueryRewriter:
    """Extract retrieval-oriented query metadata with conservative rewrites."""

    def rewrite(self, query: str | None, route: str | None = None, error_info: dict | None = None) -> dict[str, Any]:
        original_query = query or ""
        normalized_query = " ".join(original_query.split())
        libraries = self._extract_libraries(normalized_query)
        api_names = self._extract_api_names(normalized_query)
        error_type = self._detect_error_type(normalized_query, route, error_info)
        module_name = self._extract_module_name(normalized_query, error_info)

        if error_type:
            search_query, reason = self._rewrite_error_query(
                normalized_query, error_type, module_name, libraries, api_names
            )
        else:
            search_query, reason = normalized_query, "none"

        keywords = self._keywords(search_query, libraries, api_names, module_name, error_type)
        return {
            "original_query": original_query,
            "search_query": search_query,
            "keywords": keywords,
            "error_type": error_type,
            "module_name": module_name,
            "library_names": libraries,
            "api_names": api_names,
            "rewrite_applied": search_query != normalized_query,
            "rewrite_reason": reason,
        }

    def _extract_libraries(self, text: str) -> list[str]:
        lowered = text.lower()
        found = []
        for canonical, aliases in LIBRARY_ALIASES.items():
            if any(re.search(rf"(?<![a-z0-9_]){re.escape(alias)}(?![a-z0-9_])", lowered) for alias in aliases):
                found.append(canonical)
        return found

    def _extract_api_names(self, text: str) -> list[str]:
        names = []
        for match in API_NAME_RE.findall(text):
            if match in {"CUDA", "OOM", "API", "KEY"}:
                continue
            if match not in names:
                names.append(match)
        return names

    def _detect_error_type(self, text: str, route: str | None, error_info: dict | None) -> str | None:
        if error_info and error_info.get("primary_type") not in {None, "unknown"}:
            raw_type = str(error_info["primary_type"])
            return ERROR_TYPE_ALIASES.get(raw_type, raw_type)

        lowered = text.lower()
        if "module not found" in lowered or "modulenotfounderror" in lowered:
            return "ModuleNotFoundError"
        if "importerror" in lowered:
            return "ImportError"
        if ("cuda" in lowered and "out of memory" in lowered) or "cuda oom" in lowered:
            return "CUDA out of memory"
        if "openai_api_key" in lowered and any(term in lowered for term in ("missing", "not set", "not found")):
            return "OPENAI_API_KEY missing"
        if "runtimeerror" in lowered or "runtime error" in lowered:
            return "RuntimeError"
        if route == "error":
            return "unknown_error"
        return None

    def _extract_module_name(self, text: str, error_info: dict | None) -> str | None:
        if error_info:
            for item in error_info.get("errors", []):
                detail = item.get("detail")
                if detail:
                    return str(detail).strip("'\"")

        match = MODULE_NOT_FOUND_RE.search(text)
        if match:
            return match.group(1).strip("'\"")
        return None

    def _rewrite_error_query(
        self,
        text: str,
        error_type: str,
        module_name: str | None,
        libraries: list[str],
        api_names: list[str],
    ) -> tuple[str, str]:
        lowered_type = error_type.lower()
        if "module" in lowered_type or "import" in lowered_type:
            parts = [error_type, module_name, *libraries, "installation", "import error"]
            return self._join(parts), "module_import_error"
        if "cuda" in lowered_type and "memory" in lowered_type:
            parts = ["CUDA out of memory", "pytorch", "memory", "batch size"]
            return self._join(parts), "cuda_memory_error"
        if "openai_api_key" in lowered_type:
            parts = ["OPENAI_API_KEY missing", "openai", "api key", "authentication", "environment variable"]
            return self._join(parts), "openai_api_key_missing"
        if "runtime" in lowered_type:
            parts = ["RuntimeError", *libraries, *api_names, "debug runtime error"]
            return self._join(parts), "runtime_error"
        return self._join([error_type, *libraries, *api_names, text[:120]]), "generic_error"

    def _keywords(
        self,
        search_query: str,
        libraries: list[str],
        api_names: list[str],
        module_name: str | None,
        error_type: str | None,
    ) -> list[str]:
        tokens = re.findall(r"[A-Za-z0-9_+-]+", search_query)
        values = [*tokens, *libraries, *api_names]
        if module_name:
            values.append(module_name)
        if error_type:
            values.append(error_type)

        deduped = []
        seen = set()
        for value in values:
            key = value.lower()
            if key and key not in seen:
                seen.add(key)
                deduped.append(value)
        return deduped

    def _join(self, parts: list[str | None]) -> str:
        values = []
        seen = set()
        for part in parts:
            if not part:
                continue
            cleaned = " ".join(str(part).split())
            key = cleaned.lower()
            if cleaned and key not in seen:
                seen.add(key)
                values.append(cleaned)
        return " ".join(values)


def rewrite(query: str | None, route: str | None = None, error_info: dict | None = None) -> dict[str, Any]:
    """Convenience wrapper using the default rule-based rewriter."""
    return QueryRewriter().rewrite(query, route=route, error_info=error_info)
