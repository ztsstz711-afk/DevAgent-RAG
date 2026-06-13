from __future__ import annotations

import re


ERROR_PATTERNS = [
    ("cuda_oom", re.compile(r"CUDA.*out of memory|CUDA OOM", re.I), "GPU memory exhausted"),
    ("module_not_found", re.compile(r"ModuleNotFoundError(?::\s*No module named\s*['\"]?([^'\"\s]+))?", re.I), "Python module is missing"),
    ("openai_api_key_missing", re.compile(r"OPENAI_API_KEY.*(?:missing|not set|not found)|(?:missing|set).*OPENAI_API_KEY", re.I), "OpenAI API key is not configured"),
    ("runtime_error", re.compile(r"RuntimeError(?::\s*([^\n]+))?", re.I), "Runtime error"),
]


def parse_error(text: str) -> dict:
    matches = []
    for error_type, pattern, summary in ERROR_PATTERNS:
        match = pattern.search(text)
        if match:
            matches.append({
                "type": error_type,
                "summary": summary,
                "detail": match.group(1).strip() if match.lastindex and match.group(1) else "",
                "matched_text": match.group(0),
            })
    return {
        "is_error": bool(matches),
        "primary_type": matches[0]["type"] if matches else "unknown",
        "errors": matches,
        "raw": text,
    }
