from __future__ import annotations

import re


CITATION_RE = re.compile(r"\[[^\]|]+\s*\|\s*[^\]|]+\s*\|\s*chunk_\d{3}\]")


def check_quality(answer: str, min_citations: int = 1, no_evidence: bool = False) -> dict:
    citations = CITATION_RE.findall(answer)
    issues: list[str] = []
    if not answer.strip():
        issues.append("empty_answer")
    if no_evidence:
        issues.append("no_evidence")
    elif len(citations) < min_citations:
        issues.append("missing_citation")
    if len(answer.strip()) < 80 and not no_evidence:
        issues.append("answer_too_short")
    return {
        "passed": not issues,
        "score": round(max(0.0, 1.0 - 0.25 * len(issues)), 2),
        "citations": citations,
        "issues": issues,
    }
