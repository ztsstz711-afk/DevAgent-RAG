from __future__ import annotations

import re


NO_EVIDENCE_ANSWER = "未在当前文档知识库中找到明确依据。"


def citation(chunk: dict) -> str:
    return f"[{chunk['product']} | {chunk['source']} | {chunk['chunk_id']}]"


def _summary(text: str, limit: int = 360) -> str:
    cleaned = re.sub(r"```.*?```", "", text, flags=re.S)
    cleaned = re.sub(r"^#{1,6}\s*", "", cleaned, flags=re.M)
    cleaned = " ".join(cleaned.split())
    return cleaned[:limit].rstrip(" ,.;") + ("..." if len(cleaned) > limit else "")


def generate_answer(
    question: str,
    documents: list[dict],
    error_info: dict | None = None,
    code_snippets: list[dict] | None = None,
    evidence_valid: bool = True,
) -> str:
    if not documents or not evidence_valid:
        return NO_EVIDENCE_ANSWER
    lines = [f"## Answer\n\nFor: `{question}`"]
    if error_info and error_info.get("is_error"):
        lines.append(f"\nDetected issue: **{error_info['primary_type']}**. Start by confirming the failing component and its runtime configuration.")
    lines.append("\n### Recommended approach")
    for chunk in documents[:3]:
        summary = _summary(chunk["content"])
        if summary:
            lines.append(f"- {summary} {citation(chunk)}")
    snippets = code_snippets or []
    if snippets:
        snippet = next((item for item in snippets if "```" in item["content"]), None)
        if snippet:
            code_match = re.search(r"```(?:\w+)?\n(.*?)```", snippet["content"], re.S)
            if code_match:
                lines.append(f"\n### Example\n\n```python\n{code_match.group(1).strip()}\n```\n{citation(snippet)}")
    lines.append("\n### Verification\n\nApply the smallest relevant change, rerun the failing command, and check the logs before increasing scope.")
    return "\n".join(lines)
