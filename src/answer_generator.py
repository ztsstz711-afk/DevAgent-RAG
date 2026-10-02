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


def _query_tokens(text: str) -> set[str]:
    stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from",
        "how", "i", "in", "is", "it", "of", "on", "or", "the", "this", "to", "what", "when",
        "where", "which", "with", "you", "according", "documentation",
    }
    return {
        token for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]*", text.lower())
        if token not in stopwords and len(token) > 1
    }


def _focused_summary(question: str, text: str, limit: int = 360) -> tuple[str, int]:
    """Return the most query-relevant source sentences and their overlap score.

    The template path is extractive by design: it should foreground the source
    passage that answers the question, rather than blindly emitting the first
    360 characters of a retrieved chunk.
    """
    cleaned = re.sub(r"```.*?```", "", text, flags=re.S)
    cleaned = re.sub(r"^#{1,6}\s*", "", cleaned, flags=re.M)
    sentences = [" ".join(sentence.split()) for sentence in re.split(r"(?<=[.!?])\s+|\n+", cleaned) if sentence.strip()]
    # Scraped documentation often stores a heading on its own line. Joining a
    # short heading to the following sentence preserves the subject needed for
    # a useful extractive answer (for example, "AutoClass: ...").
    merged_sentences: list[str] = []
    for sentence in sentences:
        if merged_sentences and len(merged_sentences[-1]) < 24:
            merged_sentences[-1] = f"{merged_sentences[-1]}: {sentence}"
        else:
            merged_sentences.append(sentence)
    sentences = merged_sentences
    query_tokens = _query_tokens(question)

    def overlap_score(sentence: str) -> int:
        sentence_tokens = _query_tokens(sentence)
        matched = 0
        for token in query_tokens:
            if token in sentence_tokens or any(candidate.startswith(token[:4]) or token.startswith(candidate[:4]) for candidate in sentence_tokens if len(token) >= 4 and len(candidate) >= 4):
                matched += 1
        return matched

    scored = [
        (overlap_score(sentence), index, sentence)
        for index, sentence in enumerate(sentences)
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = [item for item in scored if item[0] > 0][:2]
    if not selected:
        return _summary(text, limit), 0
    selected.sort(key=lambda item: item[1])
    summary = " ".join(item[2] for item in selected)
    return summary[:limit].rstrip(" ,.;") + ("..." if len(summary) > limit else ""), selected[0][0]


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
    lines.append("\n### Evidence-grounded answer")
    focused = []
    for original_index, chunk in enumerate(documents):
        summary, overlap = _focused_summary(question, chunk["content"])
        focused.append((overlap, original_index, chunk, summary))
    focused.sort(key=lambda item: (-item[0], item[1]))
    for _, _, chunk, summary in focused[:3]:
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
