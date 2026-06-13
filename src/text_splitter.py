from __future__ import annotations

import re


BOUNDARY_RE = re.compile(r"(?m)(?=^#{1,6}\s)|(?=^```)|(?<=```\n)")


def _semantic_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    current: list[str] = []
    in_code = False
    for line in text.splitlines(keepends=True):
        is_fence = line.lstrip().startswith("```")
        is_heading = bool(re.match(r"^#{1,6}\s", line))
        if is_heading and current and not in_code:
            blocks.append("".join(current).strip())
            current = []
        if is_fence and not in_code and current:
            blocks.append("".join(current).strip())
            current = []
        current.append(line)
        if is_fence:
            in_code = not in_code
            if not in_code:
                blocks.append("".join(current).strip())
                current = []
    if current:
        blocks.append("".join(current).strip())
    return [block for block in blocks if block]


def _split_long(block: str, max_chars: int, overlap: int) -> list[str]:
    if len(block) <= max_chars:
        return [block]
    result: list[str] = []
    start = 0
    while start < len(block):
        end = min(len(block), start + max_chars)
        if end < len(block):
            boundary = block.rfind("\n", start, end)
            if boundary > start + max_chars // 2:
                end = boundary
        result.append(block[start:end].strip())
        if end == len(block):
            break
        start = max(start + 1, end - overlap)
    return [item for item in result if item]


def split_document(document: dict, max_chars: int = 1200, overlap: int = 120) -> list[dict]:
    pieces: list[str] = []
    for block in _semantic_blocks(document["content"]):
        pieces.extend(_split_long(block, max_chars, overlap))
    chunks = []
    for index, content in enumerate(pieces, start=1):
        heading = re.search(r"^#{1,6}\s+(.+)$", content, re.M)
        section_title = heading.group(1).strip() if heading else ""
        chunks.append({
            "content": content,
            "text": content,
            "source": document["source"],
            "product": document["product"],
            "path": document.get("path", ""),
            "source_path": document.get("path", ""),
            "title": document["source"].rsplit(".", 1)[0].replace("_", " "),
            "section_title": section_title,
            "has_code": "```" in content,
            "chunk_id": f"chunk_{index:03d}",
        })
    return chunks


def split_documents(documents: list[dict], max_chars: int = 1200, overlap: int = 120) -> list[dict]:
    chunks: list[dict] = []
    for document in documents:
        chunks.extend(split_document(document, max_chars=max_chars, overlap=overlap))
    return chunks
