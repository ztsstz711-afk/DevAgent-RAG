"""Fetch a small, traceable official-document corpus for the v7 real evaluation.

GitHub issues stay in the evaluation-source register and are not downloaded as
retrieval documents, because they are user-generated and version-specific.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "openai_rate_limits": {
        "url": "https://developers.openai.com/api/docs/guides/rate-limits",
        "product": "openai",
    },
    "openai_structured_outputs": {
        "url": "https://developers.openai.com/api/docs/guides/structured-outputs",
        "product": "openai",
    },
    "pytorch_data": {
        "url": "https://docs.pytorch.org/docs/main/data.html",
        "product": "pytorch",
    },
    "langchain_retrieval": {
        "url": "https://docs.langchain.com/oss/python/deepagents/retrieval",
        "product": "langchain",
    },
    "huggingface_transformers_quickstart": {
        "url": "https://huggingface.co/docs/transformers/en/quicktour",
        "product": "huggingface",
    },
}


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style", "svg", "noscript"}:
            self.skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "svg", "noscript"} and self.skip_depth:
            self.skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self.skip_depth:
            value = " ".join(data.split())
            if value:
                self.parts.append(value)


def fetch(url: str) -> tuple[str, str]:
    request = Request(url, headers={"User-Agent": "DevAgent-RAG/0.1 local research"})
    with urlopen(request, timeout=30) as response:
        raw = response.read()
        resolved_url = response.geturl()
    parser = TextExtractor()
    parser.feed(raw.decode("utf-8", errors="replace"))
    text = "\n".join(parser.parts)
    if len(text) < 500:
        raise ValueError(f"Fetched content from {resolved_url} is unexpectedly short")
    return resolved_url, text


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch the curated v7 official documentation sources.")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "data" / "docs_imported" / "v7_curated")
    parser.add_argument("--manifest_path", type=Path, default=ROOT / "data" / "output" / "real_source_manifest_v7.json")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    captured_at = datetime.now(UTC).isoformat()
    entries = []
    for source_id, source in SOURCES.items():
        url = source["url"]
        resolved_url, text = fetch(url)
        output = args.output_dir / f"{source_id}.txt"
        output.write_text(text + "\n", encoding="utf-8")
        entries.append({
            "source_id": source_id,
            "product": source["product"],
            "source_type": "official_documentation",
            "requested_url": url,
            "resolved_url": resolved_url,
            "captured_at_utc": captured_at,
            "output_file": str(output.relative_to(ROOT)),
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "characters": len(text),
            "usage": "local retrieval and frozen real-question evaluation; no model training",
        })
    manifest = {
        "protocol": "devagent_real_official_docs_v7",
        "scope": "Five official documents (OpenAI, PyTorch, LangChain, and Hugging Face). GitHub issues remain metadata-backed evaluation sources, not imported document text.",
        "entries": entries,
    }
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # `load_documents()` treats an import manifest as authoritative.  This v7
    # protocol deliberately replaces, rather than extends, older imports: a
    # retrieval corpus for the frozen official-source evaluation must contain
    # only the explicitly reviewed sources above.  Legacy local files can stay
    # on disk for unrelated experiments, but are not eligible for this index.
    import_root = args.output_dir.parent
    import_manifest_path = import_root / "_import_manifest.json"
    curated_documents = [
        {
            "source": entry["product"],
            "source_file": Path(entry["output_file"]).name,
            "original_path": entry["resolved_url"],
            "imported_path": str((ROOT / entry["output_file"]).relative_to(import_root)).replace("\\", "/"),
            "file_type": ".txt",
            "title": entry["source_id"].replace("_", " "),
            "source_url": entry["resolved_url"],
            "sha256": entry["sha256"],
            "captured_at_utc": entry["captured_at_utc"],
        }
        for entry in entries
    ]
    documents = curated_documents
    per_source = {}
    for document in documents:
        source = document.get("source", "unknown")
        per_source[source] = per_source.get(source, 0) + 1
    import_manifest = {
        "protocol": "devagent_real_official_docs_v7",
        "scope": "Only the five reviewed official-documentation snapshots listed in the v7 source manifest.",
        "imported_docs_dir": str(import_root),
        "total_imported": len(documents),
        "documents_per_source": dict(sorted(per_source.items())),
        "documents": documents,
    }
    import_manifest_path.write_text(
        json.dumps(import_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "entries": len(entries),
        "manifest": str(args.manifest_path),
        "index_import_manifest": str(import_manifest_path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
