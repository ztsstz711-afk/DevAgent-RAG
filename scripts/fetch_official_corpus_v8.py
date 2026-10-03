"""Fetch the declared versioned official-document corpus for the v8 experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_REGISTER = ROOT / "data" / "corpus" / "official_sources_v8.json"


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


def fetch_text(url: str) -> tuple[str, str]:
    request = Request(url, headers={"User-Agent": "DevAgent-RAG/0.2 local research"})
    with urlopen(request, timeout=45) as response:
        raw = response.read()
        resolved_url = response.geturl()
    parser = TextExtractor()
    parser.feed(raw.decode("utf-8", errors="replace"))
    text = "\n".join(parser.parts)
    if len(text) < 500:
        raise ValueError(f"Fetched content from {resolved_url} is unexpectedly short")
    return resolved_url, text


def fetch_corpus(source_register: Path = DEFAULT_SOURCE_REGISTER, output_dir: Path | None = None) -> dict:
    register = json.loads(source_register.read_text(encoding="utf-8"))
    sources = register.get("sources", [])
    if not sources:
        raise ValueError("Source register has no sources")
    output_dir = output_dir or ROOT / "data" / "docs_imported" / "official_v8"
    output_dir.mkdir(parents=True, exist_ok=True)
    captured_at = datetime.now(UTC).isoformat()
    entries = []
    imported_documents = []
    for source in sources:
        required = {"source_id", "product", "version", "url", "license", "license_url"}
        missing = sorted(required - set(source))
        if missing:
            raise ValueError(f"Source {source.get('source_id', '<unknown>')} missing {', '.join(missing)}")
        resolved_url, text = fetch_text(source["url"])
        filename = f"{source['source_id']}.txt"
        output = output_dir / filename
        output.write_text(text + "\n", encoding="utf-8")
        sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
        entries.append({
            **source,
            "source_type": "official_documentation",
            "requested_url": source["url"],
            "resolved_url": resolved_url,
            "captured_at_utc": captured_at,
            "output_file": str(output.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256,
            "characters": len(text),
            "usage": "local retrieval and frozen evaluation only; no model training",
        })
        imported_documents.append({
            "source": source["product"],
            "source_file": filename,
            "original_path": resolved_url,
            "imported_path": filename,
            "file_type": ".txt",
            "title": source["source_id"].replace("_", " "),
            "source_url": resolved_url,
            "source_id": source["source_id"],
            "version": source["version"],
            "license": source["license"],
            "license_url": source["license_url"],
            "sha256": sha256,
            "captured_at_utc": captured_at,
        })
    manifest = {
        "protocol": register["protocol"],
        "scope": register["purpose"],
        "source_register": str(source_register.relative_to(ROOT)).replace("\\", "/"),
        "entries": entries,
    }
    manifest_path = output_dir / "corpus_manifest_v8.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    import_manifest = {
        "protocol": register["protocol"],
        "scope": "Only documents declared in corpus_manifest_v8.json are eligible for the v8 retrieval index.",
        "imported_docs_dir": str(output_dir.relative_to(ROOT)).replace("\\", "/"),
        "total_imported": len(imported_documents),
        "documents_per_source": {
            product: sum(doc["source"] == product for doc in imported_documents)
            for product in sorted({doc["source"] for doc in imported_documents})
        },
        "documents": imported_documents,
    }
    (output_dir / "_import_manifest.json").write_text(
        json.dumps(import_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {"entries": len(entries), "manifest": str(manifest_path), "output_dir": str(output_dir)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-register", type=Path, default=DEFAULT_SOURCE_REGISTER)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(fetch_corpus(args.source_register, args.output_dir), ensure_ascii=False))
