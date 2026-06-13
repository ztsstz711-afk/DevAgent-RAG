import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_index import build_index
from scripts.prepare_external_docs import prepare_external_docs
from src.document_loader import load_documents


class ExternalDocsTests(unittest.TestCase):
    def test_missing_external_directory_does_not_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = prepare_external_docs(
                external_dir=root / "missing",
                imported_docs_dir=root / "imported",
            )
            self.assertEqual(result["total_imported"], 0)
            self.assertTrue((root / "imported" / "_import_manifest.json").exists())

    def test_imports_supported_files_and_preserves_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "external" / "openai-cookbook"
            source.mkdir(parents=True)
            (source / "guide.md").write_text("# Guide\nMarkdown", encoding="utf-8")
            (source / "page.mdx").write_text("# MDX Page\nContent", encoding="utf-8")
            (source / "notes.txt").write_text("Plain notes", encoding="utf-8")
            notebook = {"cells": [{"cell_type": "markdown", "source": ["# Notebook Title"]}]}
            (source / "demo.ipynb").write_text(json.dumps(notebook), encoding="utf-8")

            imported = root / "imported"
            result = prepare_external_docs(root / "external", imported, max_external_docs_per_source=10)
            self.assertEqual(result["total_imported"], 4)
            documents = load_documents(imported)
            self.assertEqual(len(documents), 4)
            self.assertTrue(all(document["product"] == "openai-cookbook" for document in documents))
            self.assertTrue(all(document["is_imported"] for document in documents))
            self.assertTrue(all(document["original_path"] for document in documents))
            self.assertTrue(all(document["title"] for document in documents))

    def test_skips_oversized_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "external" / "langchain-docs"
            source.mkdir(parents=True)
            (source / "large.md").write_text("x" * 2048, encoding="utf-8")
            result = prepare_external_docs(
                root / "external", root / "imported", max_file_size_kb=1
            )
            self.assertEqual(result["total_imported"], 0)

    def test_build_index_combines_sample_and_imported_docs_and_writes_stats(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sample = root / "sample" / "openai"
            sample.mkdir(parents=True)
            (sample / "sample.md").write_text("# Sample\nRate limits use backoff.", encoding="utf-8")

            external = root / "external" / "vllm-docs"
            external.mkdir(parents=True)
            (external / "server.md").write_text("# vLLM Server\nOpenAI-compatible serving.", encoding="utf-8")
            imported = root / "imported"
            prepare_external_docs(root / "external", imported)

            stats_path = root / "index_stats.json"
            count, index_path = build_index(
                docs_dir=root / "sample",
                imported_docs_dir=imported,
                index_path=root / "index.joblib",
                stats_path=stats_path,
            )
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            self.assertGreaterEqual(count, 2)
            self.assertTrue(index_path.exists())
            expected_fields = {
                "total_documents", "total_chunks", "sources", "documents_per_source",
                "chunks_per_source", "imported_docs_count", "retrieval_mode", "generated_at",
            }
            self.assertTrue(expected_fields.issubset(stats))
            self.assertEqual(stats["total_documents"], 2)
            self.assertEqual(stats["imported_docs_count"], 1)
            self.assertIn("openai", stats["sources"])
            self.assertIn("vllm-docs", stats["sources"])


if __name__ == "__main__":
    unittest.main()
