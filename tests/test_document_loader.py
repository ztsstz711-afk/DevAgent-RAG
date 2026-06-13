import json
import tempfile
import unittest
from pathlib import Path

from src.document_loader import load_document, load_documents


class DocumentLoaderTests(unittest.TestCase):
    def test_loads_supported_text_formats(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "openai"
            root.mkdir()
            for name in ("a.md", "b.mdx", "c.txt"):
                (root / name).write_text("# Title\nBody", encoding="utf-8")
            docs = load_documents(Path(temp))
            self.assertEqual(len(docs), 3)
            self.assertTrue(all(doc["product"] == "openai" for doc in docs))

    def test_loads_notebook_markdown_and_code(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "test.ipynb"
            path.write_text(json.dumps({"cells": [
                {"cell_type": "markdown", "source": ["# Notes"]},
                {"cell_type": "code", "source": ["print('ok')"]},
            ]}), encoding="utf-8")
            doc = load_document(path)
            self.assertIn("# Notes", doc["content"])
            self.assertIn("```python", doc["content"])

    def test_rejects_unsupported_format(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "x.pdf"
            path.write_text("x", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_document(path)


if __name__ == "__main__":
    unittest.main()
