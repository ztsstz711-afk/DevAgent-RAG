import unittest

from src.text_splitter import split_document


class TextSplitterTests(unittest.TestCase):
    def test_splits_headings_and_preserves_code_block(self):
        document = {"content": "# One\nText\n\n```python\nprint(1)\n```\n\n## Two\nMore", "source": "x.md", "product": "demo"}
        chunks = split_document(document, max_chars=1000, overlap=0)
        self.assertGreaterEqual(len(chunks), 3)
        code = [chunk for chunk in chunks if "print(1)" in chunk["content"]][0]
        self.assertTrue(code["content"].startswith("```python"))
        self.assertTrue(code["content"].endswith("```"))
        self.assertEqual(chunks[0]["chunk_id"], "chunk_001")

    def test_splits_long_blocks(self):
        document = {"content": "# Long\n" + ("word " * 100), "source": "x.md", "product": "demo"}
        chunks = split_document(document, max_chars=120, overlap=10)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk["content"]) <= 120 for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
