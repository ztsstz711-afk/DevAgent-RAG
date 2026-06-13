import unittest
from unittest.mock import patch

from scripts.run_llm_demo import run_llm_demo


class RunLLMDemoTests(unittest.TestCase):
    def test_demo_runs_without_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            path = run_llm_demo()
        content = path.read_text(encoding="utf-8")
        self.assertTrue(path.exists())
        self.assertIn("Answer backend: `template`", content)
        self.assertIn("OPENAI_API_KEY is not set", content)
        self.assertIn("未在当前文档知识库中找到明确依据。", content)


if __name__ == "__main__":
    unittest.main()
