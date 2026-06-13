import io
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

from src.answer_generator import NO_EVIDENCE_ANSWER
from src.llm_answer import LLMAnswerGenerator


class LLMAnswerGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.evidence = [{
            "content": "Use exponential backoff for rate limits.",
            "source": "rate_limits.md",
            "product": "openai",
            "chunk_id": "chunk_001",
        }]

    def test_missing_api_key_falls_back_to_template(self):
        with patch.dict("os.environ", {}, clear=True):
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                result = LLMAnswerGenerator(mode="openai").generate("How to handle rate limits?", self.evidence)
        self.assertEqual(result.backend, "template")
        self.assertIn("OPENAI_API_KEY is not set", result.fallback_reason)
        self.assertIn("falling back", stderr.getvalue())
        self.assertIn("[openai | rate_limits.md | chunk_001]", result.answer)

    def test_prompt_contains_evidence_and_citation_rules(self):
        prompt = LLMAnswerGenerator().build_prompt("Question", self.evidence)
        self.assertIn("EVIDENCE", prompt)
        self.assertIn("Every key conclusion must include", prompt)
        self.assertIn("[openai | rate_limits.md | chunk_001]", prompt)
        self.assertIn("## Answer, ## Evidence, ## Next Steps", prompt)

    def test_no_evidence_refuses_without_api_call(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True):
            generator = LLMAnswerGenerator(mode="openai")
            with patch.object(generator, "_call_openai") as api_call:
                result = generator.generate("How to deploy Kubernetes on AWS?", [])
        api_call.assert_not_called()
        self.assertEqual(result.answer, NO_EVIDENCE_ANSWER)
        self.assertEqual(result.fallback_reason, "no_evidence")


if __name__ == "__main__":
    unittest.main()
