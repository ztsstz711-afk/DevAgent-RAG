import unittest

from src.quality_checker import check_quality


class QualityCheckerTests(unittest.TestCase):
    def test_passes_detailed_cited_answer(self):
        answer = "Use exponential backoff and inspect rate limit headers before retrying requests. [openai | rate_limits.md | chunk_003]"
        self.assertTrue(check_quality(answer)["passed"])

    def test_flags_missing_citation(self):
        result = check_quality("This answer is long enough to explain the issue but it has no source citation attached to it.")
        self.assertFalse(result["passed"])
        self.assertIn("missing_citation", result["issues"])

    def test_flags_no_evidence(self):
        result = check_quality("未在当前文档知识库中找到明确依据。", no_evidence=True)
        self.assertFalse(result["passed"])
        self.assertIn("no_evidence", result["issues"])
        self.assertNotIn("answer_too_short", result["issues"])


if __name__ == "__main__":
    unittest.main()
