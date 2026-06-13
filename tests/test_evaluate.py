import json
import unittest

from scripts.evaluate import CASES, evaluate
from src.utils import project_path


class EvaluateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = evaluate()

    def test_eval_has_at_least_fifteen_cases_and_new_metrics(self):
        self.assertGreaterEqual(len(CASES), 15)
        expected = {
            "total_cases", "route_accuracy", "retrieval_hit_rate", "citation_rate",
            "quality_pass_rate", "no_evidence_count", "unsupported_query_count",
            "tool_success_rate", "hit_at_1", "hit_at_3",
        }
        self.assertTrue(expected.issubset(self.report))

    def test_json_report_contains_new_metrics(self):
        path = project_path("data/output/eval_report.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["total_cases"], len(CASES))
        self.assertIn("no_evidence_count", payload)
        self.assertIn("hit_at_3", payload)

    def test_markdown_report_is_generated(self):
        path = project_path("data/output/eval_report.md")
        self.assertTrue(path.exists())
        self.assertIn("# DevAgent-RAG Evaluation Report", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
