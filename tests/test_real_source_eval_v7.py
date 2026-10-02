import unittest

from scripts.real_source_eval_v7 import _retrieval_metrics, load_cases
from src.utils import load_config


class RealSourceEvalV7Tests(unittest.TestCase):
    def test_frozen_dataset_has_traceable_case_types_and_unique_ids(self):
        dataset, cases = load_cases()
        self.assertEqual(dataset["dataset_id"], "devagent_real_source_cases_v7")
        self.assertGreaterEqual(len(cases), 7)
        self.assertEqual(len({case["id"] for case in cases}), len(cases))
        self.assertTrue(any(case["source_type"] == "official_documentation" for case in cases))
        self.assertTrue(any(case["source_type"] == "public_issue_minimal_abstraction" for case in cases))
        self.assertTrue(any(case["source_type"] == "manual_negative_control" for case in cases))

    def test_dedicated_evaluation_config_excludes_demo_document_paths(self):
        config = load_config("configs/real_source_eval_v7.yaml")

        self.assertIn("real_source_eval_v7", config["paths"]["index"])
        self.assertIn("real_source_eval_v7", config["paths"]["output"])
        self.assertIn("no_sample_docs", config["paths"]["docs"])
        self.assertIn("no_uploaded_docs", config["web"]["uploaded_docs_dir"])
        self.assertEqual(config["external_docs"]["imported_docs_dir"], "data/docs_imported")

    def test_retrieval_metrics_penalize_late_and_missing_expected_sources(self):
        metrics = _retrieval_metrics([
            {"expected_source": "a.txt", "expected_source_rank": 1},
            {"expected_source": "b.txt", "expected_source_rank": 3},
            {"expected_source": "c.txt", "expected_source_rank": None},
            {"expected_source": None, "expected_source_rank": None},
        ])

        self.assertEqual(metrics["expected_source_cases"], 3)
        self.assertEqual(metrics["hit_at_1"], 1)
        self.assertEqual(metrics["hit_at_3"], 2)
        self.assertEqual(metrics["mrr"], 0.444)


if __name__ == "__main__":
    unittest.main()
