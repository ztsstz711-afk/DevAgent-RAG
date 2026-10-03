import unittest
from pathlib import Path

from scripts.evaluate_official_retrieval_v8 import load_protocol


class OfficialRetrievalV8Tests(unittest.TestCase):
    def test_frozen_queries_and_qrels_are_separate_and_consistent(self):
        queries, qrels = load_protocol()

        self.assertEqual(queries["dataset_id"], "devagent_official_issue_queries_v8")
        self.assertEqual(qrels["qrels_id"], "devagent_official_issue_qrels_v8")
        self.assertEqual(len(queries["queries"]), 11)
        self.assertEqual(
            {item["id"] for item in queries["queries"]},
            set(qrels["qrels"]),
        )
        self.assertTrue(any(item["origin_type"] == "public_github_issue_abstraction" for item in queries["queries"]))
        self.assertTrue(any(item["should_refuse"] for item in queries["queries"]))

    def test_refusal_controls_have_empty_qrels(self):
        queries, qrels = load_protocol()

        for item in queries["queries"]:
            if item["should_refuse"]:
                self.assertEqual(qrels["qrels"][item["id"]], [])
            else:
                self.assertTrue(qrels["qrels"][item["id"]])

    def test_v9_holdout_protocol_is_separate_and_consistent(self):
        root = Path(__file__).resolve().parents[1]
        queries, qrels = load_protocol(
            root / "data" / "evals" / "official_issue_queries_v9.json",
            root / "data" / "evals" / "official_issue_qrels_v9.json",
        )

        self.assertEqual(queries["split"], "frozen_holdout")
        self.assertEqual(qrels["split"], "frozen_holdout")
        self.assertEqual(len(queries["queries"]), 16)
        self.assertEqual(sum(item["should_refuse"] for item in queries["queries"]), 4)
        self.assertEqual(
            sum(item["origin_type"] == "public_github_issue_abstraction" for item in queries["queries"]),
            10,
        )


if __name__ == "__main__":
    unittest.main()
