import unittest

from src.evidence_gate import assess_evidence


class EvidenceGateTests(unittest.TestCase):
    def test_external_readme_weak_match_does_not_answer_kubernetes(self):
        documents = [{
            "content": "Deploy LLaMAFactory with Docker and expose an OpenAI-style API endpoint.",
            "source": "README.md",
            "product": "llamafactory",
            "chunk_id": "chunk_087",
            "score": 0.24,
        }]
        result = assess_evidence("How to deploy Kubernetes on AWS?", documents, min_score=0.05)
        self.assertFalse(result["valid"])
        self.assertIn("out_of_domain", result["issues"])

    def test_supported_ai_queries_pass(self):
        lora = assess_evidence(
            "LLaMAFactory LoRA SFT dataset format",
            [{"content": "LoRA SFT dataset format uses instruction input output fields", "product": "llamafactory", "source": "dataset.md", "score": 0.4}],
        )
        cuda = assess_evidence(
            "CUDA out of memory",
            [{"content": "CUDA out of memory reduce batch size", "product": "pytorch", "source": "cuda.md", "score": 0.5}],
        )
        self.assertTrue(lora["valid"])
        self.assertTrue(cuda["valid"])

    def test_unrelated_finance_and_geography_are_rejected(self):
        weak = [{"content": "Model pricing and regional deployment", "product": "openai", "source": "readme.md", "score": 0.2}]
        self.assertIn("out_of_domain", assess_evidence("How to price an insurance product?", weak)["issues"])
        self.assertIn("out_of_domain", assess_evidence("What is the capital of France?", weak)["issues"])

    def test_generic_document_cannot_support_a_guaranteed_incident_fix(self):
        documents = [{
            "content": "Use DataLoader with configurable workers and batch sizes.",
            "product": "pytorch",
            "source": "dataloader.md",
            "score": 0.4,
        }]
        result = assess_evidence(
            "Can you guarantee a fix for a Windows DataLoader memory issue?", documents
        )
        self.assertFalse(result["valid"])
        self.assertIn("unsupported_assurance_claim", result["issues"])

    def test_generic_document_cannot_support_a_promised_incident_workaround(self):
        documents = [{
            "content": "Use DataLoader with configurable workers and batch sizes.",
            "product": "pytorch",
            "source": "dataloader.md",
            "score": 0.4,
        }]
        result = assess_evidence(
            "Can you promise a compatible workaround for this DataLoader incident?", documents
        )
        self.assertFalse(result["valid"])
        self.assertIn("unsupported_assurance_claim", result["issues"])

    def test_version_specific_fix_without_a_version_is_rejected(self):
        documents = [{
            "content": "A retriever returns documents relevant to a query.",
            "product": "langchain",
            "source": "retrieval.md",
            "score": 0.4,
        }]
        result = assess_evidence(
            "Can you give a version-specific LangChain retriever fix command?", documents
        )
        self.assertFalse(result["valid"])
        self.assertIn("missing_version_constraint", result["issues"])


if __name__ == "__main__":
    unittest.main()
