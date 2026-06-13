import unittest

from src.error_parser import parse_error


class ErrorParserTests(unittest.TestCase):
    def test_known_errors(self):
        cases = {
            "CUDA out of memory": "cuda_oom",
            "ModuleNotFoundError: No module named 'torch'": "module_not_found",
            "OPENAI_API_KEY missing": "openai_api_key_missing",
            "RuntimeError: shape mismatch": "runtime_error",
        }
        for message, expected in cases.items():
            with self.subTest(message=message):
                self.assertEqual(parse_error(message)["primary_type"], expected)

    def test_regular_question_is_not_error(self):
        self.assertFalse(parse_error("How does retrieval work?")["is_error"])


if __name__ == "__main__":
    unittest.main()
