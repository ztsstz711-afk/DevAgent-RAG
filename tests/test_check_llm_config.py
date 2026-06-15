import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from scripts.check_llm_config import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    check_llm_config,
    get_llm_config,
)


class CheckLLMConfigTests(unittest.TestCase):
    def test_no_key_prints_template_fallback(self):
        output = io.StringIO()
        with redirect_stdout(output):
            config = check_llm_config({})
        self.assertFalse(config["api_key_configured"])
        self.assertIn("fall back to template", output.getvalue())

    def test_configured_key_is_masked_and_not_leaked(self):
        key = "sk-example-secret-abcd"
        output = io.StringIO()
        with redirect_stdout(output):
            config = check_llm_config({"OPENAI_API_KEY": key})
        rendered = output.getvalue()
        self.assertTrue(config["api_key_configured"])
        self.assertNotIn(key, rendered)
        self.assertIn("sk-***abcd", rendered)
        self.assertIn("python scripts/run_llm_demo.py", rendered)

    def test_default_base_url_and_model_are_displayed(self):
        config = get_llm_config({})
        self.assertEqual(config["base_url"], DEFAULT_BASE_URL)
        self.assertEqual(config["model"], DEFAULT_MODEL)
        output = io.StringIO()
        with redirect_stdout(output):
            check_llm_config({})
        self.assertIn(DEFAULT_BASE_URL, output.getvalue())
        self.assertIn(DEFAULT_MODEL, output.getvalue())

    def test_check_does_not_make_network_request(self):
        with patch("urllib.request.urlopen") as request:
            check_llm_config({"OPENAI_API_KEY": "sk-test-abcd"})
        request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
