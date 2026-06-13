import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.web_utils import (
    clone_github_repo,
    display_task_type,
    load_json_report,
    save_uploaded_file,
    validate_github_url,
)


class WebUtilsTests(unittest.TestCase):
    def test_display_task_type_maps_backend_routes(self):
        self.assertEqual(display_task_type({"route": "question"}, "How does retrieval work?"), "doc_qa")
        self.assertEqual(
            display_task_type({"route": "error", "error_info": {"is_error": True}}, "CUDA out of memory"),
            "error_debug",
        )

    def test_display_task_type_detects_code_and_configuration_queries(self):
        self.assertEqual(display_task_type({"route": "question"}, "Show me a code example"), "code_lookup")
        self.assertEqual(
            display_task_type(
                {"route": "error", "error_info": {"is_error": True, "primary_type": "openai_api_key_missing"}},
                "OPENAI_API_KEY missing",
            ),
            "config_help",
        )

    def test_validate_github_url_only_allows_https_github_repositories(self):
        valid = [
            "https://github.com/openai/openai-cookbook",
            "https://github.com/langchain-ai/docs.git",
        ]
        invalid = [
            "http://github.com/openai/openai-cookbook",
            "https://gitlab.com/openai/openai-cookbook",
            "git@github.com:openai/openai-cookbook.git",
            "https://github.com/openai",
            "https://github.com/openai/repo/extra",
            "https://github.com/openai/repo?x=1",
            "https://github.com.evil.example/openai/repo",
            "https://github.com/../repo",
            "https://github.com/openai/.git",
        ]
        self.assertTrue(all(validate_github_url(url) for url in valid))
        self.assertTrue(all(not validate_github_url(url) for url in invalid))

    def test_save_uploaded_file(self):
        upload = Mock()
        upload.name = "guide.md"
        upload.getvalue.return_value = b"# Guide\nContent"
        with tempfile.TemporaryDirectory() as temp:
            path = save_uploaded_file(upload, temp)
            self.assertEqual(path.name, "guide.md")
            self.assertEqual(path.read_text(encoding="utf-8"), "# Guide\nContent")

    def test_save_uploaded_file_strips_path_components(self):
        upload = Mock()
        upload.name = "../safe.txt"
        upload.getvalue.return_value = b"safe"
        with tempfile.TemporaryDirectory() as temp:
            path = save_uploaded_file(upload, temp)
            self.assertEqual(path.parent, Path(temp))
            self.assertEqual(path.name, "safe.txt")

    def test_load_json_report_missing_returns_empty_dict(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(load_json_report(Path(temp) / "missing.json"), {})

    def test_load_json_report_reads_payload(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "report.json"
            path.write_text(json.dumps({"ok": True}), encoding="utf-8")
            self.assertEqual(load_json_report(path), {"ok": True})

    def test_clone_rejects_invalid_url_without_running_git(self):
        with patch("src.web_utils.subprocess.run") as run:
            with self.assertRaises(ValueError):
                clone_github_repo("https://example.com/org/repo")
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
