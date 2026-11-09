"""Tests for pipeline.py (Groq Whisper -> OpenRouter polish).

All network calls are mocked; no API keys needed.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

import pipeline


def _tmp_audio(content: bytes = b"dummy audio data"):
    f = tempfile.NamedTemporaryFile(suffix=".ogg", delete=False)
    f.write(content)
    f.close()
    return f.name


def _cleanup(path):
    if path and os.path.exists(path):
        os.unlink(path)


class TestTranscribeWithGroq(unittest.TestCase):
    @patch.dict(os.environ, {"GROQ_API_KEY": "fake_groq_key"})
    @patch("pipeline.requests.post")
    def test_success(self, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.text = "This is a test transcription."
        path = _tmp_audio()
        try:
            result = pipeline.transcribe_with_groq(path)
        finally:
            _cleanup(path)
        self.assertEqual(result, "This is a test transcription.")
        _, kwargs = mock_post.call_args
        self.assertEqual(kwargs["data"]["model"], pipeline.GROQ_MODEL)
        self.assertEqual(kwargs["data"]["response_format"], "text")
        self.assertIn("file", kwargs["files"])

    @patch.dict(os.environ, {"GROQ_API_KEY": "fake_groq_key"})
    @patch("pipeline.requests.post")
    def test_failure(self, mock_post):
        mock_post.return_value.status_code = 500
        mock_post.return_value.text = "Groq API Error"
        path = _tmp_audio()
        try:
            with self.assertRaises(Exception) as cm:
                pipeline.transcribe_with_groq(path)
        finally:
            _cleanup(path)
        self.assertIn("Groq Whisper failed", str(cm.exception))

    def test_groq_model_default_is_best_accuracy(self):
        self.assertEqual(pipeline.GROQ_MODEL, "whisper-large-v3")


class TestCallOpenrouterText(unittest.TestCase):
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "fake_key"})
    @patch("pipeline.requests.post")
    def test_success(self, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": "Polished text."}}]
        }
        result = pipeline.call_openrouter_text("sys", "user")
        self.assertEqual(result, "Polished text.")
        _, kwargs = mock_post.call_args
        payload = kwargs["json"]
        self.assertEqual(payload["model"], pipeline.OPENROUTER_MODEL)
        self.assertNotIn("max_tokens", payload)
        self.assertEqual([m["role"] for m in payload["messages"]], ["system", "user"])

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "fake_key"})
    @patch("pipeline.requests.post")
    def test_failure(self, mock_post):
        mock_post.return_value.status_code = 500
        mock_post.return_value.text = "OpenRouter Error"
        with self.assertRaises(Exception) as cm:
            pipeline.call_openrouter_text("sys", "user")
        self.assertIn("OpenRouter API failed", str(cm.exception))

    def test_openrouter_model_default(self):
        self.assertEqual(pipeline.OPENROUTER_MODEL, "meta/muse-ember")


class TestProcessTranscription(unittest.TestCase):
    @patch("pipeline.call_openrouter_text")
    @patch("pipeline.transcribe_with_groq")
    def test_orchestrates_steps(self, mock_transcribe, mock_llm):
        mock_transcribe.return_value = "raw text"
        mock_llm.side_effect = ["improved text", "summary text"]
        result = pipeline.process_transcription("dummy.ogg")
        self.assertEqual(
            result,
            {"raw": "raw text", "improved": "improved text", "summary": "summary text"},
        )
        mock_transcribe.assert_called_once_with("dummy.ogg")
        self.assertEqual(mock_llm.call_count, 2)
        second_user_prompt = mock_llm.call_args_list[1].args[1]
        self.assertIn("improved text", second_user_prompt)


class TestSplitMessage(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(pipeline.split_message(""), [])

    def test_short(self):
        self.assertEqual(pipeline.split_message("hello"), ["hello"])

    def test_exact_fit(self):
        msg = "x" * 4096
        self.assertEqual(pipeline.split_message(msg), [msg])

    def test_over_limit(self):
        msg = "x" * 5000
        parts = pipeline.split_message(msg)
        self.assertEqual(len(parts), 2)
        self.assertEqual("".join(parts), msg)


if __name__ == "__main__":
    unittest.main()
