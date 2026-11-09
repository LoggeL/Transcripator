"""Tests for pipeline.py (Ember direct audio + Groq fallback).

All network calls are mocked; no API keys needed.
"""

import base64
import os
import tempfile
import unittest
from unittest.mock import patch

import pipeline


def _tmp_file(suffix: str, content: bytes = b"dummy audio data") -> str:
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    f.write(content)
    f.close()
    return f.name


def _cleanup(path):
    if path and os.path.exists(path):
        os.unlink(path)


class TestResponsesText(unittest.TestCase):
    def test_output_text_shortcut(self):
        self.assertEqual(
            pipeline._responses_text({"output_text": " hello "}), "hello"
        )

    def test_walk_output_items(self):
        data = {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "hi"}],
                }
            ]
        }
        self.assertEqual(pipeline._responses_text(data), "hi")

    def test_missing_text_raises(self):
        with self.assertRaises(Exception) as cm:
            pipeline._responses_text({"output": []})
        self.assertIn("no text in response", str(cm.exception))


class TestPrepareAudio(unittest.TestCase):
    @patch("pipeline.subprocess.run")
    def test_wav_passthrough(self, mock_run):
        path = _tmp_file(".wav")
        try:
            out, fmt = pipeline._prepare_audio(path)
        finally:
            _cleanup(path)
        self.assertEqual((out, fmt), (path, "wav"))
        mock_run.assert_not_called()

    @patch("pipeline.subprocess.run")
    def test_mp3_passthrough(self, mock_run):
        path = _tmp_file(".mp3")
        try:
            out, fmt = pipeline._prepare_audio(path)
        finally:
            _cleanup(path)
        self.assertEqual((out, fmt), (path, "mp3"))
        mock_run.assert_not_called()

    @patch("pipeline.subprocess.run")
    def test_ogg_converted_to_mp3(self, mock_run):
        def fake_run(*args, **kwargs):
            # last argv element is the converted output path
            out_path = args[0][-1]
            with open(out_path, "wb") as f:
                f.write(b"converted")

        mock_run.side_effect = fake_run
        path = _tmp_file(".ogg")
        try:
            out, fmt = pipeline._prepare_audio(path)
        finally:
            _cleanup(path)
        self.assertTrue(out.endswith(".mp3"))
        self.assertEqual(fmt, "mp3")
        self.assertTrue(os.path.exists(out))
        self.assertEqual(mock_run.call_args.args[0][0], "ffmpeg")
        _cleanup(out)

    @patch("pipeline.subprocess.run", side_effect=Exception("no ffmpeg"))
    def test_conversion_failure_raises(self, mock_run):
        path = _tmp_file(".ogg")
        try:
            with self.assertRaises(Exception) as cm:
                pipeline._prepare_audio(path)
        finally:
            _cleanup(path)
        self.assertIn("Audio conversion for Ember failed", str(cm.exception))


class TestTranscribeWithEmber(unittest.TestCase):
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "fake_key"})
    @patch("pipeline.requests.post")
    def test_wav_direct(self, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"output_text": " transcript "}
        content = b"RIFFfakeaudio"
        path = _tmp_file(".wav", content)
        try:
            result = pipeline.transcribe_with_ember(path)
        finally:
            _cleanup(path)
        self.assertEqual(result, "transcript")
        _, kwargs = mock_post.call_args
        payload = kwargs["json"]
        self.assertEqual(payload["model"], pipeline.OPENROUTER_MODEL)
        content_parts = payload["input"][0]["content"]
        audio_part = [p for p in content_parts if p["type"] == "input_audio"][0]
        self.assertEqual(audio_part["input_audio"]["format"], "wav")
        self.assertEqual(
            audio_part["input_audio"]["data"],
            base64.b64encode(content).decode("utf-8"),
        )

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "fake_key"})
    @patch("pipeline.subprocess.run")
    @patch("pipeline.requests.post")
    def test_ogg_converted_and_sent_as_mp3(self, mock_post, mock_run):
        def fake_run(cmd, **kwargs):
            with open(cmd[-1], "wb") as f:
                f.write(b"converted mp3 bytes")

        mock_run.side_effect = fake_run
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"output_text": "ogg transcript"}
        path = _tmp_file(".ogg", b"ogg bytes")
        try:
            result = pipeline.transcribe_with_ember(path)
        finally:
            _cleanup(path)
        self.assertEqual(result, "ogg transcript")
        _, kwargs = mock_post.call_args
        content_parts = kwargs["json"]["input"][0]["content"]
        audio_part = [p for p in content_parts if p["type"] == "input_audio"][0]
        self.assertEqual(audio_part["input_audio"]["format"], "mp3")
        self.assertEqual(
            audio_part["input_audio"]["data"],
            base64.b64encode(b"converted mp3 bytes").decode("utf-8"),
        )
        # converted temp file must be cleaned up
        self.assertFalse(os.path.exists(path + "_ember.mp3"))

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "fake_key"})
    @patch("pipeline.requests.post")
    def test_api_error(self, mock_post):
        mock_post.return_value.status_code = 500
        mock_post.return_value.text = "boom"
        path = _tmp_file(".wav")
        try:
            with self.assertRaises(Exception) as cm:
                pipeline.transcribe_with_ember(path)
        finally:
            _cleanup(path)
        self.assertIn("OpenRouter Responses API failed", str(cm.exception))


class TestTranscribeAudioFallback(unittest.TestCase):
    @patch("pipeline.transcribe_with_ember", return_value="ember ok")
    def test_ember_primary(self, mock_ember):
        # without GROQ key: ember result returned, no groq needed
        env = {k: v for k, v in os.environ.items() if k != "GROQ_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(pipeline.transcribe_audio("x.wav"), "ember ok")

    @patch("pipeline.transcribe_with_groq", return_value="groq fallback")
    @patch("pipeline.transcribe_with_ember", side_effect=Exception("ember down"))
    def test_fallback_when_groq_set(self, mock_ember, mock_groq):
        with patch.dict(os.environ, {"GROQ_API_KEY": "fake_groq"}):
            self.assertEqual(pipeline.transcribe_audio("x.wav"), "groq fallback")

    @patch("pipeline.transcribe_with_ember", side_effect=Exception("ember down"))
    def test_no_fallback_without_groq(self, mock_ember):
        env = {k: v for k, v in os.environ.items() if k != "GROQ_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(Exception) as cm:
                pipeline.transcribe_audio("x.wav")
        self.assertIn("ember down", str(cm.exception))


class TestMissingKeys(unittest.TestCase):
    def test_only_openrouter_required(self):
        env = {k: v for k, v in os.environ.items() if k != "GROQ_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            # Groq absence must NOT be reported as missing
            self.assertEqual(pipeline.missing_keys(), ["OPENROUTER_API_KEY"])

    def test_none_missing(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "x"}):
            self.assertEqual(pipeline.missing_keys(), [])


class TestTranscribeWithGroq(unittest.TestCase):
    @patch.dict(os.environ, {"GROQ_API_KEY": "fake_groq_key"})
    @patch("pipeline.requests.post")
    def test_success(self, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.text = "This is a test transcription."
        path = _tmp_file(".ogg")
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
        path = _tmp_file(".ogg")
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
    @patch("pipeline.transcribe_audio")
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
