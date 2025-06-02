import unittest
from unittest.mock import patch, mock_open
import os
import main # Assuming your main script is main.py

class TestApiFunctions(unittest.TestCase):
    @patch.dict(os.environ, {"GROQ_API_KEY": "fake_groq_key"})
    @patch('requests.post')
    @patch('builtins.open', new_callable=mock_open, read_data=b"dummy audio data")
    def test_transcribe_audio_success(self, mock_file_open, mock_post):
        mock_response = mock_post.return_value
        mock_response.status_code = 200
        mock_response.text = "This is a test transcription."

        result = main.transcribe_audio("dummy_file_path.ogg")
        self.assertEqual(result, "This is a test transcription.")
        mock_post.assert_called_once()
        # More specific assertions can be added here for the request arguments

    @patch.dict(os.environ, {"GROQ_API_KEY": "fake_groq_key"})
    @patch('requests.post')
    @patch('builtins.open', new_callable=mock_open, read_data=b"dummy audio data")
    def test_transcribe_audio_failure(self, mock_file_open, mock_post):
        mock_response = mock_post.return_value
        mock_response.status_code = 500
        mock_response.text = "Groq API Error"

        with self.assertRaises(Exception) as context:
            main.transcribe_audio("dummy_file_path.ogg")
        self.assertTrue("Groq API request failed" in str(context.exception))

    @patch.dict(os.environ, {"CEREBRAS_API_KEY": "fake_cerebras_key"})
    @patch('requests.post')
    def test_improve_transcription_cerebras_success(self, mock_post):
        mock_response = mock_post.return_value
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "Improved text."}}]}

        result = main.improve_transcription_cerebras("original text")
        self.assertEqual(result, "Improved text.")

        # Assert that mock_post was called and inspect its arguments
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertIn("json", kwargs)
        payload = kwargs["json"]
        self.assertNotIn("max_tokens", payload)
        self.assertEqual(payload["model"], main.CEREBRAS_MODEL) # Example of checking other relevant parts
        self.assertEqual(payload["temperature"], 0.3)

    @patch.dict(os.environ, {"CEREBRAS_API_KEY": "fake_cerebras_key"})
    @patch('requests.post')
    def test_improve_transcription_cerebras_failure(self, mock_post):
        mock_response = mock_post.return_value
        mock_response.status_code = 500
        mock_response.text = "Cerebras API Error"

        with self.assertRaises(Exception) as context:
            main.improve_transcription_cerebras("original text")
        self.assertTrue("Cerebras API request failed" in str(context.exception))

    @patch.dict(os.environ, {"CEREBRAS_API_KEY": "fake_cerebras_key"})
    @patch('requests.post')
    def test_generate_summary_cerebras_success(self, mock_post):
        mock_response = mock_post.return_value
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "Summary text."}}]}

        result = main.generate_summary_cerebras("long text")
        self.assertEqual(result, "Summary text.")

        # Assert that mock_post was called and inspect its arguments
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertIn("json", kwargs)
        payload = kwargs["json"]
        self.assertNotIn("max_tokens", payload)
        self.assertEqual(payload["model"], main.CEREBRAS_MODEL) # Example of checking other relevant parts
        self.assertEqual(payload["temperature"], 0.5)

    @patch.dict(os.environ, {"CEREBRAS_API_KEY": "fake_cerebras_key"})
    @patch('requests.post')
    def test_generate_summary_cerebras_failure(self, mock_post):
        mock_response = mock_post.return_value
        mock_response.status_code = 500
        mock_response.text = "Cerebras API Error"

        with self.assertRaises(Exception) as context:
            main.generate_summary_cerebras("long text")
        self.assertTrue("Cerebras API request failed" in str(context.exception))

if __name__ == '__main__':
    unittest.main()
