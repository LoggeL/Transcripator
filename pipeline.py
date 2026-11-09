"""Shared transcription pipeline: Groq Whisper -> OpenRouter polish.

Used by both the Telegram bot (main.py) and the web version (web.py).
Models are configurable via environment variables; defaults are the
latest approved IDs as of 2026-11:

- GROQ_MODEL (default "whisper-large-v3"): best-accuracy Whisper on Groq.
  "whisper-large-v3-turbo" is cheaper/faster (~12% vs 10.3% WER).
- OPENROUTER_MODEL (default "meta/muse-ember"): Meta's Nov-2026 text
  model for extraction/rewrite tasks, replaces muse-spark-1.3-contributor.
"""

import logging
import os

import requests
from dotenv import load_dotenv

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

load_dotenv()

MAX_MESSAGE_LENGTH = 4096

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta/muse-ember")

GROQ_API_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_MODEL = os.getenv("GROQ_MODEL", "whisper-large-v3")
GROQ_MAX_FILE_SIZE = 24 * 1024 * 1024  # 24MB


def missing_keys() -> list[str]:
    """Return required API keys that are not set."""
    return [k for k in ("OPENROUTER_API_KEY", "GROQ_API_KEY") if not os.getenv(k)]


def split_message(message: str, max_length: int = MAX_MESSAGE_LENGTH) -> list[str]:
    return [message[i : i + max_length] for i in range(0, len(message), max_length)]


def call_openrouter_text(system_prompt: str, user_prompt: str) -> str:
    """Text-only call via OpenRouter (current model: Muse Ember)."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=60)
    if response.status_code != 200:
        raise Exception(
            f"OpenRouter API failed ({response.status_code}): {response.text}"
        )
    return response.json()["choices"][0]["message"]["content"]


def improve_transcription(raw: str) -> str:
    return call_openrouter_text(
        "You are a helpful assistant that improves transcriptions.",
        "Improve this transcription: fix grammar, spelling, punctuation, and "
        f"improve readability. Maintain the original meaning and language. "
        f"Return ONLY the improved text.\n\n{raw}",
    )


def summarize_transcription(improved: str) -> str:
    return call_openrouter_text(
        "You are a helpful assistant that summarizes transcriptions.",
        "Summarize the following transcription using bullet points. "
        "Write from the perspective of the transcript. Use the same language. "
        f"ONLY RETURN THE SUMMARY.\n\nTranscription:\n{improved}",
    )


def transcribe_with_groq(file_path: str) -> str:
    """Transcribe audio using Groq Whisper (see GROQ_MODEL)."""
    import subprocess

    compressed = file_path
    if os.path.getsize(file_path) > GROQ_MAX_FILE_SIZE:
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", file_path],
                capture_output=True, text=True, timeout=30,
            )
            duration = float(result.stdout.strip())
            target_kbps = max(16, min(128, int((GROQ_MAX_FILE_SIZE * 8) / duration) // 1000))
            compressed = file_path + "_groq.ogg"
            subprocess.run(
                ["ffmpeg", "-i", file_path, "-vn", "-c:a", "libopus", "-b:a", f"{target_kbps}k", "-y", compressed],
                capture_output=True, text=True, timeout=300, check=True,
            )
        except Exception as e:
            raise ValueError(f"Audio compression for Groq failed: {e}")

    api_key = os.getenv("GROQ_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        with open(compressed, "rb") as f:
            files = {"file": (os.path.basename(compressed), f, "application/octet-stream")}
            data = {"model": GROQ_MODEL, "response_format": "text", "temperature": 0.0}
            response = requests.post(GROQ_API_URL, headers=headers, files=files, data=data, timeout=120)
    finally:
        if compressed != file_path and os.path.exists(compressed):
            os.unlink(compressed)
    if response.status_code != 200:
        raise Exception(f"Groq Whisper failed ({response.status_code}): {response.text}")
    return response.text.strip()


def process_transcription(file_path: str) -> dict:
    """Run the full pipeline: transcribe, improve, summarize."""
    raw = transcribe_with_groq(file_path)
    improved = improve_transcription(raw)
    summary = summarize_transcription(improved)
    return {"raw": raw, "improved": improved, "summary": summary}
