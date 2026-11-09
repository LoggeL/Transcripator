"""Shared transcription pipeline (Ember direct audio -> polish).

Primary path: Ember hears the audio directly via OpenRouter's Responses API
(input_audio, base64). No Groq needed. If Ember fails and GROQ_API_KEY is
set, Groq Whisper is used as fallback.

Models (overridable via env):
- OPENROUTER_MODEL (default "meta/muse-ember"): audio + text.
- GROQ_MODEL (default "whisper-large-v3"): fallback only.
"""

import base64
import logging
import os
import subprocess

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
OPENROUTER_RESPONSES_URL = "https://openrouter.ai/api/v1/responses"
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta/muse-ember")

GROQ_API_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_MODEL = os.getenv("GROQ_MODEL", "whisper-large-v3")
GROQ_MAX_FILE_SIZE = 24 * 1024 * 1024  # 24MB


def missing_keys() -> list[str]:
    """Return required API keys that are not set.

    Only OpenRouter is required now (Ember does audio directly).
    Groq is optional (fallback only).
    """
    return [k for k in ("OPENROUTER_API_KEY",) if not os.getenv(k)]


def has_groq() -> bool:
    return bool(os.getenv("GROQ_API_KEY"))


def split_message(message: str, max_length: int = MAX_MESSAGE_LENGTH) -> list[str]:
    return [message[i : i + max_length] for i in range(0, len(message), max_length)]


def _responses_text(data: dict) -> str:
    """Extract transcript text from a Responses API payload."""
    if data.get("output_text"):
        return str(data["output_text"]).strip()
    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                return str(content["text"]).strip()
    raise Exception(f"OpenRouter Responses API: no text in response: {data!r}")


def _prepare_audio(file_path: str) -> tuple[str, str]:
    """Return (path, format) suitable for the Responses API (wav/mp3 only).

    Ogg/other uploads are converted to 16 kHz mono MP3 via ffmpeg.
    """
    ext = os.path.splitext(file_path)[1].lower().lstrip(".")
    if ext in ("wav", "mp3"):
        return file_path, ext
    converted = file_path + "_ember.mp3"
    try:
        subprocess.run(
            ["ffmpeg", "-i", file_path, "-vn", "-ac", "1", "-ar", "16000",
             "-codec:a", "libmp3lame", "-b:a", "64k", "-y", converted],
            capture_output=True, text=True, timeout=300, check=True,
        )
    except Exception as e:
        if os.path.exists(converted):
            os.unlink(converted)
        raise Exception(f"Audio conversion for Ember failed (ffmpeg?): {e}")
    return converted, "mp3"


def transcribe_with_ember(file_path: str) -> str:
    """Transcribe audio by sending it directly to Ember (OpenRouter)."""
    audio_path = file_path
    fmt = "wav"
    try:
        audio_path, fmt = _prepare_audio(file_path)
        with open(audio_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")
    finally:
        if audio_path != file_path and os.path.exists(audio_path):
            os.unlink(audio_path)

    api_key = os.getenv("OPENROUTER_API_KEY")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Transcribe this audio exactly as spoken. "
                        "Return ONLY the transcript, no commentary.",
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_b64, "format": fmt},
                    },
                ],
            }
        ],
    }
    response = requests.post(
        OPENROUTER_RESPONSES_URL, headers=headers, json=payload, timeout=180
    )
    if response.status_code != 200:
        raise Exception(
            f"OpenRouter Responses API failed ({response.status_code}): {response.text}"
        )
    try:
        return _responses_text(response.json())
    except Exception as e:
        if "no text in response" in str(e):
            raise
        raise Exception(f"OpenRouter Responses API bad response: {e}")


def transcribe_audio(file_path: str) -> str:
    """Ember primary, Groq fallback (only if GROQ_API_KEY is set)."""
    try:
        return transcribe_with_ember(file_path)
    except Exception as e:
        if not has_groq():
            raise
        logger.warning(f"Ember transcription failed ({e}); falling back to Groq Whisper.")
        return transcribe_with_groq(file_path)


def call_openrouter_text(system_prompt: str, user_prompt: str) -> str:
    """Text-only call via OpenRouter Chat Completions (Ember)."""
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
    """Transcribe audio using Groq Whisper (fallback only)."""
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
    raw = transcribe_audio(file_path)
    improved = improve_transcription(raw)
    summary = summarize_transcription(improved)
    return {"raw": raw, "improved": improved, "summary": summary}
