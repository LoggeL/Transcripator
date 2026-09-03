import os
import asyncio
import logging
import tempfile
import base64
import mimetypes
import requests
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from telegram.request import HTTPXRequest
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Constants
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB
MAX_MESSAGE_LENGTH = 4096
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "meta/muse-spark-1.3-contributor"
GROQ_API_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_MAX_FILE_SIZE = 24 * 1024 * 1024  # 24MB

# Audio MIME type mapping
AUDIO_FORMATS = {
    ".ogg": "audio/ogg",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".mp4": "audio/mp4",
    ".aac": "audio/aac",
    ".flac": "audio/flac",
    ".webm": "audio/webm",
}


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Welcome! Send me a voice message or an audio file (up to 25MB), "
        "and I'll transcribe, improve, and summarize it for you."
    )


def split_message(message: str, max_length: int = MAX_MESSAGE_LENGTH) -> list[str]:
    return [message[i : i + max_length] for i in range(0, len(message), max_length)]


def call_gemini_with_audio(file_path: str, prompt: str) -> str:
    """Send audio directly to Gemini Flash Lite via OpenRouter for processing."""
    api_key = os.getenv("OPENROUTER_API_KEY")

    # Read and base64-encode the audio file
    with open(file_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")

    # Determine MIME type
    ext = os.path.splitext(file_path)[1].lower()
    mime_type = AUDIO_FORMATS.get(ext, "audio/ogg")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_b64,
                            "format": ext.lstrip("."),
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
    }
    response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=120)
    if response.status_code != 200:
        raise Exception(
            f"OpenRouter API failed ({response.status_code}): {response.text}"
        )
    return response.json()["choices"][0]["message"]["content"]


def call_gemini_text(system_prompt: str, user_prompt: str) -> str:
    """Text-only call to Gemini Flash Lite via OpenRouter."""
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


def transcribe_with_groq(file_path: str) -> str:
    """Transcribe audio using Groq Whisper large-v3."""
    import subprocess
    # Compress if needed
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
            data = {"model": "whisper-large-v3", "response_format": "text", "temperature": 0.0}
            response = requests.post(GROQ_API_URL, headers=headers, files=files, data=data, timeout=120)
    finally:
        if compressed != file_path and os.path.exists(compressed):
            os.unlink(compressed)
    if response.status_code != 200:
        raise Exception(f"Groq Whisper failed ({response.status_code}): {response.text}")
    return response.text.strip()


async def get_file_with_retry(telegram_obj, max_retries=3, base_delay=2):
    for attempt in range(max_retries):
        try:
            return await telegram_obj.get_file()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning(f"get_file() attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
            await asyncio.sleep(delay)


async def download_with_retry(file, custom_path, max_retries=3, base_delay=2):
    for attempt in range(max_retries):
        try:
            await file.download_to_drive(custom_path=custom_path)
            return
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning(f"download attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
            await asyncio.sleep(delay)


async def process_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    file = None
    file_extension = ""

    if update.message.voice:
        file = await get_file_with_retry(update.message.voice)
        file_extension = ".ogg"
    elif update.message.audio:
        file = await get_file_with_retry(update.message.audio)
        file_extension = os.path.splitext(update.message.audio.file_name)[1]

    if not file:
        await update.message.reply_text("Please send a voice message or an audio file.")
        return

    if file.file_size > MAX_FILE_SIZE:
        await update.message.reply_text("The file is too large. Please send an audio file up to 25MB.")
        return

    with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
        await download_with_retry(file, custom_path=temp_file.name)
        temp_file_path = temp_file.name

    try:
        await update.message.reply_text("Processing your audio. This may take a moment...")

        # Step 1: Transcribe via Groq Whisper
        raw_transcription = transcribe_with_groq(temp_file_path)

        # Step 2: Improve via Gemini (text-only, cheaper)
        improved = call_gemini_text(
            "You are a helpful assistant that improves transcriptions.",
            f"Improve this transcription: fix grammar, spelling, punctuation, and improve readability. "
            f"Maintain the original meaning and language. Return ONLY the improved text.\n\n{raw_transcription}",
        )

        for part in split_message(improved):
            await update.message.reply_text(part, do_quote=True)

        # Step 3: Summarize
        summary = call_gemini_text(
            "You are a helpful assistant that summarizes transcriptions.",
            f"Summarize the following transcription using bullet points. "
            f"Write from the perspective of the transcript. Use the same language. "
            f"ONLY RETURN THE SUMMARY.\n\nTranscription:\n{improved}",
        )

        for part in split_message(summary):
            await update.message.reply_text(part, do_quote=True)

    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        await update.message.reply_text(f"An error occurred: {str(e)}")
    finally:
        os.unlink(temp_file_path)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(f"Exception while handling an update: {context.error}")
    if update and hasattr(update, "message") and update.message:
        await update.message.reply_text("Sorry, something went wrong. Please try again.")


def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN is not set")
    if not os.getenv("OPENROUTER_API_KEY"):
        raise ValueError("OPENROUTER_API_KEY is not set")
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY is not set")

    request = HTTPXRequest(
        read_timeout=60, write_timeout=60, connect_timeout=30, pool_timeout=30
    )
    application = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .request(request)
        .get_updates_request(HTTPXRequest(read_timeout=60))
        .build()
    )

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, process_audio))
    application.add_error_handler(error_handler)

    application.run_polling()


if __name__ == "__main__":
    main()
