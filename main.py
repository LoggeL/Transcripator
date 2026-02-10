import os
import asyncio
import logging
import tempfile
import requests
from typing import Union
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

# Load environment variables from .env file
load_dotenv()

# Telegram bot token
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Constants
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB in bytes
MAX_MESSAGE_LENGTH = 4096
CEREBRAS_API_URL = "https://api.cerebras.ai/v1/chat/completions"
CEREBRAS_MODEL = "gpt-oss-120b"


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a welcome message when the command /start is issued."""
    await update.message.reply_text(
        "Welcome! Send me a voice message or an audio file (up to 25MB), and I'll transcribe, improve, and summarize it for you."
    )


def remove_think_content(message: str) -> str:
    """Remove content inside <think> text </think> tags and keep the rest."""
    return message.split("</think>")[1].strip() if "</think>" in message else message


def split_message(message: str, max_length: int = MAX_MESSAGE_LENGTH) -> list[str]:
    """Split a message into chunks that fit within Telegram's message length limit."""
    return [message[i : i + max_length] for i in range(0, len(message), max_length)]


def transcribe_audio(file_path: str) -> str:
    """Transcribe the audio file using Groq's API via HTTP requests."""
    GROQ_API_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
    api_key = os.getenv("GROQ_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"}

    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f, "application/octet-stream")}
        data = {
            "model": "whisper-large-v3",
            "response_format": "text",
            "temperature": 0.0,
        }
        response = requests.post(GROQ_API_URL, headers=headers, files=files, data=data)

    if response.status_code != 200:
        raise Exception(
            f"Groq API request failed with status {response.status_code}: {response.text}"
        )

    return response.text


def improve_transcription_cerebras(transcription: str) -> str:
    """Improve the transcription using the Cerebras API."""
    api_key = os.getenv("CEREBRAS_API_KEY")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    prompt = f"""
Task: Improve the following transcription
Instructions:
1. Fix any grammatical or spelling errors
2. Improve readability and coherence
3. Maintain the original meaning and context
4. Use appropriate punctuation and formatting
5. Only return the improved text without any additional comments

Original transcription:
{transcription}

Improved transcription:
"""
    payload = {
        "model": CEREBRAS_MODEL,
        "stream": False,
        "top_p": 1,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that improves transcriptions.",
            },
            {"role": "user", "content": prompt},
        ],
    }
    response = requests.post(CEREBRAS_API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(
            f"Cerebras API request failed with status {response.status_code}: {response.text}"
        )
    return response.json()["choices"][0]["message"]["content"]


def generate_summary_cerebras(transcription: str) -> str:
    """Generate a summary of the transcription using the Cerebras API."""
    api_key = os.getenv("CEREBRAS_API_KEY")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    prompt = f"""
Task: Summarize the following transcription
Instructions:
1. Provide a concise summary of the main points
2. Use bullet points for clarity
3. Write from the perspective of the transcript
4. Capture the key ideas and any important details
5. Ensure the summary is coherent and easy to understand
6. Use the same language that is used in the transcript (english, german, spanish, ...).
7. ONLY RETURN THE SUMMARY.

Transcription:
{transcription}

Summary:
"""
    payload = {
        "model": CEREBRAS_MODEL,
        "stream": False,
        "top_p": 1,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that summarizes transcriptions.",
            },
            {"role": "user", "content": prompt},
        ],
    }
    response = requests.post(CEREBRAS_API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(
            f"Cerebras API request failed with status {response.status_code}: {response.text}"
        )
    return response.json()["choices"][0]["message"]["content"]


async def get_file_with_retry(telegram_obj, max_retries=3, base_delay=2):
    """Get a Telegram file with retry logic for transient timeouts."""
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
    """Download a file with retry logic for transient timeouts."""
    for attempt in range(max_retries):
        try:
            await file.download_to_drive(custom_path=custom_path)
            return
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning(f"download_to_drive() attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
            await asyncio.sleep(delay)


async def process_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process the incoming audio file or voice message."""
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
        await update.message.reply_text(
            "The file is too large. Please send an audio file up to 25MB."
        )
        return

    with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
        await download_with_retry(file, custom_path=temp_file.name)
        temp_file_path = temp_file.name

    try:
        await update.message.reply_text(
            "Processing your audio. This may take a moment..."
        )

        original_transcription = transcribe_audio(temp_file_path)
        improved_transcription = improve_transcription_cerebras(original_transcription)
        improved_transcription = remove_think_content(improved_transcription)

        for part in split_message(improved_transcription):
            await update.message.reply_text(part, do_quote=True)

        summary = generate_summary_cerebras(improved_transcription)
        summary = remove_think_content(summary)
        for part in split_message(summary):
            await update.message.reply_text(part, do_quote=True)

    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        await update.message.reply_text(f"An error occurred: {str(e)}")

    finally:
        os.unlink(temp_file_path)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Global error handler that logs errors and notifies the user."""
    logger.error(f"Exception while handling an update: {context.error}")
    if update and hasattr(update, "message") and update.message:
        await update.message.reply_text("Sorry, something went wrong. Please try again.")


def main() -> None:
    """Set up and run the bot."""
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN is not set in the .env file")
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY is not set in the .env file")
    if not os.getenv("CEREBRAS_API_KEY"):
        raise ValueError("CEREBRAS_API_KEY is not set in the .env file")

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
    application.add_handler(
        MessageHandler(filters.VOICE | filters.AUDIO, process_audio)
    )
    application.add_error_handler(error_handler)

    application.run_polling()


if __name__ == "__main__":
    main()
