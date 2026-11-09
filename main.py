"""Telegram audio transcription bot (Transcripator).

Pipeline lives in pipeline.py (shared with the web version, web.py):
Ember transcribes audio directly, then improves + summarizes it.
"""

import asyncio
import logging
import os
import tempfile

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from telegram.request import HTTPXRequest

from pipeline import missing_keys, process_transcription, split_message

logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Constants
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Welcome! Send me a voice message or an audio file (up to 25MB), "
        "and I'll transcribe, improve, and summarize it for you."
    )


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
        file_extension = os.path.splitext(update.message.audio.file_name or "")[1]

    if not file:
        await update.message.reply_text("Please send a voice message or an audio file.")
        return

    if file.file_size and file.file_size > MAX_FILE_SIZE:
        await update.message.reply_text("The file is too large. Please send an audio file up to 25MB.")
        return

    with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
        await download_with_retry(file, custom_path=temp_file.name)
        temp_file_path = temp_file.name

    try:
        await update.message.reply_text("Processing your audio. This may take a moment...")

        result = process_transcription(temp_file_path)

        for part in split_message(result["improved"]):
            await update.message.reply_text(part, do_quote=True)

        for part in split_message(result["summary"]):
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
    missing = missing_keys()
    if missing:
        raise ValueError(f"Missing required keys: {', '.join(missing)}")

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
