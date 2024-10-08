import os
import tempfile
from typing import Union
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Telegram bot token
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Constants
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB in bytes
MAX_MESSAGE_LENGTH = 4096


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a welcome message when the command /start is issued."""
    await update.message.reply_text(
        "Welcome! Send me a voice message or an audio file (up to 25MB), and I'll transcribe, improve, and summarize it for you."
    )


def split_message(message: str, max_length: int = MAX_MESSAGE_LENGTH) -> list[str]:
    """Split a message into chunks that fit within Telegram's message length limit."""
    return [message[i : i + max_length] for i in range(0, len(message), max_length)]


def transcribe_audio(file_path: str) -> str:
    """Transcribe the audio file using Groq's API."""
    with open(file_path, "rb") as file:
        transcription = groq_client.audio.transcriptions.create(
            file=(file_path, file.read()),
            model="whisper-large-v3",
            response_format="text",
            temperature=0.0,
        )
    return transcription


def improve_transcription(transcription: str) -> str:
    """Improve the transcription using Groq's language model."""
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

    completion = groq_client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    return completion.choices[0].message.content


def generate_summary(transcription: str) -> str:
    """Generate a summary of the transcription using Groq's language model."""
    prompt = f"""
    Task: Summarize the following transcription
    Instructions:
    1. Provide a concise summary of the main points
    2. Use bullet points for clarity
    3. Write from the perspective of the transcript
    4. Capture the key ideas and any important details
    5. Ensure the summary is coherent and easy to understand

    Transcription:
    {transcription}

    Summary:
    """

    completion = groq_client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    return completion.choices[0].message.content


async def process_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process the incoming audio file or voice message."""
    file: Union[None, "telegram.File"] = None
    file_extension = ""

    if update.message.voice:
        file = await update.message.voice.get_file()
        file_extension = ".ogg"
    elif update.message.audio:
        file = await update.message.audio.get_file()
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
        await file.download_to_drive(custom_path=temp_file.name)
        temp_file_path = temp_file.name

    try:
        await update.message.reply_text(
            "Processing your audio. This may take a moment..."
        )

        original_transcription = transcribe_audio(temp_file_path)
        improved_transcription = improve_transcription(original_transcription)

        for part in split_message(improved_transcription):
            await update.message.reply_text(part, do_quote=True)

        summary = generate_summary(improved_transcription)
        for part in split_message(summary):
            await update.message.reply_text(part, do_quote=True)

    except Exception as e:
        await update.message.reply_text(f"An error occurred: {str(e)}")

    finally:
        os.unlink(temp_file_path)


def main() -> None:
    """Set up and run the bot."""
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN is not set in the .env file")
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY is not set in the .env file")

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(
        MessageHandler(filters.VOICE | filters.AUDIO, process_audio)
    )

    application.run_polling()


if __name__ == "__main__":
    main()
