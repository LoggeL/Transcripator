import os
import tempfile
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Telegram bot token
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Maximum file size (25MB in bytes)
MAX_FILE_SIZE = 25 * 1024 * 1024


async def start(update: Update, context):
    await update.message.reply_text(
        "Welcome! Send me a voice message or an audio file (up to 25MB), and I'll transcribe, improve, and summarize it for you."
    )


def transcribe_audio(file_path):
    with open(file_path, "rb") as file:
        transcription = groq_client.audio.transcriptions.create(
            file=(file_path, file.read()),
            model="whisper-large-v3",
            response_format="text",
            # language="auto",
            temperature=0.0,
        )
        print(transcription)
    return transcription


def improve_transcription(transcription):
    prompt = f"""
    Please improve the following transcription, fixing any errors and making it more readable in the target language. Only return the text, no further comments.

    {transcription}

    Improved transcription:
    """

    completion = groq_client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.3,
        # max_tokens=1000,
    )

    return completion.choices[0].message.content


def generate_summary(transcription):
    prompt = f"""
    Please provide a concise summary of the following transcription in the target language. If possible return bullet points. Write it out of the perspective of the transcript.

    {transcription}

    Summary:
    """

    completion = groq_client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.3,
        # max_tokens=300,
    )

    return completion.choices[0].message.content


async def process_audio(update: Update, context):
    # Check if it's a voice message or an audio file
    if update.message.voice:
        file = await update.message.voice.get_file()
        file_extension = ".ogg"
    elif update.message.audio:
        file = await update.message.audio.get_file()
        file_extension = os.path.splitext(update.message.audio.file_name)[1]
    else:
        await update.message.reply_text("Please send a voice message or an audio file.")
        return

    # Check file size
    if file.file_size > MAX_FILE_SIZE:
        await update.message.reply_text(
            "The file is too large. Please send an audio file up to 25MB."
        )
        return

    # Download the audio file
    with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
        await file.download_to_drive(custom_path=temp_file.name)
        temp_file_path = temp_file.name

    try:
        # Process the audio
        await update.message.reply_text(
            "Processing your audio. This may take a moment..."
        )

        original_transcription = transcribe_audio(temp_file_path)
        improved_transcription = improve_transcription(original_transcription)
        summary = generate_summary(improved_transcription)

        # Send the results
        # await update.message.reply_text(f"Original Transcription:\n{original_transcription}")
        await update.message.reply_text(f"{improved_transcription}")
        await update.message.reply_text(f"{summary}")

    except Exception as e:
        await update.message.reply_text(f"An error occurred: {str(e)}")

    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)


def main():
    # Check if required environment variables are set
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
