# Transcripator (Telegram Audio Transcription Bot)

This is a Telegram bot that transcribes, improves, and summarizes voice messages or audio files sent by users. The bot uses the Groq API for transcription and text improvement.

## Features

- Transcribes audio files and voice messages.
- Improves the transcription for readability.
- Generates a concise summary of the transcription.

## Requirements

- Python 3.10+
- Telegram Bot API token
- Groq API key

## Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/yourusername/telegram-audio-transcription-bot.git
   cd telegram-audio-transcription-bot
   ```

2. Create a virtual environment and activate it:

   ```sh
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   # source venv/bin/activate  # On macOS/Linux
   ```

3. Install the required packages:

   ```sh
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project directory and add your Telegram Bot API token and Groq API key:
   ```env
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token
   GROQ_API_KEY=your_groq_api_key
   ```

## Usage

1. Run the bot:

   ```sh
   python main.py
   ```

2. Open Telegram and start a chat with your bot. Send a voice message or an audio file (up to 25MB), and the bot will transcribe, improve, and summarize it for you.

## Code Overview

- `main.py`: The main script that initializes the bot and handles incoming messages.
  - `start`: Sends a welcome message to the user.
  - `transcribe_audio`: Transcribes the audio file using the Groq API.
  - `improve_transcription`: Improves the transcription for readability.
  - `generate_summary`: Generates a summary of the transcription.
  - `process_audio`: Processes the incoming audio file or voice message.
  - `main`: Initializes the bot and sets up the command and message handlers.

## License

This project is licensed under the MIT License.
