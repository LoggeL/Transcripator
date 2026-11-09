# Transcripator

Audio transcription as Telegram bot **and** web version, sharing one pipeline:

1. **Ember** (`meta/muse-ember` via OpenRouter) transcribes the audio directly
   (Responses API, `input_audio`).
2. **Ember** improves the transcript (grammar, punctuation, readability).
3. **Ember** summarizes it (bullet points, same language).

> No Groq required. Optional: set `GROQ_API_KEY` for a Whisper fallback
> (`whisper-large-v3`) if the Ember audio call fails.

## Models (Nov 2026)

| Step | Default | Env override | Notes |
|---|---|---|---|
| Transcription, improve, summary | `meta/muse-ember` | `OPENROUTER_MODEL` | Meta's Nov-2026 model with audio input. Replaces `muse-spark-1.3-contributor` + separate Whisper. |
| Fallback only (Groq) | `whisper-large-v3` | `GROQ_MODEL`, `GROQ_API_KEY` | Only used when the Ember audio call fails. |

Required key: `OPENROUTER_API_KEY` (see `.env.example`).
The bot additionally needs `TELEGRAM_BOT_TOKEN`.

## Setup

```sh
cp .env.example .env   # then fill in your keys
uv sync                # or: pip install -r requirements.txt
```

Requires Python 3.12+ and `ffmpeg` (converts `.ogg` etc. to MP3 for the
Responses API, and for the Groq fallback).

## Telegram bot

```sh
python main.py
```

Start a chat with your bot and send a voice message or audio file (up to 25 MB).

## Web version

```sh
python web.py          # http://localhost:8000 (PORT env, default 8000)
```

Open the page in a browser, upload an audio file, and get raw / improved /
summary. API: `POST /api/transcribe` (multipart `file`), `GET /health`.

## Docker (Dokploy etc.)

The image runs the Telegram bot by default (`CMD ["python", "main.py"]`).
For the web version, override the command to `python web.py`.

## Code overview

- `pipeline.py`: shared pipeline (`transcribe_with_ember`, Groq fallback, `process_transcription`).
- `main.py`: Telegram bot (thin wrapper around `pipeline`).
- `web.py`: FastAPI web version (thin wrapper around `pipeline`).
- `test_pipeline.py`: mocked tests, no API keys needed (`python -m unittest`).

## License

MIT License.
