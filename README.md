# Transcripator

Audio transcription as Telegram bot **and** web version, sharing one pipeline:

1. **Groq Whisper** transcribes the audio.
2. **OpenRouter** improves the transcript (grammar, punctuation, readability).
3. **OpenRouter** summarizes it (bullet points, same language).

## Models (Nov 2026)

| Step | Default | Env override | Notes |
|---|---|---|---|
| Transcription (Groq) | `whisper-large-v3` | `GROQ_MODEL` | Best accuracy on Groq (~10.3% WER). `whisper-large-v3-turbo` is cheaper/faster ($0.04 vs $0.111/hr, 216x speed) but less accurate (~12% WER). |
| Improve + summary (OpenRouter) | `meta/muse-ember` | `OPENROUTER_MODEL` | Meta's Nov-2026 extraction/rewrite model; replaces `meta/muse-spark-1.3-contributor`. |

Both keys required: `GROQ_API_KEY`, `OPENROUTER_API_KEY` (see `.env.example`).
The bot additionally needs `TELEGRAM_BOT_TOKEN`.

## Setup

```sh
cp .env.example .env   # then fill in your keys
uv sync                # or: pip install -r requirements.txt
```

Requires Python 3.12+ and (only for uploads > 24 MB) `ffmpeg`/`ffprobe`.

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

- `pipeline.py`: shared pipeline (Groq transcription, OpenRouter polish, `process_transcription`).
- `main.py`: Telegram bot (thin wrapper around `pipeline`).
- `web.py`: FastAPI web version (thin wrapper around `pipeline`).
- `test_pipeline.py`: mocked tests, no API keys needed (`python -m unittest`).

## License

MIT License.
