"""Web version of Transcripator (same pipeline as the Telegram bot).

Run: python web.py   (PORT env, default 8000)
Upload an audio file in the browser -> raw / improved / summary.
"""

import logging
import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from pipeline import GROQ_MODEL, OPENROUTER_MODEL, has_groq, missing_keys, process_transcription

logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB

app = FastAPI(title="Transcripator")


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "missing_keys": missing_keys(),
        "models": {"openrouter": OPENROUTER_MODEL, "groq_fallback": GROQ_MODEL if has_groq() else None},
    }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return PAGE


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)):
    missing = missing_keys()
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Server misconfigured: missing {', '.join(missing)}",
        )
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected.")

    suffix = os.path.splitext(file.filename)[1].lower()
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            temp_path = tmp.name
            size = 0
            while chunk := await file.read(1024 * 256):
                size += len(chunk)
                if size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="File too large (max 25MB).",
                    )
                tmp.write(chunk)

        result = process_transcription(temp_path)
        return JSONResponse(result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=502, detail=f"Transcription failed: {e}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Transcripator — Web</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 760px; margin: 2rem auto; padding: 0 1rem; }
  textarea { width: 100%; min-height: 110px; margin: .25rem 0 1rem; }
  button { padding: .5rem 1.2rem; cursor: pointer; }
  #status { color: #666; }
</style>
</head>
<body>
<h1>Transcripator</h1>
<p>Upload a voice message or audio file (up to 25MB). The bot pipeline transcribes,
improves, and summarizes it.</p>
<form id="f">
  <input type="file" name="file" accept="audio/*,.ogg,.mp3,.wav,.m4a,.mp4,.aac,.flac,.webm" required>
  <button id="go" type="submit">Transcribe</button>
</form>
<p id="status"></p>
<h2>Raw transcription</h2>
<textarea id="raw" readonly></textarea>
<h2>Improved</h2>
<textarea id="improved" readonly></textarea>
<h2>Summary</h2>
<textarea id="summary" readonly></textarea>
<script>
const form = document.getElementById('f');
const status = document.getElementById('status');
const go = document.getElementById('go');
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  go.disabled = true;
  status.textContent = 'Processing… this may take a moment.';
  try {
    const res = await fetch('/api/transcribe', { method: 'POST', body: new FormData(form) });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    document.getElementById('raw').value = data.raw || '';
    document.getElementById('improved').value = data.improved || '';
    document.getElementById('summary').value = data.summary || '';
    status.textContent = 'Done.';
  } catch (err) {
    status.textContent = 'Error: ' + err.message;
  } finally {
    go.disabled = false;
  }
});
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
