FROM python:3.13-slim

WORKDIR /app

# ffmpeg/ffprobe needed to compress uploads > 24MB for Groq
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py pipeline.py web.py .

# Telegram bot (default):
CMD ["python", "main.py"]
# Web version instead (Dokploy/Fly/…): override CMD with ["python", "web.py"]
