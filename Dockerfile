FROM python:3.11-slim

# System deps for mimicry-preproc (MediaPipe, OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgles2 libglx0 libegl1 \
    libxext6 libx11-6 libsm6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install packages from GitHub Releases (or local build)
# В production: pip install https://github.com/Marsik-dev/mimicry-preproc/releases/latest/download/...
# Для разработки: монтируем src папки
COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir -e .

EXPOSE 8501 8000

CMD ["mimicry-enroll"]
