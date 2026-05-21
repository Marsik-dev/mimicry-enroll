FROM python:3.11-slim-bookworm

# System deps:
#  - libgl1, libglib2.0-0, libgles2, libglx0, libegl1 → MediaPipe, OpenCV
#  - libxext6, libx11-6, libsm6, libxrender1 → headless X
#  - libopus0, libvpx7, libsrtp2-1 → aiortc / WebRTC
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgles2 libglx0 libegl1 \
    libxext6 libx11-6 libsm6 libxrender1 \
    libopus0 libvpx7 libsrtp2-1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ARG PREPROC_VERSION=v0.2.0
ARG NPBK_VERSION=v0.2.0

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir \
    "https://github.com/Marsik-dev/mimicry-preproc/releases/download/${PREPROC_VERSION}/mimicry_preproc-0.2.0-py3-none-any.whl" \
    "https://github.com/Marsik-dev/npbk/releases/download/${NPBK_VERSION}/npbk-0.2.0-py3-none-any.whl" \
    && pip install --no-cache-dir -e .

EXPOSE 8501 8000

CMD ["mimicry-enroll"]
