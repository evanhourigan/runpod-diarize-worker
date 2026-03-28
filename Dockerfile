FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch with CUDA 12.1
RUN pip install --no-cache-dir \
    torch torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Install pyannote and runpod from PyPI
RUN pip install --no-cache-dir \
    pyannote.audio>=3.1 \
    runpod>=1.6 \
    requests>=2.31

# Install wheel/setuptools (needed for whisperx build) then whisperx
RUN pip install --no-cache-dir wheel setuptools && \
    git clone https://github.com/m-bain/whisperx.git /tmp/whisperx && \
    pip install --no-cache-dir /tmp/whisperx && \
    rm -rf /tmp/whisperx

# Verify critical imports
RUN python3 -c "import whisperx; import pyannote.audio; print('All imports OK')"

COPY handler.py /app/handler.py

CMD ["python3", "handler.py"]
