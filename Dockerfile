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

# Install whisperx from git (separate step — needs PyPI index for its deps)
RUN pip install --no-cache-dir \
    git+https://github.com/m-bain/whisperx.git

# Verify whisperx installed correctly
RUN python3 -c "import whisperx; print('whisperx OK')"

COPY handler.py /app/handler.py

CMD ["python3", "handler.py"]
