"""RunPod serverless handler for speaker diarization.

Accepts an audio URL and transcript segments, runs pyannote diarization,
and returns segments with speaker labels.

Expected input:
{
    "audio_url": "https://...",           # Presigned URL to audio file
    "transcript_segments": [...],          # Aligned transcript segments with words
    "num_speakers": 2,                     # Expected speaker count
    "min_speakers": 2,                     # (optional) Min speakers
    "max_speakers": 2,                     # (optional) Max speakers
    "language": "en"                       # Language code
}

Returns:
{
    "segments": [
        {"start": 0.0, "end": 5.2, "text": "Hello...", "speaker": "SPEAKER_00"},
        ...
    ]
}
"""

import io
import os
import sys
import logging
import tempfile
import warnings

# Suppress noisy ML library output
warnings.filterwarnings("ignore")
os.environ["PYTORCH_LIGHTNING_LOG_LEVEL"] = "ERROR"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

for _name in [
    "pytorch_lightning", "lightning", "pyannote", "pyannote.audio",
    "speechbrain", "torch", "torchaudio", "numba", "whisperx",
]:
    logging.getLogger(_name).setLevel(logging.ERROR)

import runpod
import requests
import torch
import whisperx
from whisperx.diarize import DiarizationPipeline, assign_word_speakers

logger = logging.getLogger("diarize-worker")
logging.basicConfig(level=logging.INFO)

# Load HuggingFace token from environment (set in RunPod template env vars)
HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN", "")
if not HF_TOKEN:
    logger.warning("HUGGINGFACE_TOKEN not set — diarization will fail")

# Detect device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Using device: %s", DEVICE)

# Pre-load diarization model at startup (stays warm between jobs)
logger.info("Loading diarization model...")
DIARIZE_MODEL = DiarizationPipeline(token=HF_TOKEN, device=DEVICE)
logger.info("Diarization model loaded")


def download_audio(url: str, dest: str) -> None:
    """Download audio from a presigned URL."""
    response = requests.get(url, timeout=300)
    response.raise_for_status()
    with open(dest, "wb") as f:
        f.write(response.content)


def handler(job: dict) -> dict:
    """RunPod handler — receives a job, returns diarized segments."""
    job_input = job.get("input", {})

    audio_url = job_input.get("audio_url")
    transcript_segments = job_input.get("transcript_segments", [])
    num_speakers = job_input.get("num_speakers", 2)
    min_speakers = job_input.get("min_speakers", num_speakers)
    max_speakers = job_input.get("max_speakers", num_speakers)
    language = job_input.get("language", "en")

    if not audio_url:
        return {"error": "audio_url is required"}
    if not transcript_segments:
        return {"error": "transcript_segments is required"}

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.wav")

        # Download audio
        logger.info("Downloading audio...")
        try:
            download_audio(audio_url, audio_path)
        except Exception as e:
            return {"error": f"Failed to download audio: {e}"}

        # Re-align transcript locally for accurate word timestamps
        logger.info("Aligning transcript (language=%s)...", language)
        try:
            audio = whisperx.load_audio(audio_path)
            model_a, metadata = whisperx.load_align_model(
                language_code=language, device=DEVICE
            )
            aligned_result = whisperx.align(
                transcript_segments,
                model_a,
                metadata,
                audio,
                DEVICE,
                return_char_alignments=False,
            )
        except Exception as e:
            return {"error": f"Alignment failed: {e}"}

        # Run diarization
        logger.info("Diarizing (speakers=%d-%d)...", min_speakers, max_speakers)
        try:
            diarize_segments = DIARIZE_MODEL(
                audio_path,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
        except Exception as e:
            return {"error": f"Diarization failed: {e}"}

        # Assign speakers to segments
        logger.info("Assigning speakers to segments...")
        result = assign_word_speakers(diarize_segments, aligned_result)

        segments = []
        for seg in result.get("segments", []):
            segments.append({
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
                "text": seg.get("text", "").strip(),
                "speaker": seg.get("speaker", "UNKNOWN"),
            })

        logger.info("Done — %d segments, %d speakers",
                     len(segments),
                     len(set(s["speaker"] for s in segments)))

        return {"segments": segments}


runpod.serverless.start({"handler": handler})
