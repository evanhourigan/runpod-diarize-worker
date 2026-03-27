# RunPod Diarization Worker

Serverless RunPod worker for speaker diarization using pyannote.

## Build & Deploy

```bash
# Build the Docker image
docker build -t <your-dockerhub-user>/runpod-diarize-worker:latest .

# Push to Docker Hub
docker push <your-dockerhub-user>/runpod-diarize-worker:latest
```

Then on RunPod:
1. Go to Serverless → New Endpoint
2. Set the Docker image to `<your-dockerhub-user>/runpod-diarize-worker:latest`
3. Add environment variable: `HUGGINGFACE_TOKEN=<your-hf-token>`
4. Select a GPU type (RTX 3090/4090 or A100 recommended)
5. Copy the endpoint ID and set it as `RUNPOD_DIARIZE_ENDPOINT_ID`

## API

**Input:**
```json
{
    "input": {
        "audio_url": "https://presigned-r2-url...",
        "transcript_segments": [
            {"start": 0.0, "end": 5.2, "text": "Hello there", "words": [...]}
        ],
        "num_speakers": 2,
        "language": "en"
    }
}
```

**Output:**
```json
{
    "segments": [
        {"start": 0.0, "end": 5.2, "text": "Hello there", "speaker": "SPEAKER_00"}
    ]
}
```
