from faster_whisper import WhisperModel
import asyncio

model = WhisperModel(
    "medium",
    compute_type="int8",
    device="cpu"
)

inference_semaphore = asyncio.Semaphore(2)
