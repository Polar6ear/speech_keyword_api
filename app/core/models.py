from faster_whisper import WhisperModel
import asyncio

model = WhisperModel(
    # "base",
    "medium.en",
    compute_type="int8",
    device="cpu"
) 

inference_semaphore = asyncio.Semaphore(2)