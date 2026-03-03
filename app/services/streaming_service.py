import asyncio
import numpy as np
from collections import deque
from fastapi import WebSocket, WebSocketDisconnect
from app.core.models import inference_semaphore
from app.services.transcription import transcribe_streaming
from app.services.keyword_detector import detect_keyword as detect_keyword_service
from app.config.keyword_config import FOOD_KEYWORDS
from app.services.silero_vad import apply_silero_vad
from app.services.denoise import reduce_noise
import time
import logging
from app.config.streaming_config import (
    SAMPLE_RATE,
    WINDOW_SEC,
    HOP_SEC,
    CONTEXT_SEC,
    MAX_BUFFER_SEC,
    QUEUE_SIZE
)
sr = SAMPLE_RATE
logger = logging.getLogger(__name__)

def is_similar(a, b, threashold=0.85):
    return SequenceMatcher(None, a, b).ratio() > threashold

async def handle_stream(websocket: WebSocket):
    await websocket.accept()

    WINDOW_SIZE = int(sr * WINDOW_SEC)
    HOP_SIZE = int(sr * HOP_SEC)
    CONTEXT_SIZE = int(sr * CONTEXT_SEC)

    audio_buffer = deque(maxlen=sr * MAX_BUFFER_SEC)    
    cursor = 0
    previous_tail = np.array([], dtype=np.float32)
    last_sent_end_time = 0.0

    inference_queue = asyncio.Queue(maxsize=QUEUE_SIZE)

    async def inference_worker():
        nonlocal last_sent_end_time

        while True:
            context_window, window_start_time = await inference_queue.get()
            logger.info("Worker picked window")
            try:
                async with inference_semaphore:
                    start = time.time()
                    transcription_result = await asyncio.to_thread(
                        transcribe_streaming,
                        context_window
                    )
                    end = time.time()
                    logger.info(f"Whisper Infer Timer:{end - start:.3f}s")

                segments = transcription_result.get("segments", [])
                print("Segments returned:", segments)

                if not segments:
                    continue

                new_text_chunks = []
                max_segment_end = last_sent_end_time

                for segment in segments:
                    absolute_end = segment["end"] + window_start_time

                    if absolute_end > last_sent_end_time:
                        segment_text = segment["text"].strip()

                        if new_text_chunks:
                            if segment_text == new_text_chunks[-1]:
                                continue

                        new_text_chunks.append(segment_text)
                        max_segment_end = max(max_segment_end, absolute_end)

                final_text = " ".join(new_text_chunks).strip()

                if final_text:
                    detected = detect_keyword_service(
                        transcription_result,
                        FOOD_KEYWORDS
                    )
                    detected = [
                        d for d in detected
                        if d["end"] + window_start_time > last_sent_end_time - 0.2
                    ]       
                    last_sent_end_time = max_segment_end

                    print(" Sending text:", final_text)
                    print(" Detected:", detected)

                    await websocket.send_json({
                        "text": final_text,
                        "keywords": detected,
                        "is_final": False
                })

            except Exception as e:
                logger.error(f"Inference error: {e}")
    worker_task = asyncio.create_task(inference_worker())

    try:
        while True:
            audio_chunk = await websocket.receive_bytes()
            waveform = np.frombuffer(audio_chunk, dtype=np.float32)
            audio_buffer.extend(waveform)

            buffer_array = np.array(audio_buffer)

            while cursor + WINDOW_SIZE <= len(buffer_array):

                window_start_time = cursor / sr

                current_window = buffer_array[cursor: cursor + WINDOW_SIZE]
                cursor += HOP_SIZE

                # normalize
                max_val = np.percentile(np.abs(current_window), 95)
                if max_val > 0:
                    current_window = current_window / (max_val + 1e-6)
                
                denoised = reduce_noise(current_window, sr)
                speech_audio = apply_silero_vad(current_window, sr)
                if len(speech_audio) < sr * 0.3:
                    print(" Skipping due to VAD")
                    continue

                logger.debug("VAD passed")
                context_window = np.concatenate([previous_tail, speech_audio])
                if len(context_window) < sr * 0.5:
                    continue
                previous_tail = speech_audio[-CONTEXT_SIZE:]

                try:
                    inference_queue.put_nowait((context_window, window_start_time))
                except asyncio.QueueFull:
                    pass
                    
            if cursor > sr * 10:
                audio_buffer = deque(list(audio_buffer)[cursor:], maxlen=sr * 20)
                cursor = 0

    except WebSocketDisconnect:
        print("Client disconnected")

    except Exception as e:
        logger.exception("Unexpected error occurred")
    finally:
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass