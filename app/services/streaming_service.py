import asyncio
import numpy as np
import time
import logging
from collections import deque
from fastapi import WebSocket, WebSocketDisconnect

from app.core.model import inference_semaphore
from app.services.transcription import transcribe_streaming
from app.services.keyword_detector import detect_keyword as detect_keyword_service
from app.config.keyword_config import FOOD_KEYWORDS
from app.services.silero_vad import apply_silero_vad
from app.services.denoise import reduce_noise
from app.services.entity_extractor import extract_order_entities
from app.services.remove_overlap import remove_overlap
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


async def handle_stream(websocket: WebSocket):
    await websocket.accept()

    WINDOW_SIZE = int(sr * WINDOW_SEC)
    HOP_SIZE = int(sr * HOP_SEC)
    CONTEXT_SIZE = int(sr * CONTEXT_SEC)

    audio_buffer = deque(maxlen=sr * MAX_BUFFER_SEC)
    cursor = 0
    previous_tail = np.array([], dtype=np.float32)
    last_sent_end_time = 0.0
    last_emitted_text = ""
    inference_queue = asyncio.Queue(maxsize=QUEUE_SIZE)

    # Lock to prevent race conditions between parallel inference workers
    emit_lock = asyncio.Lock()

    async def inference_worker():
        nonlocal last_sent_end_time
        nonlocal last_emitted_text

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
                    elapsed = time.time() - start
                    logger.info(f"Whisper inference time: {elapsed:.3f}s")

                segments = transcription_result.get("segments", [])
                logger.debug(f"Segments returned: {segments}")

                if not segments:
                    continue

                new_text_chunks = []
                max_segment_end = last_sent_end_time

                for segment in segments:
                    absolute_end = segment["end"] + window_start_time

                    if absolute_end > last_sent_end_time:
                        segment_text = segment["text"].strip()

                        if new_text_chunks and segment_text == new_text_chunks[-1]:
                            continue

                        new_text_chunks.append(segment_text)
                        max_segment_end = max(max_segment_end, absolute_end)

                raw_text = " ".join(new_text_chunks).strip()
                if not raw_text:
                    continue

                async with emit_lock:
                    if raw_text == last_emitted_text:
                        continue

                    final_text = remove_overlap(last_emitted_text, raw_text)
                    if not final_text:
                        continue

                    orders = extract_order_entities(final_text, FOOD_KEYWORDS)

                    detected = detect_keyword_service(
                        transcription_result,
                        FOOD_KEYWORDS
                    )
                    # Only emit keywords that fall within the current window
                    detected = [
                        d for d in detected
                        if d["end"] + window_start_time > last_sent_end_time - 0.2
                    ]

                    last_emitted_text = raw_text
                    last_sent_end_time = max_segment_end

                    logger.info(f"Sending text: {final_text}")
                    logger.info(f"Detected keywords: {detected}")

                    await websocket.send_json({
                        "text": final_text,
                        "keywords": detected,
                        "orders": orders,
                        "is_final": False
                    })

            except Exception as e:
                logger.error(f"Inference error: {e}")

    workers = [
        asyncio.create_task(inference_worker())
        for _ in range(2)
    ]

    try:
        while True:
            try:
                audio_chunk = await websocket.receive_bytes()
                waveform = np.frombuffer(audio_chunk, dtype=np.float32)

                if waveform is None or len(waveform) == 0:
                    logger.warning("Empty audio chunk received")
                    continue

                if len(waveform) < 100:
                    logger.debug("Very small chunk, skipped")
                    continue

            except WebSocketDisconnect:
                raise

            except Exception as e:
                logger.warning(f"Invalid audio chunk: {e}")
                continue

            audio_buffer.extend(waveform)
            buffer_array = np.array(audio_buffer)

            while cursor + WINDOW_SIZE <= len(buffer_array):
                window_start_time = cursor / sr
                current_window = buffer_array[cursor: cursor + WINDOW_SIZE]
                cursor += HOP_SIZE

                # Percentile-based normalization (robust to outlier spikes)
                max_val = np.percentile(np.abs(current_window), 95)
                if max_val > 0:
                    current_window = current_window / (max_val + 1e-6)

                denoised = reduce_noise(current_window, sr)
                speech_audio = apply_silero_vad(denoised, sr)

                MIN_SPEECH_SEC = 0.3
                if len(speech_audio) < sr * MIN_SPEECH_SEC:
                    logger.debug("Skipping: insufficient speech duration")
                    continue

                speech_ratio = len(speech_audio) / (len(current_window) + 1e-6)
                if speech_ratio < 0.2:
                    logger.debug("Skipping: low speech ratio")
                    continue

                logger.debug("VAD passed")
                context_window = np.concatenate([previous_tail, speech_audio])

                if len(context_window) < sr * 0.5:
                    continue

                previous_tail = (
                    speech_audio[-CONTEXT_SIZE:]
                    if len(speech_audio) > CONTEXT_SIZE
                    else speech_audio
                )

                # Drop oldest queued item if queue is full to maintain low latency
                if inference_queue.full():
                    try:
                        inference_queue.get_nowait()
                        logger.warning("Dropped oldest queue item to maintain latency")
                    except asyncio.QueueEmpty:
                        pass

                inference_queue.put_nowait((context_window, window_start_time))

            # Trim buffer to prevent unbounded memory growth
            if len(audio_buffer) > sr * MAX_BUFFER_SEC:
                trim_size = int(sr * 5)
                old_len = len(audio_buffer)
                audio_buffer = deque(
                    list(audio_buffer)[-trim_size:],
                    maxlen=sr * MAX_BUFFER_SEC
                )
                cursor = max(0, cursor - (old_len - trim_size))

    except WebSocketDisconnect:
        logger.info("Client disconnected")

    except Exception as e:
        logger.exception("Unexpected error in stream handler")

    finally:
        for w in workers:
            w.cancel()
        for w in workers:
            try:
                await w
            except asyncio.CancelledError:
                pass