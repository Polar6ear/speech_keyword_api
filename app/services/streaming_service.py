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
from app.services.denoise import enhance_audio
from app.services.entity_extractor import extract_order_entities
from app.services.remove_overlap import remove_overlap
from app.services.order_session import OrderSession
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
    inference_queue = asyncio.Queue(maxsize=QUEUE_SIZE)
    emit_lock = asyncio.Lock()
    session = OrderSession(silence_threshold=4.0)

    async def inference_worker():
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
                if not segments:
                    continue

                new_text_chunks = []
                max_segment_end = session.last_sent_end_time

                for segment in segments:
                    absolute_end = segment["end"] + window_start_time
                    if absolute_end > session.last_sent_end_time:
                        segment_text = segment["text"].strip()
                        if new_text_chunks and segment_text == new_text_chunks[-1]:
                            continue
                        new_text_chunks.append(segment_text)
                        max_segment_end = max(max_segment_end, absolute_end)

                raw_text = " ".join(new_text_chunks).strip()
                if not raw_text:
                    continue

                # Hallucination filter
                if session.last_emitted_text and raw_text in session.last_emitted_text:
                    continue
                if session.is_hallucination(raw_text):
                    logger.debug(f"Hallucination detected: {raw_text}")
                    continue

                async with emit_lock:
                    if raw_text == session.last_emitted_text:
                        continue

                    final_text = remove_overlap(session.last_emitted_text, raw_text)
                    if not final_text:
                        continue

                    orders = extract_order_entities(final_text, FOOD_KEYWORDS)
                    detected = detect_keyword_service(transcription_result, FOOD_KEYWORDS)

                    detected = [
                        d for d in detected
                        if d["end"] + window_start_time > session.last_sent_end_time - 0.2
                    ]
                    detected = [
                        d for d in detected
                        if not session.is_duplicate_keyword(d["keyword"], d["start"])
                    ]

                    session.last_emitted_text = raw_text
                    session.last_sent_end_time = max_segment_end

                    for order in orders:
                        session.update_order(order["item"], order["quantity"])

                    logger.info(f"Sending text: {final_text}")
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

            # Silence detection — order complete
            if session.is_complete():
                await websocket.send_json({
                    "order_complete": True,
                    "orders": session.get_order_list()
                })
                # Flush queue
                while not inference_queue.empty():
                    try:
                        inference_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                # Reset
                session.reset()
                audio_buffer.clear()
                cursor = 0
                session.last_emitted_text = "" 
                logger.info("Order complete — session reset")

            buffer_array = np.array(audio_buffer)

            while cursor + WINDOW_SIZE <= len(buffer_array):
                window_start_time = cursor / sr
                current_window = buffer_array[cursor: cursor + WINDOW_SIZE]
                cursor += HOP_SIZE

                max_val = np.percentile(np.abs(current_window), 95)
                if max_val > 0:
                    current_window = current_window / (max_val + 1e-6)

                denoised = enhance_audio(current_window, sr)
                speech_audio = apply_silero_vad(denoised, sr)

                if len(speech_audio) < sr * 0.3:
                    logger.debug("Skipping: insufficient speech duration")
                    continue

                if len(speech_audio) / (len(current_window) + 1e-6) < 0.2:
                    logger.debug("Skipping: low speech ratio")
                    continue

                context_window = np.concatenate([session.previous_tail, speech_audio])

                if len(context_window) < sr * 0.5:
                    continue

                session.previous_tail = (
                    speech_audio[-CONTEXT_SIZE:]
                    if len(speech_audio) > CONTEXT_SIZE
                    else speech_audio
                )

                session.mark_speech()

                if inference_queue.full():
                    try:
                        inference_queue.get_nowait()
                        logger.warning("Dropped oldest queue item")
                    except asyncio.QueueEmpty:
                        pass

                inference_queue.put_nowait((context_window, window_start_time))

            # Trim buffer
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