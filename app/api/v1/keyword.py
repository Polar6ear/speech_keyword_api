import os
import uuid
from fastapi import APIRouter, File, UploadFile, HTTPException, WebSocketDisconnect, WebSocket
from fastapi.responses import JSONResponse 
import numpy as np
from starlette.websockets import WebSocketDisconnect
import asyncio
# from app.services.audio_processing import load_audio_from_upload
from app.services.transcription import transcribe_streaming
from app.services.keyword_detector import detect_keyword as detect_keyword_service
from app.config.keyword_config import FOOD_KEYWORDS
from collections import deque
from app.services.remove_overlap import remove_overlap
# from app.services.vad import apply_vad
from app.services.silero_vad import apply_silero_vad

router = APIRouter(
    prefix='/keyword',
    tags=['Keyword Detection']
)
from collections import deque
import numpy as np
import asyncio
from fastapi import WebSocket, WebSocketDisconnect

# def is_sentence_complate(text: str) -> bool:
#     return text.endswith(('.', '!', '?'))

@router.websocket("/stream")
async def stream_audio(websocket: WebSocket):
    await websocket.accept()

    sr = 16000
    WINDOW_SIZE = sr * 6     
    HOP_SIZE = int(sr * 3)         

    audio_buffer = deque(maxlen=sr * 12)   
    cursor = 0
    previous_tail = np.array([], dtype=np.float32)
    
    silence_duration = 0.0
    SILENCE_THRESHOLD = 1.5
    utterance_audio = []

    #last_sent_text = ""
    last_sent_end_time = 0.0
    window_index = 0
    # inference_lock = asyncio.Lock()

    try:
        while True:
            audio_chunk = await websocket.receive_bytes()
            waveform = np.frombuffer(audio_chunk, dtype=np.float32)

            audio_buffer.extend(waveform)
            buffer_array = np.array(audio_buffer)

            while cursor + WINDOW_SIZE <= len(buffer_array):
                
                current_window_start_time = window_index * 3
                window_index += 1
                
                current_window = buffer_array[cursor: cursor + WINDOW_SIZE]
                cursor += HOP_SIZE

                # rms = np.sqrt(np.mean(current_window ** 2))
                # if rms >  0:
                #     current_window = current_window / rms * 0.2
                max_val = np.percentile(np.abs(current_window), 95)
                if max_val > 0:
                    current_window = current_window / (max_val + 1e-6)

                speech_audio = apply_silero_vad(current_window, sr)
                # if len(speech_audio) < int(sr * 0.2):
                if len(speech_audio) == 0:
                    silence_duration += 3.0   
                else:
                    silence_duration = 0.0
                    utterance_audio.append(speech_audio)

                if silence_duration >= SILENCE_THRESHOLD and utterance_audio:
                    full_audio = np.concatenate(utterance_audio)
                    
                    transcription_result = await asyncio.to_thread(
                        transcribe_streaming,
                        full_audio
                    )
                    
                    detected_items = detect_keyword_service(
                        transcription_result,
                        FOOD_KEYWORDS
                    )

                    await websocket.send_json({
                        "text": transcription_result.get("text", "").strip(),
                        "keywords": detected_items,
                        "is_final": True
                    })

                    await websocket.close()
                    return

                context_window = np.concatenate([previous_tail, speech_audio])
                previous_tail = current_window[-int(sr * 1.5):]

                # async with inference_lock:
                #     transcription_result = await asyncio.to_thread(
                #         transcribe_streaming,
                #         context_window
                #     )
                transcription_result = await asyncio.to_thread(
                    transcribe_streaming,
                    context_window
                )

                # current_text = transcription_result.get("text", "").strip()
                segments = transcription_result.get("segments", [])
                if not segments:
                    continue
                
                new_text_chunks = []
                new_keywords = []
                max_segment_end = last_sent_end_time

                for segment in segments:
                    absolute_start = segment["start"] + current_window_start_time
                    absolute_end = segment["end"] + current_window_start_time

                    if absolute_end > last_sent_end_time:
                        new_text_chunks.append(segment["text"].strip())
                        max_segment_end = max(max_segment_end, absolute_end)

                final_text = " ".join(new_text_chunks).strip()

                if not final_text:
                    continue 
                # if last_sent_text:
                #     new_text = remove_overlap(last_sent_text, current_text)
                # else:
                #     new_text = current_text

                # if len(new_text) < 2:
                #     continue 
                
                # if not is_sentence_complate(current_text):
                #     continue 

                # if len(new_text.split()) >= 3:
                #     last_sent_text += " " + new_text
                # else:
                #     continue
                # if new_text and new_text not in last_sent_text:
                #     last_sent_text += " " + new_text
                # else:
                #     continue 
                if not final_text:
                    continue
                last_sent_end_time = max_segment_end
                detected_items = detect_keyword_service(
                    transcription_result,
                    FOOD_KEYWORDS
                )
                for kw in detected_items:
                    kw_absolute_end = kw["end"] + current_window_start_time
                    if kw_absolute_end > last_sent_end_time:
                        new_keywords.append(kw)

                await websocket.send_json({
                    "text": final_text,
                    "keywords": new_keywords
                })

            if cursor >= len(buffer_array):
                cursor = 0
                audio_buffer.clear()
                window_index = 0
                last_sent_end_time = 0.0

    except WebSocketDisconnect:
        print("Client disconnected normally")

    except Exception as e:
        print("Unexpected error:", e)
        await websocket.close(code=1011)
# @router.post('/detect')
# async def detect_keyword(file: UploadFile = File(...)):
#     if not file.content_type or not file.content_type.startswith('audio/'):
#         raise HTTPException(
#             status_code=400,
#             detail="Invalid file type. Please upload a valid audio file. " 
#         )

#     try:
#         file_bytes = await file.read()
        
#         waveform, sr = load_audio_from_upload(file_bytes)

#         transcription_result = transcribe_audio(waveform)

#         if not transcription_result or "text" not in transcription_result:
#             raise HTTPException(status_code=500, detail="Transcription Failed")
        

#         detected_items = detect_keyword_service(
#             transcription_result,
#             FOOD_KEYWORDS 
#         )

#         response_payload = {
#             "transcript": transcription_result.get("text"),
#             "total_detected": len(detected_items),
#             "detected_keywords": detected_items
#         }

#         return JSONResponse(status_code=200, content=response_payload)
#     except HTTPException:
#         raise

#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f'Processing failed: {str(e)}'
#         )
    

    