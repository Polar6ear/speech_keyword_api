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
from app.services.vad import apply_vad
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
    
    last_sent_text = ""

    inference_lock = asyncio.Lock()

    try:
        while True:
            audio_chunk = await websocket.receive_bytes()
            waveform = np.frombuffer(audio_chunk, dtype=np.float32)

            audio_buffer.extend(waveform)

            buffer_array = np.array(audio_buffer)

            while cursor + WINDOW_SIZE <= len(buffer_array):
                
                current_window = buffer_array[cursor: cursor + WINDOW_SIZE]
                cursor += HOP_SIZE

                rms = np.sqrt(np.mean(current_window ** 2))
                if rms >  0:
                    current_window = current_window / rms * 0.2

                speech_audio = apply_vad(current_window, sr)

                if len(speech_audio) < int(sr * 0.4):
                    continue

                context_window = np.concatenate([previous_tail, speech_audio])
                previous_tail = current_window[-int(sr * 1.5):]

                async with inference_lock:
                    transcription_result = await asyncio.to_thread(
                        transcribe_streaming,
                        context_window
                    )

                current_text = transcription_result.get("text", "").strip()

                if len(current_text) < 3:
                    continue

                if last_sent_text:
                    new_text = remove_overlap(last_sent_text, current_text)
                else:
                    new_text = current_text

                if len(new_text) < 2:
                    continue 
                
                # if not is_sentence_complate(current_text):
                #     continue 

                # if len(new_text.split()) >= 3:
                #     last_sent_text += " " + new_text
                # else:
                #     continue
                if new_text and new_text not in last_sent_text:
                    last_sent_text += " " + new_text
                else:
                    continue 
                 
                detected_items = detect_keyword_service(
                    transcription_result,
                    FOOD_KEYWORDS
                )

                await websocket.send_json({
                    "text": new_text,
                    "keywords": detected_items
                })

            if cursor >= len(buffer_array):
                cursor = 0
                audio_buffer.clear()

    except WebSocketDisconnect:
        print("Client disconnected normally")

    except Exception as e:
        print("Unexpected error:", e)
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
    

    