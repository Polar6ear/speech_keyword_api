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

router = APIRouter(
    prefix='/keyword',
    tags=['Keyword Detection']
)
@router.websocket("/stream")
async def stream_audio(websocket: WebSocket):
    await websocket.accept()

    audio_buffer = np.zeros(0, dtype=np.float32)

    sr = 16000
    WINDOW_SIZE = sr * 5      
    HOP_SIZE = sr * 2        
    processed_until = 0

    try:
        while True:
            audio_chunk = await websocket.receive_bytes()
            waveform = np.frombuffer(audio_chunk, dtype=np.float32)

            # Efficient append
            audio_buffer = np.append(audio_buffer, waveform)

            # Enough data for first window?
            if len(audio_buffer) >= WINDOW_SIZE:

                # Sliding window condition
                if len(audio_buffer) - processed_until >= HOP_SIZE:

                    start = len(audio_buffer) - WINDOW_SIZE
                    current_window = audio_buffer[start:]

                    processed_until = len(audio_buffer)

                    transcription_result = await asyncio.to_thread(
                        transcribe_streaming,
                        current_window
                    )

                    current_text = transcription_result.get("text", "").strip()

                    if len(current_text) < 3:
                        continue

                    detected_items = detect_keyword_service(
                        transcription_result,
                        FOOD_KEYWORDS
                    )

                    await websocket.send_json({
                        "text": current_text,
                        "keywords": detected_items
                    })

            # Prevent infinite growth (memory safety)
            if len(audio_buffer) > sr * 12:
                audio_buffer = audio_buffer[-sr * 6:]
                processed_until = max(0, processed_until - sr * 6)

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
    

    