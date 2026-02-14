import os
import uuid
from fastapi import APIRouter, File, UploadFile, HTTPException, WebSocketDisconnect, WebSocket
from fastapi.responses import JSONResponse 
import numpy as np
import asyncio
# from app.services.audio_processing import load_audio_from_upload
from app.services.transcription import transcribe_streaming
from app.services.keyword_detector import detect_keyword as detect_keyword_service
from app.config.keyword_config import FOOD_KEYWORDS

router = APIRouter(
    prefix='/keyword',
    tags=['Keyword Detection']
)

@router.websocket('/stream') #ws://localhost aesa krke hoga ye not http://localhost...
async def stream_audio(websocket: WebSocket):
    await websocket.accept()
    audio_buffer = np.array([], dtype=np.float32)
    previous_text = ""
    try:
        while True:
            audio_chunk = await websocket.receive_bytes()
            #waveform, sr = load_audio_from_upload(audio_chunk)
            waveform = np.frombuffer(audio_chunk, dtype=np.float32)
            
            sr=16000

            audio_buffer = np.concatenate((audio_buffer, waveform))
            if len(audio_buffer) >= sr * 3:
            
                if len(audio_buffer) > sr * 5:
                    audio_buffer = audio_buffer[-sr * 5:]
                
                transcription_result = await asyncio.to_thread(
                    transcribe_streaming,
                    audio_buffer
                )
                current_text = transcription_result['text']

                if current_text != previous_text:
                    previous_text = current_text

                    detected_items = detect_keyword_service(
                        transcription_result,
                        FOOD_KEYWORDS
                    )
                    
                    await websocket.send_json({
                        "text": current_text,
                        "keywords": detected_items
                })
                # audio_buffer = audio_buffer[-sr:]

            # detect_items = detect_keyword(
            #     transcription_result,
            #     FOOD_KEYWORDS
            # )

            # await websocket.send_json({
            #     "transcript": transcription_result.get("text"),
            #     "keyword": detect_items
            # })

    except WebSocketDisconnect: 
        print('Client Disconnect')


@router.post('/detect')
async def detect_keyword(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith('audio/'):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a valid audio file. " 
        )

    try:
        file_bytes = await file.read()
        
        waveform, sr = load_audio_from_upload(file_bytes)

        transcription_result = transcribe_audio(waveform)

        if not transcription_result or "text" not in transcription_result:
            raise HTTPException(status_code=500, detail="Transcription Failed")
        

        detected_items = detect_keyword_service(
            transcription_result,
            FOOD_KEYWORDS 
        )

        response_payload = {
            "transcript": transcription_result.get("text"),
            "total_detected": len(detected_items),
            "detected_keywords": detected_items
        }

        return JSONResponse(status_code=200, content=response_payload)
    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f'Processing failed: {str(e)}'
        )
    

    