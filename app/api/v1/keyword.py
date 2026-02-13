import os
import uuid
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse 

from app.services.audio_processing import load_audio_from_upload
from app.services.transcription import transcribe_audio
from app.services.keyword_detector import detect_keyword as detect_keyword_service
from app.config.keyword_config import FOOD_KEYWORDS

router = APIRouter(
    prefix='/keyword',
    tags=['Keyword Detection']
)

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
    

    