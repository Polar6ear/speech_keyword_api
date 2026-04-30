from app.services.streaming_service import handle_stream
from fastapi import APIRouter, WebSocket

router = APIRouter(
    prefix='/keyword',
    tags=['Keyword Detection']
)

@router.websocket("/stream")
async def stream_audio(websocket: WebSocket):
    await handle_stream(websocket)
