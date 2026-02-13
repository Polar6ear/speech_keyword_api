from fastapi import FastAPI
from app.api.v1.keyword import router as keyword_router

app = FastAPI(
    title='STT keyword detection API',
    version='1.0.0'
)

app.include_router(keyword_router, prefix='/api/v1')