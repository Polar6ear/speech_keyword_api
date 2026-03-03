from fastapi import FastAPI
from app.api.v1.keyword import router as keyword_router
import logging

app = FastAPI()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
app = FastAPI(
    title='STT keyword detection API',
    version='1.0.0'
)


@app.on_event("shutdown")
async def shutdown_event():
    logging.info("streaming ended")

app.include_router(keyword_router, prefix='/api/v1')