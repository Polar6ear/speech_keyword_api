import asyncio
import websockets
import sounddevice as sd
import numpy as np

url = "ws://127.0.0.1:8000/api/v1/keyword/stream"

async def send_audio():
    async with websockets.connect(
        url,
        ping_interval=20,
        ping_timeout=60
        ) as websocket:

        loop = asyncio.get_running_loop()

        async def receive():
            while True:
                message = await websocket.recv()
                print("Server:", message)

        asyncio.create_task(receive())

        def callback(indata, frames, time, status):
            audio_bytes = indata.astype(np.float32).tobytes()
            asyncio.run_coroutine_threadsafe(
                websocket.send(audio_bytes),
                loop
            )

        with sd.InputStream(
            device=1,
            samplerate=16000,
            channels=1,
            dtype='float32',
            callback=callback
        ):
            print("streaming started...")
            await asyncio.Future()
            
asyncio.run(send_audio())
