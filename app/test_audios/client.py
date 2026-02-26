import asyncio
import websockets
import sounddevice as sd
import numpy as np

url = "ws://127.0.0.1:8000/api/v1/keyword/stream"

async def send_audio():
    async with websockets.connect(
        url,
        ping_interval=30,
        ping_timeout=120
        ) as websocket:

        loop = asyncio.get_running_loop()
        audio_queue = asyncio.Queue(maxsize=10)

        async def receive():
            try:
                while True:
                    message = await websocket.recv()
                    print("Server:", message)
            except websockets.exceptions.ConnectionClosed:
                print("connection closed")

        asyncio.create_task(receive())
        
        async def sender():
            while True:
                audio_bytes = await audio_queue.get()
                await websocket.send(audio_bytes)

        asyncio.create_task(sender())

        def callback(indata, frames, time, status):
            audio_bytes = indata.astype(np.float32).tobytes()

            def safe_put():
                if not audio_queue.full():
                    audio_queue.put_nowait(audio_bytes)

            loop.call_soon_threadsafe(safe_put)
        
        with sd.InputStream(
            device = 1,
            samplerate=16000,
            channels=1,
            dtype='float32',
            callback=callback
        ):
            print(('streaming started...'))
            await asyncio.Future()
asyncio.run(send_audio())
