import asyncio
import websockets
import sounddevice as sd
import numpy as np
import json

url = "ws://127.0.0.1:8000/api/v1/keyword/stream"


def print_order_slip(orders: list):
    print("\n" + "─" * 35)
    print("         🧾 ORDER RECEIVED")
    print("─" * 35)
    for order in orders:
        item = order["item"].title()
        qty = order["quantity"]
        print(f"  {item:<20} × {qty}")
    print("─" * 35)
    print("  [Listening for next order...]\n")


async def send_audio():
    async with websockets.connect(
        url,
        ping_interval=30,
        ping_timeout=120,
        close_timeout=30
    ) as websocket:

        loop = asyncio.get_running_loop()
        audio_queue = asyncio.Queue(maxsize=10)

        async def receive():
            try:
                while True:
                    message = await websocket.recv()
                    data = json.loads(message)

                    # Order complete — print slip
                    if data.get("order_complete"):
                        print_order_slip(data.get("orders", []))
                    else:
                        # Real time text — optional, comment out if too noisy
                        print(f"  → {data.get('text', '')}")

            except Exception as e:
                print("connection closed:", e)

        async def sender():
            while True:
                audio_bytes = await audio_queue.get()
                await websocket.send(audio_bytes)

        asyncio.create_task(receive())
        asyncio.create_task(sender())

        def callback(indata, frames, time, status):
            if status:
                print("Status:", status)
            if indata is None or len(indata) == 0:
                return

            audio_bytes = indata.copy().astype(np.float32).tobytes()

            def put_audio(b=audio_bytes):
                try:
                    audio_queue.put_nowait(b)
                except asyncio.QueueFull:
                    pass

            loop.call_soon_threadsafe(put_audio)

        with sd.InputStream(
            samplerate=16000,
            channels=1,
            dtype='float32',
            blocksize=1600,
            callback=callback
        ):
            print("\n" + "─" * 35)
            print("   🎤 Food Order System Ready")
            print("─" * 35)
            print("  Speak your order...\n")
            await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(send_audio())
    except KeyboardInterrupt:
        print("\n  System stopped.")