#accent tolerant
#noisy condition tolerant
#high recall (No keyword should miss)

import whisper
import numpy as np

model = whisper.load_model("base")

def transcribe_audio(waveform: np.ndarray) -> dict: #return scripts and metadata
    result = model.transcribe(
        waveform,
        language="en",
        word_timestamps=True,
        fp16=False
    )

    return {
        "text": result["text"],
        "segments": result["segments"]
    }
