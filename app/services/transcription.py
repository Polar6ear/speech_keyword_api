#accent tolerant
#noisy condition tolerant
#high recall (No keyword should miss)

#import whisper
from faster_whisper import WhisperModel
import numpy as np

#model = whisper.load_model("base")
model = WhisperModel(
    # "base",
    "tiny",
    compute_type='int8',
    device="cpu"
)
def transcribe_streaming(waveform: np.ndarray) -> dict: #return scripts and metadata
    segments, _ = model.transcribe(
        waveform,
        language="en",
        beam_size=1,
        vad_filter=True, 
        word_timestamps=True
        # fp16=False
    )
    full_text = ""
    segment_list = []

    for segment in segments:
        full_text += segment.text + " "
        words_list = []
        if segment.words is not None:
            for word in segment.words:
                words_list.append({
                    "word": word.word,
                    "start": word.start,
                    "end": word.end
                })
        segment_list.append(
            {
                "text": segment.text,
                "start": segment.start,
                "end": segment.end,
                "words": words_list
            }
        )
    return {
        "text": full_text.strip(),
        "segments": segment_list
    }
