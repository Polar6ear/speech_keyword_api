#accent tolerant
#noisy condition tolerant
#high recall (No keyword should miss)

#import whisper
from app.core.models import model
import numpy as np

def transcribe_streaming(waveform: np.ndarray) -> dict: #return scripts and metadata
    segments, _ = model.transcribe(
        waveform,
        language="en",
        beam_size=2,
        best_of=2,
        vad_filter=False, 
        word_timestamps=True,
        temperature=0.0,
        condition_on_previous_text=False,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.2,
        no_speech_threshold=0.35,
        repetition_penalty=1.05
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
