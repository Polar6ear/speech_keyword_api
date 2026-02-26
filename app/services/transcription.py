#accent tolerant
#noisy condition tolerant
#high recall (No keyword should miss)

#import whisper
from faster_whisper import WhisperModel
import numpy as np

#model = whisper.load_model("base")
model = WhisperModel(
    # "base",
    "medium.en",
    compute_type="float32",
    device="cpu"
)
def transcribe_streaming(waveform: np.ndarray) -> dict: #return scripts and metadata
    segments, _ = model.transcribe(
        waveform,
        language="en",
        beam_size=8,
        best_of=5,
        vad_filter=False, 
        word_timestamps=True,
        temperature=0.0,
        condition_on_previous_text=True,
        repetition_penalty=1.05,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
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
