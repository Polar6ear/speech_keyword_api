from app.core.model import model
import numpy as np


def transcribe_streaming(waveform: np.ndarray) -> dict:
    """
    Transcribe a waveform using faster-whisper with word-level timestamps.

    Configuration rationale:
    - beam_size=2, best_of=2: balance between speed and accuracy for real-time use
    - vad_filter=False: VAD is handled upstream by Silero for more control
    - word_timestamps=True: required for keyword timestamp extraction
    - temperature=0.0: deterministic output, avoids random variation
    - condition_on_previous_text=False: prevents hallucination carry-over between windows
    - compression_ratio_threshold=2.4: filters out overly repetitive outputs
    - log_prob_threshold=-1.2: rejects very low-confidence segments
    - no_speech_threshold=0.35: conservative threshold to retain borderline speech
    - repetition_penalty=1.05: mild penalty to reduce looping transcriptions
    """
    segments, _ = model.transcribe(
        waveform,
        language="en",
        beam_size=5,
        best_of=2,
        vad_filter=False,
        word_timestamps=True,
        temperature=0.0,
        condition_on_previous_text=False,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.2,
        no_speech_threshold=0.35,
        repetition_penalty=1.05,
        initial_prompt="Food ordering system. Items: pizza, burger, sandwich, coffee, tea, cold drink, fries, pasta, biryani. Quantities: one, two, three, four, five, half, large, small, medium.", 

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
        segment_list.append({
            "text": segment.text,
            "start": segment.start,
            "end": segment.end,
            "words": words_list
        })

    return {
        "text": full_text.strip(),
        "segments": segment_list
    }
