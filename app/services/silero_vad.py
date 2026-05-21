import torch
import numpy as np
from silero_vad import load_silero_vad, get_speech_timestamps

model = load_silero_vad()


def apply_silero_vad(waveform: np.ndarray, sr: int) -> np.ndarray:
    """
    Filter out non-speech segments using Silero VAD (neural network-based).

    Uses a low threshold (0.25) to maximize recall — we prefer to keep
    borderline audio rather than miss a keyword.
    Returns a concatenated array of speech-only audio chunks.
    """
    if len(waveform) == 0:
        return np.array([], dtype=np.float32)

    audio_tensor = torch.from_numpy(waveform)

    speech_timestamps = get_speech_timestamps(
        audio_tensor,
        model,
        sampling_rate=sr,
        threshold=0.25,
        min_speech_duration_ms=150,  # ← ADD: "tea", "one" jaise short words catch hoge
        min_silence_duration_ms=300, # ← ADD: mid-sentence split nahi hoga
        speech_pad_ms=100            # ← ADD: word edges clip nahi honge
    )

    if not speech_timestamps:
        return np.array([], dtype=np.float32)

    speech_chunks = [waveform[ts["start"]:ts["end"]] for ts in speech_timestamps]
    return np.concatenate(speech_chunks).astype(np.float32)
