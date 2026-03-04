import torch
import numpy as np
from silero_vad import load_silero_vad, get_speech_timestamps

model = load_silero_vad()

def apply_silero_vad(waveform: np.ndarray, sr: int) -> np.ndarray:
    if len(waveform) == 0:
        return np.array([], dtype=np.float32)

    audio_tensor = torch.from_numpy(waveform)

    speech_timestamps = get_speech_timestamps(
        audio_tensor,
        model,
        sampling_rate=sr,
        threshold=0.7 
    )

    if not speech_timestamps:
        return np.array([], dtype=np.float32)

    speech_chunks = []
    for ts in speech_timestamps:
        speech_chunks.append(waveform[ts["start"]:ts["end"]])

    return np.concatenate(speech_chunks).astype(np.float32)