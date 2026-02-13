import librosa
import soundfile as sf
import numpy as np
from io import BytesIO

def load_audio_from_upload(file_bytes: bytes, target_sr: int = 16000): #human speech 8khz --> nyquist rule = 8 * 2 = 16khz
    audio_buffer = BytesIO(file_bytes)
    waveform, sr = sf.read(audio_buffer)

    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1) #speech models mono expect krta hai sterio unnecessary data 

    if sr != target_sr:
        waveform = librosa.resample(
            waveform,
            orig_sr=sr,
            target_sr=target_sr
        )
        sr = target_sr

    max_val = np.max(np.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val

    return waveform.astype(np.float32), sr


