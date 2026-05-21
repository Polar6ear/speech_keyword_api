import librosa
import soundfile as sf
import numpy as np
from io import BytesIO


def load_audio_from_upload(file_bytes: bytes, target_sr: int = 16000):
    """
    Load audio from raw bytes, convert to mono, resample to target_sr,
    and normalize to float32.

    Human speech sits around 8kHz; by Nyquist's rule we need 2x = 16kHz
    to capture it faithfully.
    """
    audio_buffer = BytesIO(file_bytes)
    waveform, sr = sf.read(audio_buffer)

    # Convert stereo to mono — speech models expect single-channel input
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)

    # Resample if needed
    if sr != target_sr:
        waveform = librosa.resample(
            waveform,
            orig_sr=sr,
            target_sr=target_sr,
            rs_type='kaiser_best'
        )
        sr = target_sr

    # Peak normalization
    max_val = np.max(np.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val

    return waveform.astype(np.float32), sr
