import numpy as np
import noisereduce as nr


def reduce_noise(waveform: np.ndarray, sr: int) -> np.ndarray:
    """
    Reduce background noise using a noise profile sampled from
    the first 0.5 seconds of the audio segment.
    """
    noise_sample = waveform[0:int(0.5 * sr)]
    return nr.reduce_noise(
        y=waveform,
        sr=sr,
        y_noise=noise_sample,
        prop_decrease=0.8
    ).astype(np.float32)
