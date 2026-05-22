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


def adaptive_denoise(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Adaptive noise reduction using low-energy frames as noise profile.
    More accurate than fixed 0.5s profile in dynamic environments.
    """
    energy = np.abs(audio)
    noise_mask = energy < np.percentile(energy, 20)  # bottom 20% = noise
    noise_profile = audio[noise_mask]

    if len(noise_profile) > sr * 0.1:  # minimum 100ms noise sample
        return nr.reduce_noise(
            y=audio,
            sr=sr,
            y_noise=noise_profile,
            prop_decrease=0.75,  # slightly less aggressive than fixed
            stationary=False     # non-stationary mode for dynamic noise
        ).astype(np.float32)

    # fallback to original method if noise profile too short
    return reduce_noise(audio, sr)