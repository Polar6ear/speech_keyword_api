import numpy as np
import noisereduce as nr 

def reduce_noise(waveform: np.ndarray, sr: int) -> np.ndarray:
    noise_sample = waveform[0:int(0.5 * sr)] #is segment se system learn krta hai noise pattern 
    return nr.reduce_noise(
        y=waveform,
        sr=sr,
        y_noise=noise_sample,
        prop_decrease=0.8
    ).astype(np.float32)