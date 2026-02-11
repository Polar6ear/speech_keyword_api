import webrtcvad
import numpy as np

def apply_vad(waveform: np.ndarray, sr: int) -> np.ndarray:  #returns speech only waveform
    pcm_waveform = (waveform * 32768).astype(np.int16)
    vad = webrtcvad.Vad(3) # 0 --> least aggresive, 3 --> most aggresive
    frame_duration = 30
    frame_size = int(sr * frame_duration / 1000) # if sr = 16000 fraze_size = 480 samples
    
