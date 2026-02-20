import webrtcvad
import numpy as np

def apply_vad(waveform: np.ndarray, sr: int) -> np.ndarray:  #returns speech only waveform
    pcm_waveform = (waveform * 32768).astype(np.int16)
    vad = webrtcvad.Vad(1) # 0 --> least aggresive, 3 --> most aggresive
    frame_duration = 30
    frame_size = int(sr * frame_duration / 1000) # if sr = 16000 fraze_size = 480 samples
    
    speech_frames = []
    for i in range(0, len(pcm_waveform) - frame_size, frame_size):
        frame = pcm_waveform[i: i + frame_size]
        is_speech = vad.is_speech(frame.tobytes(), sr)

        if is_speech:
            speech_frames.append(frame)

    if len(speech_frames) == 0:
        return np.array([], dtype=np.float32)
    
    speech_audio = np.concatenate(speech_frames)
    return speech_audio.astype(np.float32) / 32768.0

