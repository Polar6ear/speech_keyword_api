import webrtcvad
import numpy as np


def apply_vad(waveform: np.ndarray, sr: int) -> np.ndarray:
    """
    Filter non-speech frames using WebRTC VAD.

    Processes audio in 30ms frames. Aggressiveness level 1 (0=least, 3=most)
    is used to avoid cutting off soft speech at the cost of retaining
    some non-speech frames.

    Returns a float32 waveform containing only speech frames.
    """
    pcm_waveform = (waveform * 32768).astype(np.int16)
    vad = webrtcvad.Vad(1)
    frame_duration = 30
    frame_size = int(sr * frame_duration / 1000)  # 480 samples at 16kHz

    speech_frames = []
    for i in range(0, len(pcm_waveform) - frame_size, frame_size):
        frame = pcm_waveform[i: i + frame_size]
        if vad.is_speech(frame.tobytes(), sr):
            speech_frames.append(frame)

    if len(speech_frames) == 0:
        return np.array([], dtype=np.float32)

    speech_audio = np.concatenate(speech_frames)
    return speech_audio.astype(np.float32) / 32768.0
