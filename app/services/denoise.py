import numpy as np
import noisereduce as nr
import logging
import librosa
import torch

logger = logging.getLogger(__name__)

# DeepFilterNet — load once at startup
_df_model = None
_df_state = None

def _load_deepfilter():
    global _df_model, _df_state
    if _df_model is None:
        try:
            from df.enhance import enhance, init_df
            _df_model, _df_state, _ = init_df()
            logger.info("DeepFilterNet3 loaded successfully")
        except Exception as e:
            logger.warning(f"DeepFilterNet3 failed to load: {e}. Falling back to noisereduce.")
    return _df_model, _df_state

# Load at import time
_load_deepfilter()


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

def _deepfilter_enhance(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    DeepFilterNet3 enhancement.
    Handles loud music and restaurant background noise.
    
    Pipeline:
    16kHz → upsample 48kHz → DeepFilterNet3 → downsample 16kHz
    
    Latency: ~5-8ms per chunk on CPU
    """
    from df.enhance import enhance

    # DeepFilterNet expects 48kHz
    audio_48k = librosa.resample(
        audio,
        orig_sr=sr,
        target_sr=48000,
        res_type='kaiser_fast'   # fast resampling — quality difference negligible
    )

    # Shape: (1, samples) — batch of 1
    audio_tensor = torch.from_numpy(audio_48k).unsqueeze(0)

    with torch.no_grad():
        enhanced_48k = enhance(_df_model, _df_state, audio_tensor)

    # Back to 16kHz
    enhanced_16k = librosa.resample(
        enhanced_48k.squeeze(0).numpy(),
        orig_sr=48000,
        target_sr=sr,
        res_type='kaiser_fast'
    )

    return enhanced_16k.astype(np.float32)


def enhance_audio(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Main entry point — use this everywhere in the pipeline.
    
    Strategy:
    1. Try DeepFilterNet3 (neural, handles loud music)
    2. Fallback to adaptive_denoise (statistical)
    3. Fallback to reduce_noise (basic)
    
    Future upgrade: swap internals of this function only.
    Rest of pipeline stays unchanged.
    """
    if len(audio) == 0:
        return audio

    # Try DeepFilterNet3
    if _df_model is not None:
        try:
            return _deepfilter_enhance(audio, sr)
        except Exception as e:
            logger.warning(f"DeepFilterNet3 failed: {e}. Using adaptive fallback.")

    # Fallback
    return adaptive_denoise(audio, sr)