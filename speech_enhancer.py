"""
Speech enhancement using DeepFilterNet3.

DeepFilterNet operates at 48 kHz.  Our ASR pipeline works at 16 kHz, so we
resample before enhancement and back again afterward.  The enhanced audio is
then blended with the original according to `mix_factor`:

    output = mix_factor * enhanced + (1 - mix_factor) * original

mix_factor = 1.0  →  fully enhanced  (default)
mix_factor = 0.0  →  original only   (effectively disabled)

The model is loaded lazily on the first call to enhance_audio() / _load_model().
is_deepfilter_available() uses importlib.util.find_spec() — no import, instant.
"""

import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Module-level singletons — loaded lazily on first call to enhance_audio()
_model = None
_df_state = None


def _patch_torchaudio_compat() -> None:
    """
    deepfilternet <=0.5.6 imports torchaudio.backend.common.AudioMetaData,
    which was removed in torchaudio 2.1+.  Inject a stub module so the
    import succeeds with any torchaudio version.
    """
    import sys
    import types
    try:
        from torchaudio.backend.common import AudioMetaData  # noqa: F401
        return  # already works — nothing to do
    except (ImportError, ModuleNotFoundError):
        pass

    import torchaudio
    AM = getattr(torchaudio, "AudioMetaData", type("AudioMetaData", (), {}))
    bm = types.ModuleType("torchaudio.backend")
    cm = types.ModuleType("torchaudio.backend.common")
    cm.AudioMetaData = AM
    bm.common = cm
    sys.modules.setdefault("torchaudio.backend", bm)
    sys.modules["torchaudio.backend.common"] = cm
    if not hasattr(torchaudio, "backend"):
        torchaudio.backend = bm


def is_deepfilter_available() -> bool:
    """
    Return True if deepfilternet is installed.

    Uses importlib.util.find_spec — no import, no Rust extension loading,
    executes in microseconds.  Safe to call at module import time.
    """
    import importlib.util
    return importlib.util.find_spec("df") is not None


def _load_model():
    """Load DeepFilterNet3 model (once, cached for the process lifetime)."""
    global _model, _df_state

    if _model is not None:
        return _model, _df_state

    _patch_torchaudio_compat()
    from df import init_df

    # DF_PRETRAINED_MODELS_PATH is set in the Dockerfile to /app/models so that
    # the model baked into the image is found without any network access.
    # The subdirectory name deepfilternet uses is "DeepFilterNet3".
    model_base = os.environ.get("DF_PRETRAINED_MODELS_PATH", "")
    candidate = os.path.join(model_base, "DeepFilterNet3") if model_base else ""
    init_arg = candidate if (candidate and os.path.isdir(candidate)) else "DeepFilterNet3"

    print(f"🔊 Loading DeepFilterNet3 from '{init_arg}' …")
    _model, _df_state, _ = init_df(init_arg, log_level="none")
    print(f"✅ DeepFilterNet3 loaded (sample rate: {_df_state.sr()} Hz)")
    return _model, _df_state


def enhance_audio(
    audio: np.ndarray,
    sr: int,
    mix_factor: float = 1.0,
) -> np.ndarray:
    """
    Denoise `audio` with DeepFilterNet3, then blend with the original.

    Parameters
    ----------
    audio : np.ndarray
        Mono float32 waveform at sample rate `sr`.
    sr : int
        Sample rate of `audio` (typically 16 000 Hz from the ASR pipeline).
    mix_factor : float
        0.0 = original unchanged, 1.0 = fully enhanced output.

    Returns
    -------
    np.ndarray
        Processed audio at the original sample rate `sr`, float32.
    """
    if mix_factor <= 0.0:
        return audio  # no-op

    import torch
    from scipy import signal as scipy_signal

    model, df_state = _load_model()
    target_sr: int = df_state.sr()  # 48 000 for DeepFilterNet3

    # ── Upsample to model's native sample rate ─────────────────────────
    if sr != target_sr:
        n_up = int(round(len(audio) * target_sr / sr))
        audio_up = scipy_signal.resample(audio, n_up).astype(np.float32)
    else:
        audio_up = audio.astype(np.float32)

    # ── Enhance ────────────────────────────────────────────────────────
    _patch_torchaudio_compat()
    from df import enhance as df_enhance

    # enhance() expects a (C, T) float32 tensor
    audio_tensor = torch.from_numpy(audio_up).unsqueeze(0)  # (1, T)
    enhanced_tensor = df_enhance(model, df_state, audio_tensor)
    enhanced_up = enhanced_tensor.squeeze(0).numpy().astype(np.float32)

    # ── Downsample back to original sample rate ────────────────────────
    if sr != target_sr:
        enhanced = scipy_signal.resample(enhanced_up, len(audio)).astype(np.float32)
    else:
        enhanced = enhanced_up

    # ── Blend with original ────────────────────────────────────────────
    if mix_factor >= 1.0:
        return enhanced
    return (mix_factor * enhanced + (1.0 - mix_factor) * audio).astype(np.float32)


def enhance_file(
    input_path: str,
    output_path: str,
    mix_factor: float = 1.0,
) -> str:
    """
    Load `input_path`, enhance, and write the result to `output_path`.

    Returns `output_path`.
    """
    import soundfile as sf

    audio, file_sr = sf.read(input_path, dtype="float32")

    # Ensure mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    enhanced = enhance_audio(audio, file_sr, mix_factor=mix_factor)

    sf.write(output_path, enhanced, file_sr)
    logger.info("✅ Speech enhancement saved: %s", output_path)
    return output_path
