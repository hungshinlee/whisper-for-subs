"""
Speech enhancement module — supports two backends:

  1. DeepFilterNet3  (default / existing)
     - Full-band 48 kHz neural noise suppression
     - Requires: pip install deepfilternet
     - Resamples 16 kHz ↔ 48 kHz internally

  2. DPDFNet  (new)
     - TFLite-based causal streaming models — extends DeepFilterNet2 with
       Dual-Path RNN blocks for stronger long-range and cross-band modeling
     - 16 kHz models: baseline / dpdfnet2 / dpdfnet4 / dpdfnet8
     - 48 kHz fullband model: dpdfnet2_48khz_hr
     - Requires: pip install tflite-runtime  (or tensorflow)
     - 16 kHz models work DIRECTLY with our pipeline — no resampling needed!
     - Paper: https://arxiv.org/abs/2512.16420
     - Models: https://huggingface.co/Ceva-IP/DPDFNet

Shared API
----------
Both backends expose the same top-level functions:

    enhance_audio(audio, sr, mix_factor, model_name) → np.ndarray
    enhance_file(input_path, output_path, mix_factor, model_name) → str
    is_model_available(model_name) → bool
    available_enhancement_models() → list[tuple[str, str]]

mix_factor (0.0–1.0):
    output = mix_factor × enhanced + (1 − mix_factor) × original
"""

import os
import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

# ── Backend constants ─────────────────────────────────────────────────────────

DEEPFILTERNET3 = "deepfilternet3"

DPDFNET_MODELS_16K = {
    "dpdfnet-baseline": "baseline.tflite",
    "dpdfnet-2":        "dpdfnet2.tflite",
    "dpdfnet-4":        "dpdfnet4.tflite",
    "dpdfnet-8":        "dpdfnet8.tflite",
}
DPDFNET_MODELS_48K = {
    "dpdfnet-2-48k": "dpdfnet2_48khz_hr.tflite",
}
DPDFNET_ALL_MODELS = {**DPDFNET_MODELS_16K, **DPDFNET_MODELS_48K}

# Directory where .tflite files are stored (configurable via env var)
DPDFNET_MODELS_DIR = os.environ.get("DPDFNET_MODELS_DIR", "/app/models/dpdfnet")

# ── DeepFilterNet3 backend ────────────────────────────────────────────────────

_df3_model    = None
_df3_state    = None


def _patch_torchaudio_compat() -> None:
    """
    deepfilternet <=0.5.6 imports torchaudio.backend.common.AudioMetaData,
    which was removed in torchaudio 2.1+.  Inject a stub so the import succeeds.
    """
    import sys
    import types
    try:
        from torchaudio.backend.common import AudioMetaData  # noqa: F401
        return
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
    """Return True if deepfilternet is installed (no import cost)."""
    import importlib.util
    return importlib.util.find_spec("df") is not None


def _load_df3_model():
    """Load DeepFilterNet3 (once, cached for process lifetime)."""
    global _df3_model, _df3_state
    if _df3_model is not None:
        return _df3_model, _df3_state

    _patch_torchaudio_compat()
    from df import init_df

    model_base = os.environ.get("DF_PRETRAINED_MODELS_PATH", "")
    candidate  = os.path.join(model_base, "DeepFilterNet3") if model_base else ""
    init_arg   = candidate if (candidate and os.path.isdir(candidate)) else "DeepFilterNet3"

    print(f"🔊 Loading DeepFilterNet3 from '{init_arg}' …")
    _df3_model, _df3_state, _ = init_df(init_arg, log_level="none")
    print(f"✅ DeepFilterNet3 loaded (sample rate: {_df3_state.sr()} Hz)")
    return _df3_model, _df3_state


def _enhance_audio_df3(
    audio: np.ndarray,
    sr: int,
    mix_factor: float = 1.0,
) -> np.ndarray:
    """Denoise with DeepFilterNet3 (resamples 16 kHz ↔ 48 kHz internally)."""
    if mix_factor <= 0.0:
        return audio

    import torch
    from scipy import signal as scipy_signal

    model, df_state = _load_df3_model()
    target_sr: int = df_state.sr()  # 48 000

    if sr != target_sr:
        n_up    = int(round(len(audio) * target_sr / sr))
        audio_up = scipy_signal.resample(audio, n_up).astype(np.float32)
    else:
        audio_up = audio.astype(np.float32)

    _patch_torchaudio_compat()
    from df import enhance as df_enhance

    audio_tensor    = torch.from_numpy(audio_up).unsqueeze(0)  # (1, T)
    enhanced_tensor = df_enhance(model, df_state, audio_tensor)
    enhanced_up     = enhanced_tensor.squeeze(0).numpy().astype(np.float32)

    if sr != target_sr:
        enhanced = scipy_signal.resample(enhanced_up, len(audio)).astype(np.float32)
    else:
        enhanced = enhanced_up

    if mix_factor >= 1.0:
        return enhanced
    return (mix_factor * enhanced + (1.0 - mix_factor) * audio).astype(np.float32)


# ── DPDFNet backend ───────────────────────────────────────────────────────────

# Cache: model_name → tflite Interpreter
_dpdfnet_interpreters: dict = {}


def is_tflite_available() -> bool:
    """Return True if tflite-runtime (or tensorflow) is importable."""
    import importlib.util
    return (
        importlib.util.find_spec("tflite_runtime") is not None
        or importlib.util.find_spec("tensorflow") is not None
    )


def _tflite_interpreter(model_path: str):
    """Load a TFLite Interpreter from a .tflite file."""
    try:
        import tflite_runtime.interpreter as tflite
        interp = tflite.Interpreter(model_path=model_path)
    except ImportError:
        import tensorflow as tf
        interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    return interp


def _load_dpdfnet_interpreter(model_name: str):
    """Load and cache a DPDFNet TFLite interpreter."""
    if model_name in _dpdfnet_interpreters:
        return _dpdfnet_interpreters[model_name]

    filename = DPDFNET_ALL_MODELS.get(model_name)
    if filename is None:
        raise ValueError(f"Unknown DPDFNet model: {model_name!r}")

    model_path = os.path.join(DPDFNET_MODELS_DIR, filename)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"DPDFNet model file not found: {model_path}\n"
            f"  Download it with:\n"
            f"    huggingface-cli download Ceva-IP/DPDFNet {filename} "
            f"--local-dir {DPDFNET_MODELS_DIR} --local-dir-use-symlinks False"
        )

    print(f"🔊 Loading DPDFNet '{model_name}' from {model_path} …")
    interp = _tflite_interpreter(model_path)
    _dpdfnet_interpreters[model_name] = interp
    in_shape = interp.get_input_details()[0]["shape"]
    print(f"✅ DPDFNet '{model_name}' loaded (input shape: {in_shape})")
    return interp


def is_dpdfnet_model_available(model_name: str) -> bool:
    """
    Return True if the given DPDFNet model can be used:
      - tflite-runtime or tensorflow is installed
      - The .tflite file exists in DPDFNET_MODELS_DIR
    """
    if not is_tflite_available():
        return False
    filename = DPDFNET_ALL_MODELS.get(model_name)
    if filename is None:
        return False
    return os.path.isfile(os.path.join(DPDFNET_MODELS_DIR, filename))


def _enhance_audio_dpdfnet(
    audio: np.ndarray,
    sr: int,
    model_name: str,
    mix_factor: float = 1.0,
) -> np.ndarray:
    """
    Denoise with a DPDFNet TFLite model.

    16 kHz models (baseline/dpdfnet2/dpdfnet4/dpdfnet8) work directly with
    our 16 kHz pipeline — no resampling.

    The 48 kHz model (dpdfnet2-48k) resamples 16 kHz → 48 kHz → 16 kHz.

    Frame-by-frame streaming inference with RNN state reset per file.
    """
    if mix_factor <= 0.0:
        return audio

    interp = _load_dpdfnet_interpreter(model_name)
    in_details  = interp.get_input_details()
    out_details = interp.get_output_details()

    # Determine whether to resample
    is_48k_model = model_name in DPDFNET_MODELS_48K
    target_sr    = 48000 if is_48k_model else 16000

    if sr != target_sr:
        from scipy import signal as scipy_signal
        n_target   = int(round(len(audio) * target_sr / sr))
        proc_audio = scipy_signal.resample(audio, n_target).astype(np.float32)
    else:
        proc_audio = audio.astype(np.float32)

    # Auto-detect frame size from model input tensor
    in_shape   = in_details[0]["shape"]   # e.g. [1, 160] or [1, 480]
    frame_size = int(in_shape[-1])
    out_shape  = out_details[0]["shape"]

    # Pad to a multiple of frame_size
    orig_len     = len(proc_audio)
    remainder    = orig_len % frame_size
    padded_audio = (
        np.concatenate([proc_audio, np.zeros(frame_size - remainder, dtype=np.float32)])
        if remainder else proc_audio
    )

    # Reset RNN state (stateful model — state persists between invoke() calls
    # within the same file; must be reset at the start of each new file)
    interp.reset_all_variables()

    output_frames = []
    for i in range(0, len(padded_audio), frame_size):
        frame = padded_audio[i : i + frame_size].reshape(in_shape).astype(np.float32)
        interp.set_tensor(in_details[0]["index"], frame)
        interp.invoke()
        out = interp.get_tensor(out_details[0]["index"]).flatten().copy()
        output_frames.append(out)

    enhanced_proc = np.concatenate(output_frames)[:orig_len]

    # Resample back if needed
    if sr != target_sr:
        from scipy import signal as scipy_signal
        enhanced = scipy_signal.resample(enhanced_proc, len(audio)).astype(np.float32)
    else:
        enhanced = enhanced_proc

    if mix_factor >= 1.0:
        return enhanced
    return (mix_factor * enhanced + (1.0 - mix_factor) * audio).astype(np.float32)


# ── Unified public API ────────────────────────────────────────────────────────

def is_model_available(model_name: str) -> bool:
    """Return True if the given enhancement model can be used."""
    if model_name == DEEPFILTERNET3:
        return is_deepfilter_available()
    return is_dpdfnet_model_available(model_name)


def available_enhancement_models() -> list:
    """
    Return a list of (label, value) tuples for every enhancement model
    that is actually usable in the current environment.

    Intended for building the Gradio dropdown choices.
    """
    models = []

    if is_deepfilter_available():
        models.append(("DeepFilterNet3 (48 kHz)", DEEPFILTERNET3))

    dpdfnet_labels = {
        "dpdfnet-baseline": "DPDFNet Baseline (16 kHz · fastest)",
        "dpdfnet-2":        "DPDFNet-2 (16 kHz · balanced)",
        "dpdfnet-4":        "DPDFNet-4 (16 kHz · high quality)",
        "dpdfnet-8":        "DPDFNet-8 (16 kHz · best quality)",
        "dpdfnet-2-48k":    "DPDFNet-2 HR (48 kHz · fullband)",
    }
    for key, label in dpdfnet_labels.items():
        if is_dpdfnet_model_available(key):
            models.append((label, key))

    return models


def enhance_audio(
    audio: np.ndarray,
    sr: int,
    mix_factor: float = 1.0,
    model_name: str = DEEPFILTERNET3,
) -> np.ndarray:
    """
    Denoise audio with the chosen enhancement backend.

    Parameters
    ----------
    audio      : Mono float32 waveform at sample rate `sr`.
    sr         : Sample rate of `audio` (typically 16 000 Hz).
    mix_factor : 0.0 = original, 1.0 = fully enhanced.
    model_name : One of DEEPFILTERNET3 or any key in DPDFNET_ALL_MODELS.

    Returns
    -------
    np.ndarray — processed audio at the same sample rate and length.
    """
    if mix_factor <= 0.0:
        return audio

    if model_name == DEEPFILTERNET3:
        return _enhance_audio_df3(audio, sr, mix_factor)

    if model_name in DPDFNET_ALL_MODELS:
        return _enhance_audio_dpdfnet(audio, sr, model_name, mix_factor)

    raise ValueError(
        f"Unknown enhancement model: {model_name!r}\n"
        f"Valid values: {DEEPFILTERNET3!r}, {list(DPDFNET_ALL_MODELS)}"
    )


def enhance_file(
    input_path: str,
    output_path: str,
    mix_factor: float = 1.0,
    model_name: str = DEEPFILTERNET3,
) -> str:
    """
    Load `input_path`, enhance with the chosen model, write to `output_path`.

    Returns `output_path`.
    """
    import soundfile as sf

    audio, file_sr = sf.read(input_path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    enhanced = enhance_audio(audio, file_sr, mix_factor=mix_factor, model_name=model_name)

    sf.write(output_path, enhanced, file_sr)
    logger.info("✅ Speech enhancement (%s) saved: %s", model_name, output_path)
    return output_path
