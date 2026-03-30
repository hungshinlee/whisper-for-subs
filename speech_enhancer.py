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
     - Paper: https://arxiv.org/abs/2512.16420
     - Models: https://huggingface.co/Ceva-IP/DPDFNet

DPDFNet input/output format
----------------------------
The TFLite models are **frequency-domain** (STFT-based), NOT time-domain.
Input shape: [1, 1, n_fft//2+1, 2]  — one STFT frame, complex (real+imag)
Output shape: same as input.

STFT parameters are derived from the model's freq_bins dimension:
    n_fft    = (freq_bins - 1) * 2
    hop_size = n_fft // 2
    window   = Hann

Pipeline per file:
    audio  →  resample (if needed)
           →  STFT  →  frame loop (stateful TFLite invoke)  →  iSTFT
           →  resample back (if needed)
           →  mix with original

Shared API
----------
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

DPDFNET_MODELS_DIR = os.environ.get("DPDFNET_MODELS_DIR", "/app/models/dpdfnet")


# ── DeepFilterNet3 backend ────────────────────────────────────────────────────

_df3_model = None
_df3_state = None


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
    import importlib.util
    return importlib.util.find_spec("df") is not None


def _load_df3_model():
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


def _enhance_audio_df3(audio: np.ndarray, sr: int, mix_factor: float = 1.0) -> np.ndarray:
    if mix_factor <= 0.0:
        return audio
    import torch
    from scipy import signal as scipy_signal
    model, df_state = _load_df3_model()
    target_sr = df_state.sr()
    if sr != target_sr:
        audio_up = scipy_signal.resample(audio, int(round(len(audio) * target_sr / sr))).astype(np.float32)
    else:
        audio_up = audio.astype(np.float32)
    _patch_torchaudio_compat()
    from df import enhance as df_enhance
    enhanced_up = df_enhance(model, df_state, torch.from_numpy(audio_up).unsqueeze(0)).squeeze(0).numpy().astype(np.float32)
    enhanced = scipy_signal.resample(enhanced_up, len(audio)).astype(np.float32) if sr != target_sr else enhanced_up
    return enhanced if mix_factor >= 1.0 else (mix_factor * enhanced + (1.0 - mix_factor) * audio).astype(np.float32)


# ── DPDFNet backend ───────────────────────────────────────────────────────────

_dpdfnet_interpreters: dict = {}


def is_tflite_available() -> bool:
    import importlib.util
    return (importlib.util.find_spec("tflite_runtime") is not None
            or importlib.util.find_spec("tensorflow") is not None)


def _tflite_interpreter(model_path: str):
    try:
        import tflite_runtime.interpreter as tflite
        interp = tflite.Interpreter(model_path=model_path)
    except ImportError:
        import tensorflow as tf
        interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    return interp


def _load_dpdfnet_interpreter(model_name: str):
    if model_name in _dpdfnet_interpreters:
        return _dpdfnet_interpreters[model_name]
    filename = DPDFNET_ALL_MODELS.get(model_name)
    if filename is None:
        raise ValueError(f"Unknown DPDFNet model: {model_name!r}")
    model_path = os.path.join(DPDFNET_MODELS_DIR, filename)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"DPDFNet model file not found: {model_path}\n"
            f"  Download: huggingface-cli download Ceva-IP/DPDFNet {filename} "
            f"--local-dir {DPDFNET_MODELS_DIR} --local-dir-use-symlinks False"
        )
    print(f"🔊 Loading DPDFNet '{model_name}' from {model_path} …")
    interp = _tflite_interpreter(model_path)
    _dpdfnet_interpreters[model_name] = interp
    in_shape = interp.get_input_details()[0]["shape"]
    print(f"✅ DPDFNet '{model_name}' loaded (input shape: {list(in_shape)})")
    return interp


def is_dpdfnet_model_available(model_name: str) -> bool:
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
    Frequency-domain (STFT-based) frame-by-frame inference with DPDFNet.

    The TFLite models are NOT time-domain: they consume one STFT frame at a
    time and output the enhanced complex spectrum for that frame.

    Input tensor shape: [1, 1, freq_bins, 2]   (batch, ch, bins, real/imag)
    Output tensor shape: same.

    STFT parameters are derived from freq_bins:
        n_fft    = (freq_bins - 1) * 2   e.g. 320 for 16 kHz models (161 bins)
        hop_size = n_fft // 2            e.g. 160
        window   = Hann

    The model is stateful (RNN layers).  reset_all_variables() resets state at
    the start of each file so batch calls remain independent.
    """
    if mix_factor <= 0.0:
        return audio

    from scipy import signal as scipy_signal

    interp      = _load_dpdfnet_interpreter(model_name)
    in_details  = interp.get_input_details()
    out_details = interp.get_output_details()

    # ── Derive STFT parameters from model input shape ─────────────────────
    # Expected: [batch=1, ch=1, freq_bins, 2]
    in_shape  = in_details[0]["shape"]   # e.g. [1, 1, 161, 2]
    freq_bins = int(in_shape[2])
    n_fft     = (freq_bins - 1) * 2     # 320 for 16kHz, 960 for 48kHz HR
    hop_size  = n_fft // 2              # 160 / 480

    # ── Resample to model's target SR if necessary ────────────────────────
    is_48k_model = model_name in DPDFNET_MODELS_48K
    target_sr    = 48000 if is_48k_model else 16000

    if sr != target_sr:
        n_target   = int(round(len(audio) * target_sr / sr))
        proc_audio = scipy_signal.resample(audio, n_target).astype(np.float32)
    else:
        proc_audio = audio.astype(np.float32)

    orig_proc_len = len(proc_audio)

    # ── STFT ─────────────────────────────────────────────────────────────
    # Use a Hann window; boundary='zeros' avoids edge padding so the number of
    # output frames is predictable and matches ISTFT exactly.
    window = np.hanning(n_fft).astype(np.float32)

    freqs, times, Zxx = scipy_signal.stft(
        proc_audio,
        fs=target_sr,
        window=window,
        nperseg=n_fft,
        noverlap=n_fft - hop_size,
        boundary=None,
        padded=True,
    )
    # Zxx: complex128, shape (freq_bins, n_frames)
    n_frames = Zxx.shape[1]

    # ── Frame-by-frame inference ─────────────────────────────────────────
    interp.reset_all_variables()   # reset RNN state for this file

    enhanced_frames = np.zeros_like(Zxx)   # complex

    for t in range(n_frames):
        real = Zxx[:, t].real.astype(np.float32)
        imag = Zxx[:, t].imag.astype(np.float32)

        # shape → [1, 1, freq_bins, 2]
        frame_in = np.stack([real, imag], axis=-1)[np.newaxis, np.newaxis, :, :]

        interp.set_tensor(in_details[0]["index"], frame_in)
        interp.invoke()

        out = interp.get_tensor(out_details[0]["index"])   # [1, 1, freq_bins, 2]
        enhanced_frames[:, t] = out[0, 0, :, 0] + 1j * out[0, 0, :, 1]

    # ── iSTFT ─────────────────────────────────────────────────────────────
    _, enhanced_proc = scipy_signal.istft(
        enhanced_frames,
        fs=target_sr,
        window=window,
        nperseg=n_fft,
        noverlap=n_fft - hop_size,
        boundary=None,
    )
    enhanced_proc = enhanced_proc.real.astype(np.float32)

    # Trim / pad to match the processed-audio length exactly
    if len(enhanced_proc) > orig_proc_len:
        enhanced_proc = enhanced_proc[:orig_proc_len]
    elif len(enhanced_proc) < orig_proc_len:
        enhanced_proc = np.pad(enhanced_proc, (0, orig_proc_len - len(enhanced_proc)))

    # ── Resample back to original SR ─────────────────────────────────────
    if sr != target_sr:
        enhanced = scipy_signal.resample(enhanced_proc, len(audio)).astype(np.float32)
    else:
        enhanced = enhanced_proc

    print(f"✅ DPDFNet '{model_name}' inference done "
          f"({n_frames} frames, n_fft={n_fft}, hop={hop_size})")

    if mix_factor >= 1.0:
        return enhanced
    return (mix_factor * enhanced + (1.0 - mix_factor) * audio).astype(np.float32)


# ── Unified public API ────────────────────────────────────────────────────────

def is_model_available(model_name: str) -> bool:
    if model_name == DEEPFILTERNET3:
        return is_deepfilter_available()
    return is_dpdfnet_model_available(model_name)


def available_enhancement_models() -> list:
    """
    Return a list of (label, value) tuples for every enhancement model
    that is actually usable in the current environment.
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
    model_name : DEEPFILTERNET3 or any key in DPDFNET_ALL_MODELS.
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
    """Load `input_path`, enhance, write to `output_path`. Returns `output_path`."""
    import soundfile as sf
    audio, file_sr = sf.read(input_path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    enhanced = enhance_audio(audio, file_sr, mix_factor=mix_factor, model_name=model_name)
    sf.write(output_path, enhanced, file_sr)
    logger.info("✅ Speech enhancement (%s) saved: %s", model_name, output_path)
    return output_path
