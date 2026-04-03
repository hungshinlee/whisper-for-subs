"""
Gradio-based web interface for Whisper ASR service.

Supports two modes (selectable via the Mode radio at the top of Settings):

  🎙️ ASR (Transcribe)  — full pipeline: enhance → VAD → Whisper → SRT
  🔊 Enhance Only      — speech enhancement only; outputs enhanced WAV
                          + a time-aligned spectrogram image
"""

import os
import glob
import tempfile
import time
import shutil
import uuid
from datetime import datetime, timedelta
from typing import Optional, Tuple, Generator, Dict
from threading import Lock, Semaphore

import gradio as gr
import soundfile as sf
import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse

from transcriber import (
    WhisperTranscriber,
    SUPPORTED_LANGUAGES,
    MODEL_SIZES,
    MODEL_CONFIGS,
)
from srt_utils import segments_to_srt, merge_segments
from chinese_converter import convert_segments_to_traditional, get_converter
from hakka_translator import (
    translate_segments,
    is_llm_enabled,
    check_vllm_ready,
    load_lexicon,
    DEFAULT_SYSTEM_PROMPT,
)

# Pre-load lexicon at startup (cheap, ~ms)
_HAKKA_LEXICON = load_lexicon()
import numpy as np
from vad import SileroVAD
from speech_enhancer import (
    DEEPFILTERNET3,
    available_enhancement_models,
    enhance_file,
    is_model_available,
)

# ── Enhancement model registry ────────────────────────────────────────────────
ENHANCEMENT_MODEL_CHOICES: list = available_enhancement_models()
SPEECH_ENHANCEMENT_AVAILABLE: bool = len(ENHANCEMENT_MODEL_CHOICES) > 0

_DEFAULT_ENHANCEMENT_MODEL: str = (
    DEEPFILTERNET3
    if is_model_available(DEEPFILTERNET3)
    else (ENHANCEMENT_MODEL_CHOICES[0][1] if ENHANCEMENT_MODEL_CHOICES else DEEPFILTERNET3)
)

LLM_ENABLED = is_llm_enabled()

# ── Whisper model ID lists ────────────────────────────────────────────────────
GENERAL_MODELS_IDS: list = [
    m for m, cfg in MODEL_CONFIGS.items() if cfg["label"] == "General"
]
HAKKA_MODELS_IDS: list = [
    m for m, cfg in MODEL_CONFIGS.items() if cfg["label"] == "Hakka"
]
TAIGI_MODELS_IDS: list = [
    m for m, cfg in MODEL_CONFIGS.items() if cfg["label"] == "Taigi"
]

# ── Operating modes ───────────────────────────────────────────────────────────
MODE_ASR          = "asr"
MODE_ENHANCE_ONLY = "enhance_only"


# Custom CSS with Roboto font
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');

* { font-family: 'Roboto', sans-serif !important; }
.gradio-container { font-family: 'Roboto', sans-serif !important; }
.prose { font-family: 'Roboto', sans-serif !important; }
textarea, input, button, select { font-family: 'Roboto', sans-serif !important; }
.progress-bar-container { margin: 10px 0; }
.copy-button { margin-top: 10px; }
.copy-success { color: #4CAF50; font-weight: 500; margin-top: 5px; }
"""


# ═════════════════════════════════════════════════════════════════════════════
# TranscriberPool
# ═════════════════════════════════════════════════════════════════════════════

class TranscriberPool:
    """
    Thread-safe pool that distributes single-GPU transcribers across all
    available GPUs, so concurrent requests use GPU 0, 1, 2 … instead of
    queuing on GPU 0.

    GPU selection strategy (in priority order)
    ------------------------------------------
    1. Cached model + idle GPU  — best case: zero wait, no model reload.
    2. Any completely free GPU  — load model there, start immediately.
    3. Cached model on busy GPU — queue behind it (avoids reloading weights).
    4. Least-loaded GPU overall — last resort, load model and queue.

    The semaphore is acquired OUTSIDE self.lock to avoid deadlocks.
    """

    def __init__(self):
        self.lock = Lock()

        _gpu_str = os.environ.get(
            "SINGLE_GPU_DEVICES",
            os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        )
        if _gpu_str:
            self.single_gpu_ids = [int(x.strip()) for x in _gpu_str.split(",") if x.strip()]
        elif torch.cuda.is_available():
            self.single_gpu_ids = list(range(torch.cuda.device_count()))
        else:
            self.single_gpu_ids = []

        print(f"🎛️  TranscriberPool: GPU slots {self.single_gpu_ids}")

        self.single_gpu_pool: Dict[int, WhisperTranscriber] = {}
        self.gpu_active: Dict[int, int] = {g: 0 for g in self.single_gpu_ids}
        self.gpu_semaphores: Dict = {g: Semaphore(1) for g in self.single_gpu_ids}
        self.gpu_semaphores["cpu"] = Semaphore(1)

    def _least_loaded_gpu(self) -> int:
        return min(self.single_gpu_ids, key=lambda g: self.gpu_active.get(g, 0))

    def get_single_gpu_transcriber(
        self,
        model_size: str,
        use_vad: bool,
        min_silence_duration_s: float,
    ) -> Tuple[WhisperTranscriber, int]:
        min_silence_duration_ms = int(min_silence_duration_s * 1000)
        device = os.environ.get("WHISPER_DEVICE", "cuda")

        with self.lock:
            if not self.single_gpu_ids:
                if "cpu" not in self.single_gpu_pool:
                    self.single_gpu_pool["cpu"] = WhisperTranscriber(
                        model_size=model_size,
                        device="cpu",
                        compute_type="float32",
                        use_vad=use_vad,
                        min_silence_duration_ms=min_silence_duration_ms,
                    )
                gpu_id = "cpu"
                trans = self.single_gpu_pool["cpu"]
                self.gpu_active["cpu"] = self.gpu_active.get("cpu", 0) + 1
            else:
                # Priority 1: cached model + idle GPU
                idle_cached_gpu = None
                for gid, t in self.single_gpu_pool.items():
                    if t.model_size == model_size:
                        sem = self.gpu_semaphores.get(gid)
                        if sem is not None and sem.acquire(blocking=False):
                            sem.release()
                            idle_cached_gpu = gid
                            break

                if idle_cached_gpu is not None:
                    gpu_id = idle_cached_gpu
                    trans = self.single_gpu_pool[gpu_id]
                    self.gpu_active[gpu_id] += 1
                    print(f"♛️  Reusing idle GPU {gpu_id} "
                          f"(queued={self.gpu_active[gpu_id]}, model={model_size})")
                else:
                    # Priority 2: any completely free GPU
                    free_gpu = next(
                        (gid for gid in self.single_gpu_ids
                         if self.gpu_active.get(gid, 0) == 0),
                        None,
                    )
                    if free_gpu is not None:
                        gpu_id = free_gpu
                        if gpu_id in self.single_gpu_pool:
                            old = self.single_gpu_pool[gpu_id].model_size
                            print(f"🔄 GPU {gpu_id}: replacing {old} → {model_size}")
                            del self.single_gpu_pool[gpu_id]
                        print(f"✨ Loading {model_size} on free GPU {gpu_id}")
                        trans = WhisperTranscriber(
                            model_size=model_size,
                            device=device,
                            device_index=gpu_id,
                            compute_type=os.environ.get("WHISPER_COMPUTE_TYPE", "float16"),
                            use_vad=use_vad,
                            min_silence_duration_ms=min_silence_duration_ms,
                        )
                        self.single_gpu_pool[gpu_id] = trans
                        self.gpu_active[gpu_id] += 1
                        print(f"✅ GPU {gpu_id} ready (queued={self.gpu_active[gpu_id]})")
                    else:
                        # Priority 3: cached model on a busy GPU
                        busy_cached_gpu = min(
                            (gid for gid, t in self.single_gpu_pool.items()
                             if t.model_size == model_size),
                            key=lambda gid: self.gpu_active.get(gid, 0),
                            default=None,
                        )
                        if busy_cached_gpu is not None:
                            gpu_id = busy_cached_gpu
                            trans = self.single_gpu_pool[gpu_id]
                            self.gpu_active[gpu_id] += 1
                            print(f"⏳ Queuing on cached GPU {gpu_id} "
                                  f"(queued={self.gpu_active[gpu_id]}, model={model_size})")
                        else:
                            # Priority 4: least-loaded GPU
                            gpu_id = self._least_loaded_gpu()
                            if gpu_id in self.single_gpu_pool:
                                old = self.single_gpu_pool[gpu_id].model_size
                                print(f"🔄 GPU {gpu_id}: replacing {old} → {model_size}")
                                del self.single_gpu_pool[gpu_id]
                            print(f"✨ Loading {model_size} on least-loaded GPU {gpu_id}")
                            trans = WhisperTranscriber(
                                model_size=model_size,
                                device=device,
                                device_index=gpu_id,
                                compute_type=os.environ.get("WHISPER_COMPUTE_TYPE", "float16"),
                                use_vad=use_vad,
                                min_silence_duration_ms=min_silence_duration_ms,
                            )
                            self.single_gpu_pool[gpu_id] = trans
                            self.gpu_active[gpu_id] += 1
                            print(f"✅ GPU {gpu_id} ready (queued={self.gpu_active[gpu_id]})")

        sem = self.gpu_semaphores.get(gpu_id)
        if sem is not None:
            if not sem.acquire(blocking=False):
                print(f"⏳ GPU {gpu_id} semaphore busy — waiting "
                      f"(queued={self.gpu_active.get(gpu_id, '?')})")
                sem.acquire()
                print(f"▶️  GPU {gpu_id} acquired, starting transcription")

        return trans, gpu_id

    def release_single_gpu_transcriber(self, gpu_id):
        with self.lock:
            if gpu_id in self.gpu_active and self.gpu_active[gpu_id] > 0:
                self.gpu_active[gpu_id] -= 1
                print(f"✅ Released GPU {gpu_id} (queued={self.gpu_active[gpu_id]})")
        sem = self.gpu_semaphores.get(gpu_id)
        if sem is not None:
            sem.release()


transcriber_pool = TranscriberPool()


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def cleanup_old_files(max_age_hours: int = 24):
    now = datetime.now()
    for tmp_dir in ["/tmp/whisper-sessions"]:
        if not os.path.exists(tmp_dir):
            continue
        for f in glob.glob(os.path.join(tmp_dir, "*")):
            try:
                if now - datetime.fromtimestamp(os.path.getmtime(f)) > timedelta(hours=max_age_hours):
                    (os.unlink if os.path.isfile(f) else shutil.rmtree)(f)
            except Exception:
                pass
    output_dir = "/app/outputs"
    if os.path.exists(output_dir):
        for f in (glob.glob(os.path.join(output_dir, "*.srt"))
                  + glob.glob(os.path.join(output_dir, "*.wav"))
                  + glob.glob(os.path.join(output_dir, "*.png"))):
            try:
                if now - datetime.fromtimestamp(os.path.getmtime(f)) > timedelta(hours=max_age_hours):
                    os.unlink(f)
            except Exception:
                pass


def format_progress_html(percent: int, message: str) -> str:
    return f"""
<div class="progress-bar-container">
    <div style="margin-bottom:5px;font-weight:500;">{message}</div>
    <div style="background-color:#e0e0e0;border-radius:10px;height:20px;overflow:hidden;">
        <div style="background:linear-gradient(90deg,#2196F3,#21CBF3);height:100%;width:{percent}%;
                    transition:width 0.3s ease;border-radius:10px;"></div>
    </div>
    <div style="text-align:right;font-size:12px;color:#666;margin-top:3px;">{percent}%</div>
</div>"""


_SF_UNSUPPORTED_EXTS = {'.mp3', '.aac', '.m4a', '.m4b', '.opus', '.wma', '.amr', '.3gp', '.3gpp'}


def _ensure_wav(src_path: str, session_dir: str) -> str:
    ext = os.path.splitext(src_path)[1].lower()
    if ext not in _SF_UNSUPPORTED_EXTS:
        return src_path
    wav_path = os.path.join(session_dir, f"converted_{uuid.uuid4().hex[:8]}.wav")
    print(f"🔄 Converting {ext} → WAV: {wav_path}")
    import subprocess
    subprocess.run(
        ["ffmpeg", "-y", "-i", src_path, "-ar", "16000", "-ac", "1", wav_path],
        capture_output=True, check=True,
    )
    print("✅ Conversion complete")
    return wav_path


def _save_srt(srt_content: str, safe_title: str, suffix: str, output_dir: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:6]
    path = os.path.join(output_dir, f"{safe_title}_{suffix}_{timestamp}_{unique_id}.srt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(srt_content)
    return path


def _safe_title(video_title: str) -> str:
    return "".join(c for c in video_title if c.isalnum() or c in " -_").strip()[:40]


def _get_output_dir() -> str:
    out = "/app/outputs" if os.path.exists("/app/outputs") else tempfile.gettempdir()
    os.makedirs(out, exist_ok=True)
    return out


def _generate_spectrogram_png(
    audio: np.ndarray,
    sr: int,
    output_path: str,
    title: str = "Spectrogram",
) -> str:
    """
    Compute STFT, render a log-power spectrogram with a time axis in seconds
    (matching the audio player timeline), and save to output_path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import signal as scipy_signal

    # STFT — 25 ms window, 10 ms hop → good time resolution for speech
    nperseg  = int(sr * 0.025)
    noverlap = int(sr * 0.015)

    freqs, times, Zxx = scipy_signal.stft(
        audio, fs=sr, nperseg=nperseg, noverlap=noverlap
    )
    power_db = 20 * np.log10(np.maximum(np.abs(Zxx), 1e-10))

    # Cap frequency axis at 8 kHz for speech readability
    freq_limit_hz = min(8000, freqs[-1])
    freq_mask     = freqs <= freq_limit_hz
    freqs_plot    = freqs[freq_mask]
    power_plot    = power_db[freq_mask, :]

    vmax = power_plot.max()
    vmin = vmax - 80  # 80 dB dynamic range

    duration_s = len(audio) / sr
    fig_width  = max(10, min(20, duration_s * 0.6))
    fig, ax    = plt.subplots(figsize=(fig_width, 3.5), dpi=120)

    pcm = ax.pcolormesh(
        times,
        freqs_plot / 1000,   # Hz → kHz
        power_plot,
        cmap="inferno",
        shading="gouraud",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("Frequency (kHz)", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_xlim(0, times[-1])
    ax.set_ylim(0, freq_limit_hz / 1000)

    cbar = fig.colorbar(pcm, ax=ax, pad=0.01)
    cbar.set_label("dB", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.tick_params(labelsize=9)
    fig.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 Spectrogram saved: {output_path}")
    return output_path


# ═════════════════════════════════════════════════════════════════════════════
# Shared yield-tuple layout (9 values)
#
#  0  status_html          str / HTML
#  1  asr_srt_text         str
#  2  asr_srt_file         str | None
#  3  translated_col       gr.update (visible)
#  4  translated_srt_text  str
#  5  translated_srt_file  str | None
#  6  enhance_only_col     gr.update (visible)
#  7  enhanced_audio       str | None  (WAV path for gr.Audio)
#  8  spectrogram_image    str | None  (PNG path for gr.Image)
# ═════════════════════════════════════════════════════════════════════════════

_NO_TRANSLATE = (gr.update(visible=False), "", None)
_NO_ENHANCE   = (gr.update(visible=False), None, None)


def _prog_asr(pct, msg):
    return format_progress_html(pct, msg), "", None, *_NO_TRANSLATE, *_NO_ENHANCE


def _prog_enh(pct, msg):
    return format_progress_html(pct, msg), "", None, *_NO_TRANSLATE, *_NO_ENHANCE


# ─────────────────────────────────────────────────────────────────────────────
# Input preparation — shared by both modes
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_input(audio_file, prog_fn):
    """
    Resolve uploaded audio file → local WAV path.
    Yields progress tuples, then a 3-tuple sentinel (audio_path, title, temp_files).
    """
    temp_files  = []
    video_title = "output"

    if audio_file:
        ext         = os.path.splitext(audio_file)[1]
        upload_copy = os.path.join(tempfile.mkdtemp(), f"upload_{uuid.uuid4().hex[:8]}{ext}")
        shutil.copy2(audio_file, upload_copy)
        temp_files.append(upload_copy)
        video_title = os.path.splitext(os.path.basename(audio_file))[0]
        audio_path  = _ensure_wav(upload_copy, os.path.dirname(upload_copy))
        if audio_path != upload_copy:
            temp_files.append(audio_path)
        yield prog_fn(10, "Audio file loaded")
        print(f"📁 Session audio: {audio_path}")
    else:
        yield "❌ Please upload an audio file", "", None, *_NO_TRANSLATE, *_NO_ENHANCE
        return

    yield (audio_path, video_title, temp_files)   # sentinel


# ═════════════════════════════════════════════════════════════════════════════
# Enhance-Only pipeline
# ═════════════════════════════════════════════════════════════════════════════

def _run_enhance_only(audio_file, enhancement_model, enhancement_mix):
    session_id  = uuid.uuid4().hex[:12]
    session_dir = os.path.join("/tmp/whisper-sessions", session_id)
    os.makedirs(session_dir, exist_ok=True)
    temp_files  = []
    start_time  = time.time()

    print(f"\n{'=' * 60}")
    print(f"🔊 Enhance-Only session: {session_id}")
    print(f"{'=' * 60}\n")

    try:
        result = None
        for item in _prepare_input(audio_file, _prog_enh):
            if (isinstance(item, tuple) and len(item) == 3
                    and isinstance(item[0], str) and os.path.isfile(item[0])):
                result = item
                temp_files.extend(item[2])
                break
            yield item

        if result is None:
            return

        audio_path, video_title, _ = result

        if not is_model_available(enhancement_model):
            yield (f"❌ Enhancement model '{enhancement_model}' is not available.",
                   "", None, *_NO_TRANSLATE, *_NO_ENHANCE)
            return

        model_label = next(
            (lbl for lbl, val in ENHANCEMENT_MODEL_CHOICES if val == enhancement_model),
            enhancement_model,
        )

        raw_audio, file_sr = sf.read(audio_path, dtype="float32")
        if raw_audio.ndim == 2:
            raw_audio = raw_audio.mean(axis=1)
        audio_duration = len(raw_audio) / file_sr
        print(f"⏱️  Audio duration: {audio_duration:.1f}s")

        yield _prog_enh(30, f"Enhancing speech ({model_label}, mix={enhancement_mix:.2f})…")

        output_dir      = _get_output_dir()
        safe            = _safe_title(video_title)
        ts              = datetime.now().strftime("%Y%m%d_%H%M%S")
        uid             = uuid.uuid4().hex[:6]
        enhanced_wav    = os.path.join(output_dir, f"{safe}_enhanced_{ts}_{uid}.wav")
        spectrogram_png = os.path.join(output_dir, f"{safe}_spectrogram_{ts}_{uid}.png")

        enhance_file(audio_path, enhanced_wav,
                     mix_factor=enhancement_mix, model_name=enhancement_model)
        print(f"💾 Enhanced audio saved: {enhanced_wav}")

        yield _prog_enh(80, "Generating spectrogram…")

        enh_np, enh_sr = sf.read(enhanced_wav, dtype="float32")
        if enh_np.ndim == 2:
            enh_np = enh_np.mean(axis=1)

        _generate_spectrogram_png(
            enh_np, enh_sr, spectrogram_png,
            title=f"Enhanced Spectrogram — {model_label}",
        )

        elapsed = time.time() - start_time

        badge_style = (
            "display:inline-flex;align-items:center;gap:6px;"
            "background:#f0f7ff;border:1px solid #c7dff7;border-radius:8px;"
            "padding:6px 12px;margin:4px;font-size:14px;"
        )
        ls = "color:#555;font-weight:400;"
        vs = "color:#1565c0;font-weight:600;"

        def _badge(icon, label, value):
            return (f'<span style="{badge_style}">'
                    f'{icon} <span style="{ls}">{label}:</span>'
                    f'<span style="{vs}">{value}</span></span>')

        badges = "".join([
            _badge("⏱️", "Audio",      f"{audio_duration:.1f}s"),
            _badge("⚡", "Processing", f"{elapsed:.1f}s"),
            _badge("🔊", "Model",      model_label),
            _badge("🎚️", "Mix",        f"{enhancement_mix:.2f}"),
        ])
        status_html = (
            '<div style="margin-bottom:6px;font-weight:600;color:#2e7d32;font-size:15px">'
            '✅ Enhancement complete</div>'
            f'<div style="display:flex;flex-wrap:wrap;gap:2px">{badges}</div>'
        )

        print(f"\n{'=' * 60}")
        print(f"✅ Enhance-Only done: {session_id}  ({elapsed:.1f}s)")
        print(f"{'=' * 60}\n")

        yield (
            status_html,
            "", None,
            gr.update(visible=False),
            "", None,
            gr.update(visible=True),   # enhance_only_col
            enhanced_wav,
            spectrogram_png,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n❌ Enhance-Only failed: {session_id}\nError: {e}\n")
        yield "❌ 增強失敗，請稍後再試。", "", None, *_NO_TRANSLATE, *_NO_ENHANCE

    finally:
        for f in temp_files:
            if f and os.path.exists(f):
                try:
                    os.unlink(f)
                except Exception:
                    pass
        if os.path.exists(session_dir):
            try:
                shutil.rmtree(session_dir)
            except Exception:
                pass


# ═════════════════════════════════════════════════════════════════════════════
# ASR pipeline
# ═════════════════════════════════════════════════════════════════════════════

def _run_asr(
    audio_file, model_size, language, task,
    use_vad, min_silence_duration_s, merge_subtitles,
    convert_to_traditional, max_chars, translate_hakka,
    llm_system_prompt, use_lexicon,
    use_enhancement, enhancement_model, enhancement_mix,
):
    if language == "taigi":
        language = "zh"
        task = "transcribe"
    elif language == "hakka":
        language = "zh"
    if task == "translate_mandarin":
        task = "transcribe"

    session_id  = uuid.uuid4().hex[:12]
    session_dir = os.path.join("/tmp/whisper-sessions", session_id)
    os.makedirs(session_dir, exist_ok=True)

    start_time     = time.time()
    temp_files     = []
    audio_duration = 0.0
    worker_id      = None

    print(f"\n{'=' * 60}")
    print(f"🎦 ASR session: {session_id}")
    print(f"{'=' * 60}\n")

    try:
        result = None
        for item in _prepare_input(audio_file, _prog_asr):
            if (isinstance(item, tuple) and len(item) == 3
                    and isinstance(item[0], str) and os.path.isfile(item[0])):
                result = item
                temp_files.extend(item[2])
                break
            yield item

        if result is None:
            return

        audio_path, video_title, _ = result

        # ── Speech Enhancement ────────────────────────────────────────────
        if use_enhancement and enhancement_mix > 0.0:
            if not is_model_available(enhancement_model):
                print(f"⚠️  Enhancement model '{enhancement_model}' not available — skipping")
            else:
                model_label = next(
                    (lbl for lbl, val in ENHANCEMENT_MODEL_CHOICES if val == enhancement_model),
                    enhancement_model,
                )
                yield _prog_asr(28, f"Enhancing speech ({model_label}, mix={enhancement_mix:.2f})…")
                enhanced_path = os.path.join(
                    session_dir,
                    f"enhanced_{uuid.uuid4().hex[:8]}{os.path.splitext(audio_path)[1]}",
                )
                try:
                    enhance_file(audio_path, enhanced_path,
                                 mix_factor=enhancement_mix, model_name=enhancement_model)
                    audio_path = enhanced_path
                    temp_files.append(enhanced_path)
                    print(f"✅ Enhancement applied: {enhancement_model}")
                except Exception as e:
                    print(f"⚠️  Enhancement failed, continuing without it: {e}")

        try:
            audio_duration = sf.info(audio_path).duration
            print(f"⏱️  Audio duration: {audio_duration:.1f}s")
        except Exception as e:
            print(f"Warning: Could not get audio duration: {e}")

        # ── VAD ───────────────────────────────────────────────────────────
        vad_chunks = None
        if use_vad:
            yield _prog_asr(32, "Detecting speech segments with VAD...")
            _vad = SileroVAD(min_silence_duration_ms=int(min_silence_duration_s * 1000))
            _audio, _sr = sf.read(audio_path, dtype="float32")
            if _audio.ndim == 2:
                _audio = _audio.mean(axis=1)
            if _sr != 16000:
                from scipy import signal as _sig
                _audio = _sig.resample(_audio, int(len(_audio) * 16000 / _sr)).astype(np.float32)
            vad_chunks = _vad.segment_audio(_audio, merge=True, min_duration=0.5, max_duration=30.0)
            n_chunks = len(vad_chunks)
            print(f"🎯 VAD detected {n_chunks} speech segment(s)")
            if n_chunks == 0:
                yield "⚠️ No speech detected", "", None, *_NO_TRANSLATE, *_NO_ENHANCE
                return
            yield _prog_asr(34, f"VAD: {n_chunks} speech segment(s) detected")

        yield _prog_asr(35, "Loading Whisper model...")

        trans, worker_id = transcriber_pool.get_single_gpu_transcriber(
            model_size, use_vad, min_silence_duration_s
        )
        yield _prog_asr(40, f"Model loaded on GPU {worker_id}. Starting transcription...")

        segments = trans.transcribe(
            audio_path,
            language=language if language != "auto" else None,
            task=task,
            progress_callback=None,
            vad_chunks=vad_chunks,
        )

        yield _prog_asr(85, "Transcription complete")

        if not segments:
            yield "⚠️ No speech detected", "", None, *_NO_TRANSLATE, *_NO_ENHANCE
            return

        print(f"📝 Generated {len(segments)} segments")
        asr_segments        = [seg.copy() for seg in segments]
        is_hakka_model      = model_size in HAKKA_MODELS_IDS
        do_translate        = translate_hakka and is_hakka_model and LLM_ENABLED
        translated_segments = None

        if do_translate:
            yield _prog_asr(86, "Translating Hakka → Mandarin via LLM...")
            translated_segments = translate_segments(
                [seg.copy() for seg in asr_segments],
                system_prompt=llm_system_prompt,
                use_lexicon=use_lexicon,
                lexicon=_HAKKA_LEXICON,
            )
            print("✅ Hakka translation complete")
        elif translate_hakka and not LLM_ENABLED:
            print("⚠️  LLM translation requested but ENABLE_LLM=false — skipping")

        if language == "zh" and convert_to_traditional:
            converter = get_converter()
            if converter.is_available():
                yield _prog_asr(88, "Converting to Traditional Chinese...")
                asr_segments = convert_segments_to_traditional(asr_segments)
                if translated_segments is not None:
                    translated_segments = convert_segments_to_traditional(translated_segments)

        if merge_subtitles:
            yield _prog_asr(90, "Merging subtitle segments...")
            asr_segments = merge_segments(asr_segments, max_chars=max_chars)
            if translated_segments is not None:
                translated_segments = merge_segments(translated_segments, max_chars=max_chars)

        yield _prog_asr(95, "Generating SRT file(s)...")

        output_dir = _get_output_dir()
        safe        = _safe_title(video_title)

        asr_srt_content = segments_to_srt(asr_segments)
        asr_srt_path    = _save_srt(asr_srt_content, safe, "asr", output_dir)
        print(f"💾 ASR SRT saved: {asr_srt_path}")

        translated_srt_content = ""
        translated_srt_path    = None
        translated_col_update  = gr.update(visible=False)

        if translated_segments is not None:
            translated_srt_content = segments_to_srt(translated_segments)
            translated_srt_path    = _save_srt(translated_srt_content, safe, "translated", output_dir)
            translated_col_update  = gr.update(visible=True)
            print(f"💾 Translated SRT saved: {translated_srt_path}")

        elapsed = time.time() - start_time
        rtf     = (elapsed / audio_duration) if audio_duration > 0 else None

        badge_style = (
            "display:inline-flex;align-items:center;gap:6px;"
            "background:#f0f7ff;border:1px solid #c7dff7;border-radius:8px;"
            "padding:6px 12px;margin:4px;font-size:14px;"
        )
        ls = "color:#555;font-weight:400;"
        vs = "color:#1565c0;font-weight:600;"

        metrics = [
            ("📝", "Segments",   str(len(asr_segments))),
            ("⏱️", "Audio",      f"{audio_duration:.1f}s" if audio_duration > 0 else None),
            ("⚡", "Processing", f"{elapsed:.1f}s"),
            ("🚀", "RTF",        f"{rtf:.4f}" if rtf is not None else None),
        ]
        badges_html = "".join(
            f'<span style="{badge_style}">'
            f'{icon} <span style="{ls}">{label}:</span>'
            f'<span style="{vs}">{value}</span></span>'
            for icon, label, value in metrics if value is not None
        )
        status_html = (
            '<div style="margin-bottom:6px;font-weight:600;color:#2e7d32;font-size:15px">'
            '✅ Transcription complete</div>'
            f'<div style="display:flex;flex-wrap:wrap;gap:2px">{badges_html}</div>'
        )

        print(f"\n{'=' * 60}")
        print(f"✅ ASR done: {session_id}  ({elapsed:.1f}s)")
        print(f"{'=' * 60}\n")

        yield (
            status_html,
            asr_srt_content, asr_srt_path,
            translated_col_update,
            translated_srt_content, translated_srt_path,
            gr.update(visible=False), None, None,   # enhance_only_col stays hidden
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n❌ ASR failed: {session_id}\nError: {e}\n")
        yield "❌ 處理失敗，請稍後再試。", "", None, *_NO_TRANSLATE, *_NO_ENHANCE

    finally:
        if worker_id is not None:
            transcriber_pool.release_single_gpu_transcriber(worker_id)
        for f in temp_files:
            if f and os.path.exists(f):
                try:
                    os.unlink(f)
                except Exception:
                    pass
        if os.path.exists(session_dir):
            try:
                shutil.rmtree(session_dir)
            except Exception:
                pass


# ═════════════════════════════════════════════════════════════════════════════
# Dispatcher
# ═════════════════════════════════════════════════════════════════════════════

def run(
    mode, audio_file,
    use_enhancement, enhancement_model, enhancement_mix,
    model_size, language, task,
    use_vad, min_silence_duration_s, merge_subtitles,
    convert_to_traditional, max_chars,
    translate_hakka, llm_system_prompt, use_lexicon,
):
    if mode == MODE_ENHANCE_ONLY:
        yield from _run_enhance_only(audio_file, enhancement_model, enhancement_mix)
    else:
        yield from _run_asr(
            audio_file, model_size, language, task,
            use_vad, min_silence_duration_s, merge_subtitles,
            convert_to_traditional, max_chars, translate_hakka,
            llm_system_prompt, use_lexicon,
            use_enhancement, enhancement_model, enhancement_mix,
        )


# ═════════════════════════════════════════════════════════════════════════════
# Gradio interface
# ═════════════════════════════════════════════════════════════════════════════

def create_interface() -> gr.Blocks:

    lang_choices = [
        ("Auto",     "auto"),
        ("Mandarin", "zh"),
        ("English",  "en"),
    ]
    if HAKKA_MODELS_IDS:
        lang_choices.append(("Hakka", "hakka"))
    if TAIGI_MODELS_IDS:
        lang_choices.append(("Taigi", "taigi"))

    with gr.Blocks(
        title="FormoSTT: Speech-to-Text System for Taiwanese Languages",
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
        analytics_enabled=False,
    ) as app:

        gr.Markdown(
            """
            # FormoSTT: Speech-to-Text System for Taiwanese Languages
            ## 臺灣語音辨識暨翻譯系統

            ### Model Developers
            - **[李鴻欣 Hung-Shin Lee](https://www.linkedin.com/in/hungshinlee)**（聯和科創股份有限公司）
            - **[陳力瑋 Li-Wei Chen](mailto:wayne900619@gmail.com)**（國立清華大學資訊工程學研究所）
            ### Machine Providers (RTX 2080 Ti * 4)
            - **[王新民 Hsin-Min Wang](https://homepage.iis.sinica.edu.tw/pages/whm/index_zh.html)**（中央研究院資訊科學研究所）
            - **[廖沛俊 Pei-Jun Liao](mailto:newsboy3423@gmail.com)**（中央研究院資訊科學研究所）
            ### Supported Languages
            - **Mandarin**, **Hakka**, **Taigi**, and **English**
            """
        )

        pdf_filename = "Terms_and_Privacy.pdf"
        pdf_path = (
            f"/app/docs/{pdf_filename}"
            if os.path.exists(f"/app/docs/{pdf_filename}")
            else f"docs/{pdf_filename}"
        )
        if os.path.exists(pdf_path):
            gr.HTML(
                '<a href="/terms-and-privacy" target="_blank">'
                '使用者條款、資訊安全與隱私權政策 (Terms and Privacy Policy)</a>'
            )

        with gr.Row():
            # ── Left: Input & Settings ──────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### 📥 Input")

                audio_input = gr.Audio(
                    label="Upload Audio or Video",
                    type="filepath",
                    sources=["upload", "microphone"],
                )

                gr.Markdown("### ⚙️ Settings")

                mode_radio = gr.Radio(
                    choices=[
                        ("🎙️ ASR (Transcribe)", MODE_ASR),
                        ("🔊 Enhance Only",      MODE_ENHANCE_ONLY),
                    ],
                    value=MODE_ASR,
                    label="Mode",
                    info=(
                        "ASR: full speech-to-text pipeline  |  "
                        "Enhance Only: noise suppression → enhanced WAV + spectrogram"
                    ),
                )

                # ── ASR settings ────────────────────────────────────────
                with gr.Column(visible=True) as asr_settings_col:
                    default_model = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
                    if default_model in TAIGI_MODELS_IDS:
                        init_lang_sel    = "taigi"
                        init_model_value = default_model
                    elif default_model in HAKKA_MODELS_IDS:
                        init_lang_sel    = "hakka"
                        init_model_value = default_model
                    else:
                        init_lang_sel    = "auto"
                        init_model_value = (default_model if default_model in GENERAL_MODELS_IDS
                                            else "large-v3-turbo")

                    language_selector = gr.Radio(
                        choices=lang_choices,
                        value=init_lang_sel,
                        label="Language",
                    )

                    if init_lang_sel == "taigi":
                        init_model_choices = [(MODEL_CONFIGS[m]["display_name"], m) for m in TAIGI_MODELS_IDS]
                    elif init_lang_sel == "hakka":
                        init_model_choices = [(MODEL_CONFIGS[m]["display_name"], m) for m in HAKKA_MODELS_IDS]
                    else:
                        init_model_choices = [(MODEL_CONFIGS[m]["display_name"], m) for m in GENERAL_MODELS_IDS]

                    model_dropdown = gr.Radio(
                        choices=init_model_choices,
                        value=init_model_value,
                        label="Model",
                    )

                    def _task_cfg_for_model(model_name):
                        if model_name in TAIGI_MODELS_IDS:
                            return ([("Translate to Mandarin", "translate_mandarin")],
                                    "translate_mandarin", False, "Taigi model always outputs Mandarin")
                        elif model_name == "large-v3-turbo":
                            return ([("Transcribe", "transcribe")],
                                    "transcribe", False, "Note: large-v3-turbo only supports Transcribe")
                        elif model_name in HAKKA_MODELS_IDS:
                            return ([("Transcribe", "transcribe")],
                                    "transcribe", False, "Hakka model only supports Transcribe")
                        else:
                            return ([("Transcribe", "transcribe"), ("Translate to English", "translate")],
                                    "transcribe", True, None)

                    def _task_update_for_model(model_name):
                        choices, value, interactive, info = _task_cfg_for_model(model_name)
                        return gr.update(choices=choices, value=value,
                                         interactive=interactive, info=info)

                    init_task_choices, init_task_value, init_task_interactive, init_task_info = \
                        _task_cfg_for_model(init_model_value)

                    task_radio = gr.Radio(
                        choices=init_task_choices,
                        value=init_task_value,
                        label="Task",
                        interactive=init_task_interactive,
                        info=init_task_info,
                    )

                    with gr.Row():
                        use_vad_checkbox = gr.Checkbox(value=True,  label="Enable VAD")
                        merge_checkbox   = gr.Checkbox(value=True,  label="Merge Short Subtitles")
                        zh_conv_checkbox = gr.Checkbox(value=False, label="Convert to zh-TW")

                    min_silence_slider = gr.Slider(
                        minimum=0.01, maximum=2.0, value=0.2, step=0.01,
                        label="VAD: Minimum Silence Duration (seconds)",
                    )
                    max_chars_slider = gr.Slider(
                        minimum=40, maximum=120, value=80, step=10,
                        label="Max Characters Per Line",
                    )

                    with gr.Column(visible=False) as llm_col:
                        translate_hakka_checkbox = gr.Checkbox(
                            value=True,
                            label="🤖 Translate Hakka → Mandarin (via Ollama LLM)",
                            interactive=LLM_ENABLED,
                            info=None if LLM_ENABLED else "LLM not deployed (ENABLE_LLM=false)",
                        )
                        use_lexicon_checkbox = gr.Checkbox(
                            value=True,
                            label="📚 Use Lexicon Augmentation",
                            interactive=LLM_ENABLED,
                        )
                        llm_prompt_textbox = gr.Textbox(
                            value=DEFAULT_SYSTEM_PROMPT,
                            label="🤖 LLM System Prompt",
                            lines=6,
                        )

                # ── Speech Enhancement ──────────────────────────────────
                with gr.Column(visible=True):
                    use_enhancement_checkbox = gr.Checkbox(
                        value=False,
                        label="🔊 Speech Enhancement",
                        interactive=SPEECH_ENHANCEMENT_AVAILABLE,
                        info=None if SPEECH_ENHANCEMENT_AVAILABLE
                             else "No enhancement backend installed",
                    )

                    _enh_info_parts = []
                    if any(v == DEEPFILTERNET3 for _, v in ENHANCEMENT_MODEL_CHOICES):
                        _enh_info_parts.append("DeepFilterNet3: 48 kHz fullband")
                    if any(v.startswith("dpdfnet") for _, v in ENHANCEMENT_MODEL_CHOICES):
                        _enh_info_parts.append("DPDFNet: 16 kHz causal — Baseline (fastest) → 2 → 4 → 8 (best)")
                    _enh_model_info = " | ".join(_enh_info_parts) or None

                    enhancement_model_dropdown = gr.Dropdown(
                        choices=ENHANCEMENT_MODEL_CHOICES,
                        value=_DEFAULT_ENHANCEMENT_MODEL,
                        label="Enhancement Model",
                        visible=False,
                        interactive=True,
                        info=_enh_model_info,
                    )
                    enhancement_mix_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, value=1.0, step=0.05,
                        label="Enhancement Blend (0 = original, 1 = fully enhanced)",
                        visible=False,
                        interactive=True,
                    )

                process_btn = gr.Button("🚀 Start", variant="primary", size="lg")

            # ── Right: Output ───────────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Output")

                status_text = gr.HTML("Waiting for input...")

                with gr.Column(visible=True) as asr_output_col:
                    gr.Markdown("#### 🗣️ ASR Result")
                    asr_srt_output = gr.Textbox(
                        label="SRT Subtitle Content (ASR)",
                        lines=12,
                        max_lines=20,
                    )
                    with gr.Row():
                        asr_copy_btn    = gr.Button("Copy", elem_classes="copy-button")
                        asr_copy_status = gr.HTML("", elem_classes="copy-success")
                    asr_srt_file = gr.File(label="Download ASR SRT")

                    with gr.Column(visible=False) as translated_col:
                        gr.Markdown("#### 🌐 Translation Result (Mandarin)")
                        translated_srt_output = gr.Textbox(
                            label="SRT Subtitle Content (Translated)",
                            lines=12,
                            max_lines=20,
                        )
                        with gr.Row():
                            trans_copy_btn    = gr.Button("Copy", elem_classes="copy-button")
                            trans_copy_status = gr.HTML("", elem_classes="copy-success")
                        translated_srt_file = gr.File(label="Download Translated SRT")

                with gr.Column(visible=False) as enhance_only_col:
                    gr.Markdown("#### 🔊 Enhanced Audio")
                    enhanced_audio = gr.Audio(
                        label="Enhanced Audio (playable & downloadable)",
                        type="filepath",
                        interactive=False,
                    )
                    gr.Markdown(
                        "#### 📊 Spectrogram\n"
                        "<small>X-axis = time (seconds), aligned with the audio player above. "
                        "Y-axis = frequency (kHz). Colour = log power (dB).</small>"
                    )
                    # show_download_button was added in Gradio 4.x — omit for compatibility
                    spectrogram_image = gr.Image(
                        label="Time-Aligned Spectrogram",
                        type="filepath",
                        interactive=False,
                    )

        # ── Event handlers ──────────────────────────────────────────────

        process_btn.click(
            fn=run,
            inputs=[
                mode_radio,
                audio_input,
                use_enhancement_checkbox,
                enhancement_model_dropdown,
                enhancement_mix_slider,
                model_dropdown,
                language_selector,
                task_radio,
                use_vad_checkbox,
                min_silence_slider,
                merge_checkbox,
                zh_conv_checkbox,
                max_chars_slider,
                translate_hakka_checkbox,
                llm_prompt_textbox,
                use_lexicon_checkbox,
            ],
            outputs=[
                status_text,
                asr_srt_output,
                asr_srt_file,
                translated_col,
                translated_srt_output,
                translated_srt_file,
                enhance_only_col,
                enhanced_audio,
                spectrogram_image,
            ],
        )

        def on_mode_change(mode):
            is_asr      = (mode == MODE_ASR)
            enh_checked = not is_asr   # force-on in Enhance Only mode
            return (
                gr.update(visible=is_asr),       # asr_settings_col
                gr.update(visible=is_asr),       # asr_output_col
                gr.update(visible=False),         # enhance_only_col (reset)
                gr.update(value=enh_checked,
                          interactive=is_asr and SPEECH_ENHANCEMENT_AVAILABLE),
                gr.update(visible=not is_asr and SPEECH_ENHANCEMENT_AVAILABLE),  # model dropdown
                gr.update(visible=not is_asr and SPEECH_ENHANCEMENT_AVAILABLE),  # blend slider
            )

        mode_radio.change(
            fn=on_mode_change,
            inputs=[mode_radio],
            outputs=[
                asr_settings_col,
                asr_output_col,
                enhance_only_col,
                use_enhancement_checkbox,
                enhancement_model_dropdown,
                enhancement_mix_slider,
            ],
            queue=False,
        )

        def on_language_change(lang):
            if lang == "taigi":
                mc, nm, is_hakka = [(MODEL_CONFIGS[m]["display_name"], m) for m in TAIGI_MODELS_IDS], TAIGI_MODELS_IDS[0], False
            elif lang == "hakka":
                mc, nm, is_hakka = [(MODEL_CONFIGS[m]["display_name"], m) for m in HAKKA_MODELS_IDS], HAKKA_MODELS_IDS[0], True
            else:
                mc, nm, is_hakka = [(MODEL_CONFIGS[m]["display_name"], m) for m in GENERAL_MODELS_IDS], "large-v3-turbo", False
            return (
                gr.update(choices=mc, value=nm),
                _task_update_for_model(nm),
                gr.update(visible=is_hakka),
                gr.update(value=is_hakka),
                gr.update(visible=is_hakka),
                gr.update(visible=False),
            )

        language_selector.change(
            fn=on_language_change,
            inputs=[language_selector],
            outputs=[model_dropdown, task_radio, llm_col,
                     translate_hakka_checkbox, llm_prompt_textbox, translated_col],
            queue=False,
        )

        def on_model_change(model_name):
            is_hakka = model_name in HAKKA_MODELS_IDS
            return (
                _task_update_for_model(model_name),
                gr.update(visible=is_hakka),
                gr.update(value=is_hakka),
                gr.update(visible=is_hakka),
                gr.update(visible=False),
            )

        model_dropdown.change(
            fn=on_model_change,
            inputs=[model_dropdown],
            outputs=[task_radio, llm_col, translate_hakka_checkbox,
                     llm_prompt_textbox, translated_col],
            queue=False,
        )

        def on_enhancement_toggle(checked):
            vis = checked and SPEECH_ENHANCEMENT_AVAILABLE
            return gr.update(visible=vis), gr.update(visible=vis)

        use_enhancement_checkbox.change(
            fn=on_enhancement_toggle,
            inputs=[use_enhancement_checkbox],
            outputs=[enhancement_model_dropdown, enhancement_mix_slider],
            queue=False,
            show_progress="hidden",
        )

        merge_checkbox.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[merge_checkbox], outputs=[max_chars_slider], queue=False,
        )
        use_vad_checkbox.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[use_vad_checkbox], outputs=[min_silence_slider], queue=False,
        )

        _COPY_JS = """(content) => {
            if (!content) return "⚠️ No content to copy";
            navigator.clipboard.writeText(content).then(
                () => "✅ Copied!",
                (err) => "❌ Failed: " + err
            );
            return "✅ Copied!";
        }"""
        asr_copy_btn.click(fn=None, inputs=[asr_srt_output], outputs=[asr_copy_status], js=_COPY_JS)
        trans_copy_btn.click(fn=None, inputs=[translated_srt_output], outputs=[trans_copy_status], js=_COPY_JS)

        gr.Markdown("### 📋 Examples")
        ground_truth_textbox = gr.Textbox(label="Ground Truth", interactive=False, lines=3)
        gr.Examples(
            examples=[
                ["samples/734a04794010481cb3eed411b6e005cc.wav", "hakka",
                 "下二隻月就愛過年吔，魚仔相關个產品就開始起價，因為呢愛分民眾在防疫期間乜買得著萋萋个魚貨，"
                 "苗栗魚市場就特別推出咧限量个過年禮盒，用網路，注文還過送貨到屋个服務，還過較便宜个價數，"
                 "分苗栗鄉親在屋下裡肚，乜買得著萋萋又有保障个魚貨。"],
                ["samples/874062dc1657497b9ac996971c9ce4bb.wav", "hakka",
                 "這隻世界項有當多人高不將愛摎自家个夢想放忒去，你既然做得追求你个夢想，"
                 "你就愛認真煞猛分佢兜試著當見笑啊。"],
                ["samples/fmZk_OSHbiY.wav", "taigi",
                 "我相信在這樣的景況下，你的心必會很難過，不是嗎？甚至會生氣。"
                 "不過我們都知道，這樣的生氣實在是一種愛的表達，因為生的痛，所以心會痛。"],
            ],
            inputs=[audio_input, language_selector, ground_truth_textbox],
        )

    return app


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("🚀 Starting FormoSTT")
    print("=" * 60 + "\n")

    if ENHANCEMENT_MODEL_CHOICES:
        print("🔊 Enhancement models available:")
        for label, val in ENHANCEMENT_MODEL_CHOICES:
            print(f"   • {val}  ({label})")
    else:
        print("ℹ️  No speech enhancement backend installed")

    print("🧹 Cleaning up temporary files...")
    for path in ["/tmp/whisper-sessions"]:
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
                print(f"  ✓ Cleaned {path}")
            except Exception as e:
                print(f"  ⚠️  Failed to clean {path}: {e}")

    output_dir = "/app/outputs"
    if os.path.exists(output_dir):
        try:
            for f in (glob.glob(os.path.join(output_dir, "*.srt"))
                      + glob.glob(os.path.join(output_dir, "*.wav"))
                      + glob.glob(os.path.join(output_dir, "*.png"))):
                os.unlink(f)
            print(f"  ✓ Cleaned old outputs in {output_dir}")
        except Exception as e:
            print(f"  ⚠️  Failed to clean {output_dir}: {e}")

    if LLM_ENABLED:
        print("🤖 LLM enabled — checking vLLM availability...")
        check_vllm_ready()
    else:
        print("ℹ️  LLM disabled (ENABLE_LLM=false) — skipping vLLM setup")

    default_model = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
    if os.environ.get("PRELOAD_MODEL", "false").lower() == "true":
        print(f"🔄 Pre-loading model: {default_model}")
        trans, gpu_id = transcriber_pool.get_single_gpu_transcriber(default_model, True, 0.1)
        transcriber_pool.release_single_gpu_transcriber(gpu_id)
        print("✅ Model pre-loaded")

    def _load_users() -> dict:
        candidates = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), ".users"),
            "/app/.users",
        ]
        users_file = next((p for p in candidates if os.path.exists(p)), None)
        if users_file is None:
            return {}
        users = {}
        with open(users_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or ":" not in line:
                    continue
                u, p = line.split(":", 1)
                users[u.strip()] = p.strip()
        return users

    _users = _load_users()
    if not _users:
        pw = os.environ.get("GRADIO_PASSWORD", "").strip()
        if pw:
            _users = {"admin": pw}

    auth_list = [(u, p) for u, p in _users.items()] if _users else None
    print(
        ("Authentication enabled: " + ", ".join(_users.keys()))
        if auth_list else "No credentials configured - running without authentication"
    )

    gradio_app = create_interface()
    gradio_app.queue(max_size=10, default_concurrency_limit=2, api_open=False)
    gradio_app.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", 7860)),
        auth=auth_list,
        auth_message="FormoSTT 臺灣語音辨識暨翻譯系統",
        share=False,
    )


if __name__ == "__main__":
    main()
