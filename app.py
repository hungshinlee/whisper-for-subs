"""
Gradio-based web interface for Whisper ASR service.
"""

import os
import glob
import tempfile
import time
import shutil
import uuid
from datetime import datetime, timedelta
from typing import Optional, Tuple, Generator, Dict
from threading import Lock

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
from youtube_downloader import (
    is_youtube_url,
    download_audio_with_progress,
    get_video_info,
)
from srt_utils import segments_to_srt, merge_segments
from chinese_converter import convert_segments_to_traditional, get_converter
from hakka_translator import (
    translate_segments,
    is_llm_enabled,
    pull_model_if_needed,
    load_lexicon,
    DEFAULT_SYSTEM_PROMPT,
)

# Pre-load lexicon at startup (cheap, ~ms)
_HAKKA_LEXICON = load_lexicon()
import numpy as np
from vad import SileroVAD
from speech_enhancer import is_deepfilter_available, enhance_file

SPEECH_ENHANCEMENT_AVAILABLE = is_deepfilter_available()

# Whether LLM translation is available in this deployment
LLM_ENABLED = is_llm_enabled()


# Custom CSS with Roboto font
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');

* {
    font-family: 'Roboto', sans-serif !important;
}

.gradio-container {
    font-family: 'Roboto', sans-serif !important;
}

.prose {
    font-family: 'Roboto', sans-serif !important;
}

textarea, input, button, select {
    font-family: 'Roboto', sans-serif !important;
}

.progress-bar-container {
    margin: 10px 0;
}

.copy-button {
    margin-top: 10px;
}

.copy-success {
    color: #4CAF50;
    font-weight: 500;
    margin-top: 5px;
}
"""


class TranscriberPool:
    """
    Thread-safe pool that distributes single-GPU transcribers across all
    available GPUs, so concurrent short-audio requests use GPU 0, 1, 2 …
    instead of queuing on GPU 0.

    GPU assignment strategy: least-loaded first.
    Each GPU can hold at most one resident WhisperTranscriber per model.
    When a GPU has a cached transcriber for the requested model it is
    preferred (avoids reloading weights); otherwise the GPU with the
    fewest active jobs is chosen.
    """

    def __init__(self):
        self.lock = Lock()

        # Detect GPUs available for single-GPU work.
        # SINGLE_GPU_DEVICES env var overrides; falls back to CUDA_VISIBLE_DEVICES.
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

        print(f"🎰 TranscriberPool: single-GPU slots on GPU(s) {self.single_gpu_ids}")

        # gpu_id -> WhisperTranscriber (one resident transcriber per GPU)
        self.single_gpu_pool: Dict[int, WhisperTranscriber] = {}
        # gpu_id -> number of requests currently using that GPU
        self.gpu_active: Dict[int, int] = {g: 0 for g in self.single_gpu_ids}


    def _least_loaded_gpu(self) -> int:
        """Return the GPU id with the fewest active jobs."""
        return min(self.single_gpu_ids, key=lambda g: self.gpu_active.get(g, 0))

    def get_single_gpu_transcriber(
        self,
        model_size: str,
        use_vad: bool,
        min_silence_duration_s: float,
    ) -> Tuple[WhisperTranscriber, int]:
        """
        Returns (transcriber, gpu_id).  The caller must pass gpu_id to
        release_single_gpu_transcriber() when done.
        """
        min_silence_duration_ms = int(min_silence_duration_s * 1000)
        device = os.environ.get("WHISPER_DEVICE", "cuda")

        with self.lock:
            if not self.single_gpu_ids:
                # CPU fallback
                if "cpu" not in self.single_gpu_pool:
                    self.single_gpu_pool["cpu"] = WhisperTranscriber(
                        model_size=model_size,
                        device="cpu",
                        compute_type="float32",
                        use_vad=use_vad,
                        min_silence_duration_ms=min_silence_duration_ms,
                    )
                return self.single_gpu_pool["cpu"], "cpu"

            # Prefer a GPU that already has this model loaded and is least loaded
            cached_gpu = None
            cached_load = float("inf")
            for gpu_id, trans in self.single_gpu_pool.items():
                if trans.model_size == model_size:
                    load = self.gpu_active.get(gpu_id, 0)
                    if load < cached_load:
                        cached_gpu = gpu_id
                        cached_load = load

            if cached_gpu is not None:
                self.gpu_active[cached_gpu] += 1
                print(f"♻️  Reusing GPU {cached_gpu} transcriber "
                      f"(active={self.gpu_active[cached_gpu]}, model={model_size})")
                return self.single_gpu_pool[cached_gpu], cached_gpu

            # No cached model — pick least-loaded GPU and load there
            gpu_id = self._least_loaded_gpu()

            # If this GPU already has a transcriber for a different model, replace it
            if gpu_id in self.single_gpu_pool:
                print(f"🔄 GPU {gpu_id}: replacing "
                      f"{self.single_gpu_pool[gpu_id].model_size} → {model_size}")
                del self.single_gpu_pool[gpu_id]

            print(f"✨ Loading {model_size} on GPU {gpu_id}")
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
            print(f"✅ GPU {gpu_id} ready (active={self.gpu_active[gpu_id]})")
            return trans, gpu_id

    def release_single_gpu_transcriber(self, gpu_id):
        with self.lock:
            if gpu_id in self.gpu_active and self.gpu_active[gpu_id] > 0:
                self.gpu_active[gpu_id] -= 1
                print(f"✅ Released GPU {gpu_id} "
                      f"(active={self.gpu_active[gpu_id]})")



# Global transcriber pool — GPU slots auto-detected from CUDA_VISIBLE_DEVICES
transcriber_pool = TranscriberPool()


def cleanup_old_files(max_age_hours: int = 24):
    now = datetime.now()

    tmp_dir = "/tmp/whisper-downloads"
    if os.path.exists(tmp_dir):
        for f in glob.glob(os.path.join(tmp_dir, "*")):
            try:
                mtime = datetime.fromtimestamp(os.path.getmtime(f))
                if now - mtime > timedelta(hours=max_age_hours):
                    if os.path.isfile(f):
                        os.unlink(f)
                    elif os.path.isdir(f):
                        shutil.rmtree(f)
            except Exception:
                pass

    output_dir = "/app/outputs"
    if os.path.exists(output_dir):
        for f in glob.glob(os.path.join(output_dir, "*.srt")):
            try:
                mtime = datetime.fromtimestamp(os.path.getmtime(f))
                if now - mtime > timedelta(hours=max_age_hours):
                    os.unlink(f)
            except Exception:
                pass

    sessions_dir = "/tmp/whisper-sessions"
    if os.path.exists(sessions_dir):
        for session_dir in glob.glob(os.path.join(sessions_dir, "*")):
            try:
                if os.path.isdir(session_dir):
                    mtime = datetime.fromtimestamp(os.path.getmtime(session_dir))
                    if now - mtime > timedelta(hours=max_age_hours):
                        shutil.rmtree(session_dir)
            except Exception:
                pass


def format_progress_html(percent: int, message: str) -> str:
    return f"""
<div class="progress-bar-container">
    <div style="margin-bottom: 5px; font-weight: 500;">{message}</div>
    <div style="background-color: #e0e0e0; border-radius: 10px; height: 20px; overflow: hidden;">
        <div style="background: linear-gradient(90deg, #2196F3, #21CBF3); height: 100%; width: {percent}%; transition: width 0.3s ease; border-radius: 10px;"></div>
    </div>
    <div style="text-align: right; font-size: 12px; color: #666; margin-top: 3px;">{percent}%</div>
</div>
"""


def _save_srt(srt_content: str, safe_title: str, suffix: str, output_dir: str) -> str:
    """Write SRT content to a file and return the path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:6]
    filename = f"{safe_title}_{suffix}_{timestamp}_{unique_id}.srt"
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(srt_content)
    return path


# Yield helper: 6-tuple used throughout process_audio
# (status, asr_srt, asr_file, translated_col_update, translated_srt, translated_file)
_NO_TRANSLATION = (gr.update(visible=False), "", None)


def process_audio(
    audio_file: Optional[str],
    youtube_url: str,
    model_size: str,
    language: str,
    task: str,
    use_vad: bool,
    min_silence_duration_s: float,
    merge_subtitles: bool,
    convert_to_traditional: bool,
    max_chars: int,
    translate_hakka: bool = False,
    llm_system_prompt: str = "",
    use_lexicon: bool = False,
    use_enhancement: bool = False,
    enhancement_mix: float = 1.0,
) -> Generator:
    """
    Process audio from file or YouTube URL.

    Yields 6-tuples:
        (status_html, asr_srt_text, asr_srt_file,
         translated_col_update, translated_srt_text, translated_srt_file)
    """
    # "hakka" is a UI-level selector value — map it to the actual Whisper language code
    if language == "hakka":
        language = "zh"

    session_id = uuid.uuid4().hex[:12]
    session_dir = os.path.join("/tmp/whisper-sessions", session_id)
    os.makedirs(session_dir, exist_ok=True)

    start_time = time.time()
    audio_path = None
    temp_files = []
    video_title = "output"
    audio_duration = 0.0
    worker_id = None

    # Shorthand for progress-only yields (translation panel stays hidden)
    def prog(pct, msg):
        return format_progress_html(pct, msg), "", None, *_NO_TRANSLATION

    print(f"\n{'=' * 60}")
    print(f"🎬 Starting session: {session_id}")
    print(f"{'=' * 60}\n")

    try:
        # ── Input preparation ──────────────────────────────────────────
        if youtube_url and youtube_url.strip():
            if not is_youtube_url(youtube_url):
                yield "❌ Invalid YouTube URL", "", None, *_NO_TRANSLATION
                return

            yield prog(5, "Fetching video information...")
            info = get_video_info(youtube_url)
            if info:
                video_title = info.get("title", "youtube_audio")
                yield prog(10, f"Downloading: {video_title[:40]}...")

            download_dir = os.path.join(session_dir, "downloads")
            os.makedirs(download_dir, exist_ok=True)

            audio_path, title = download_audio_with_progress(
                youtube_url, output_dir=download_dir, progress_callback=None,
            )
            yield prog(30, "Download complete")

            if audio_path is None:
                yield "❌ Download failed. Please check the URL.", "", None, *_NO_TRANSLATION
                return

            if title:
                video_title = title
            temp_files.append(audio_path)

        elif audio_file:
            upload_copy = os.path.join(
                session_dir,
                f"upload_{uuid.uuid4().hex[:8]}{os.path.splitext(audio_file)[1]}",
            )
            shutil.copy2(audio_file, upload_copy)
            audio_path = upload_copy
            temp_files.append(upload_copy)
            video_title = os.path.splitext(os.path.basename(audio_file))[0]
            yield prog(10, "Audio file loaded and copied to session")
            print(f"📁 Uploaded file copied to session: {upload_copy}")
        else:
            yield "❌ Please upload an audio file or enter a YouTube URL", "", None, *_NO_TRANSLATION
            return

        # ── Speech Enhancement ──────────────────────────────────────────
        if use_enhancement and SPEECH_ENHANCEMENT_AVAILABLE and enhancement_mix > 0.0:
            yield prog(28, f"Enhancing speech (DeepFilterNet3, mix={enhancement_mix:.2f})…")
            enhanced_path = os.path.join(
                session_dir,
                f"enhanced_{uuid.uuid4().hex[:8]}{os.path.splitext(audio_path)[1]}",
            )
            try:
                enhance_file(audio_path, enhanced_path, mix_factor=enhancement_mix)
                audio_path = enhanced_path
                temp_files.append(enhanced_path)
                print(f"✅ Speech enhancement applied (mix={enhancement_mix:.2f})")
            except Exception as e:
                print(f"⚠️  Speech enhancement failed, continuing without it: {e}")

        try:
            audio_info = sf.info(audio_path)
            audio_duration = audio_info.duration
            print(f"⏱️  Audio duration: {audio_duration:.1f}s")
        except Exception as e:
            print(f"Warning: Could not get audio duration: {e}")
            audio_duration = 0.0

        # ── Transcription ──────────────────────────────────────────────
        # ── VAD ──────────────────────────────────────────────────────
        # Running VAD here — after enhancement — guarantees speech detection
        # always operates on the cleaned-up audio.
        vad_chunks = None
        if use_vad:
            yield prog(32, "Detecting speech segments with VAD...")
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
                yield "⚠️ No speech detected", "", None, *_NO_TRANSLATION
                return
            yield prog(34, f"VAD: {n_chunks} speech segment(s) detected")

        yield prog(35, "Loading Whisper model...")

        trans, worker_id = transcriber_pool.get_single_gpu_transcriber(
            model_size, use_vad, min_silence_duration_s
        )
        yield prog(40, f"Model loaded on GPU {worker_id}. Starting transcription...")
        print(f"🔧 Using single-GPU transcriber on GPU {worker_id}")

        segments = trans.transcribe(
            audio_path,
            language=language if language != "auto" else None,
            task=task,
            progress_callback=None,
            vad_chunks=vad_chunks,  # pre-computed from enhanced audio
        )

        yield prog(85, "Transcription complete")

        if not segments:
            yield "⚠️ No speech detected", "", None, *_NO_TRANSLATION
            return

        print(f"📝 Generated {len(segments)} segments")

        # ── Post-processing ────────────────────────────────────────────

        # Save a deep copy of raw ASR segments before any mutation
        asr_segments = [seg.copy() for seg in segments]

        # LLM translation
        is_hakka_model = any(m in model_size for m in [
            "formospeech/whisper-large-v2-taiwanese-hakka-v1",
            "formospeech/whisper-large-v3-taiwanese-hakka",
        ])
        do_translate = translate_hakka and is_hakka_model and LLM_ENABLED
        translated_segments = None

        if do_translate:
            yield prog(86, "Translating Hakka → Mandarin via LLM...")
            translated_segments = translate_segments(
                [seg.copy() for seg in asr_segments],
                system_prompt=llm_system_prompt,
                use_lexicon=use_lexicon,
                lexicon=_HAKKA_LEXICON,
            )
            print("✅ Hakka translation complete")
        elif translate_hakka and not LLM_ENABLED:
            print("⚠️  LLM translation requested but ENABLE_LLM=false — skipping")

        # Traditional Chinese conversion (applied to both pipelines if active)
        if language == "zh" and convert_to_traditional:
            converter = get_converter()
            if converter.is_available():
                yield prog(88, "Converting to Traditional Chinese...")
                asr_segments = convert_segments_to_traditional(asr_segments)
                if translated_segments is not None:
                    translated_segments = convert_segments_to_traditional(translated_segments)
                print("✅ Converted to Traditional Chinese")

        # Merge
        if merge_subtitles:
            yield prog(90, "Merging subtitle segments...")
            asr_segments = merge_segments(asr_segments, max_chars=max_chars)
            if translated_segments is not None:
                translated_segments = merge_segments(translated_segments, max_chars=max_chars)

        # ── Generate SRT files ─────────────────────────────────────────
        yield prog(95, "Generating SRT file(s)...")

        output_dir = (
            "/app/outputs" if os.path.exists("/app/outputs") else tempfile.gettempdir()
        )
        os.makedirs(output_dir, exist_ok=True)

        safe_title = "".join(
            c for c in video_title if c.isalnum() or c in " -_"
        ).strip()[:40]

        asr_srt_content = segments_to_srt(asr_segments)
        asr_srt_path    = _save_srt(asr_srt_content, safe_title, "asr", output_dir)
        print(f"💾 ASR SRT saved: {asr_srt_path}")

        translated_srt_content = ""
        translated_srt_path    = None
        translated_col_update  = gr.update(visible=False)

        if translated_segments is not None:
            translated_srt_content = segments_to_srt(translated_segments)
            translated_srt_path    = _save_srt(translated_srt_content, safe_title, "translated", output_dir)
            translated_col_update  = gr.update(visible=True)
            print(f"💾 Translated SRT saved: {translated_srt_path}")

        # ── Final status ───────────────────────────────────────────────
        processing_time = time.time() - start_time
        parts = [f"✅ 轉識完成！共 {len(asr_segments)} 段字幕。"]
        if audio_duration > 0:
            parts.append(f"音訊長度：{audio_duration:.1f} 秒")
        parts.append(f"處理時間：{processing_time:.1f} 秒")
        if audio_duration > 0 and processing_time > 0:
            parts.append(f"處理倉數：{audio_duration / processing_time:.2f}x")

        print(f"\n{'=' * 60}")
        print(f"✅ Session completed: {session_id}  ({processing_time:.1f}s)")
        print(f"{'=' * 60}\n")

        yield (
            " | ".join(parts),
            asr_srt_content,
            asr_srt_path,
            translated_col_update,
            translated_srt_content,
            translated_srt_path,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n❌ Session failed: {session_id}\nError: {str(e)}\n")
        yield "❌ 處理失敗，請稍後再試。", "", None, *_NO_TRANSLATION

    finally:
        if worker_id:
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


# ── Gradio interface ──────────────────────────────────────────────────────────

def create_interface() -> gr.Blocks:

    with gr.Blocks(
        title="FormoSST: Speech-to-Text System for Taiwanese Languages",
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
    ) as app:
        gr.Markdown(
            """
            # FormoSST: Speech-to-Text System for Taiwanese Languages
            ## 臺灣語音辨識暨翻譯系統

            ### Model Developers
            - **[李鴻欣 Hung-Shin Lee](https://www.linkedin.com/in/hungshinlee)**（聯和科創股份有限公司）
            - **[陳力瑋 Li-Wei Chen](mailto:wayne900619@gmail.com)**（國立清華大學資訊工程學研究所）
            ### Machine Providers (RTX 2080 Ti * 4)
            - **[王新民 Hsin-Min Wang](https://homepage.iis.sinica.edu.tw/pages/whm/index_zh.html)**（中央研究院資訊科學研究所）
            - **[廖沛俊 Pei-Jun Liao](mailto:newsboy3423@gmail.com)**（中央研究院資訊科學研究所）
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
            # ── Left column: Input & Settings ──────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### 📥 Input")

                audio_input = gr.Audio(
                    label="Upload Audio or Video",
                    type="filepath",
                    sources=["upload", "microphone"],
                )

                gr.Markdown("**OR**")

                youtube_input = gr.Textbox(
                    label="YouTube URL",
                    placeholder="https://www.youtube.com/watch?v=...",
                    value="https://www.youtube.com/watch?v=Z-RUXs5YOyE",
                )

                gr.Markdown("### ⚙️ Settings")

                # ── Step 1: Language selector ───────────────────────────
                GENERAL_MODELS_IDS = ["large-v3", "large-v3-turbo"]
                HAKKA_MODELS_IDS = [
                    "formospeech/whisper-large-v2-taiwanese-hakka-v1",
                    "formospeech/whisper-large-v3-taiwanese-hakka",
                ]

                default_model = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
                default_is_hakka = any(m in default_model for m in HAKKA_MODELS_IDS)

                # Determine initial language selector value and initial model choices
                if default_is_hakka:
                    init_lang_sel = "hakka"
                    init_model_choices = [
                        (MODEL_CONFIGS[m]["display_name"], m) for m in HAKKA_MODELS_IDS
                    ]
                    init_model_value = default_model if default_model in HAKKA_MODELS_IDS else HAKKA_MODELS_IDS[0]
                else:
                    init_lang_sel = "auto"
                    init_model_choices = [
                        (MODEL_CONFIGS[m]["display_name"], m) for m in GENERAL_MODELS_IDS
                    ]
                    init_model_value = default_model if default_model in GENERAL_MODELS_IDS else "large-v3-turbo"

                language_selector = gr.Radio(
                    choices=[
                        ("Auto",     "auto"),
                        ("Mandarin", "zh"),
                        ("English",  "en"),
                        ("Hakka",    "hakka"),
                    ],
                    value=init_lang_sel,
                    label="Language",
                    info="先選擇語言，再選擇對應模型",
                )

                # ── Step 2: Model selector (filtered by language) ───────
                model_dropdown = gr.Dropdown(
                    choices=init_model_choices,
                    value=init_model_value,
                    label="Model",
                )

                task_interactive = init_model_value != "large-v3-turbo"

                with gr.Row():
                    task_radio = gr.Radio(
                        choices=[
                            ("Transcribe", "transcribe"),
                            ("Translate to English", "translate"),
                        ],
                        value="transcribe",
                        label="Task",
                        interactive=task_interactive,
                        info="Note: large-v3-turbo only supports Transcribe"
                        if not task_interactive else None,
                    )

                with gr.Row():
                    use_vad_checkbox   = gr.Checkbox(value=True,  label="Enable VAD")
                    merge_checkbox     = gr.Checkbox(value=True,  label="Merge Short Subtitles")
                    zh_conv_checkbox   = gr.Checkbox(value=False, label="Convert to zh-TW")

                # Speech Enhancement controls
                with gr.Column(visible=True) as enhancement_col:
                    use_enhancement_checkbox = gr.Checkbox(
                        value=True,
                        label="🔊 Speech Enhancement (DeepFilterNet3)",
                        interactive=SPEECH_ENHANCEMENT_AVAILABLE,
                        info=None if SPEECH_ENHANCEMENT_AVAILABLE
                             else "deepfilternet not installed",
                    )
                    enhancement_mix_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, value=1.0, step=0.05,
                        label="Enhancement Blend (0 = original, 1 = fully enhanced)",
                        visible=True,
                        interactive=True,
                    )

                # LLM controls — wrapped in Column to avoid Gradio hidden-element event bugs
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
                        info="將詞彙表的參考譯文注入 Prompt，幫助 LLM 正確翻譯客語詞彙",
                        interactive=LLM_ENABLED,
                    )
                    llm_prompt_textbox = gr.Textbox(
                        value=DEFAULT_SYSTEM_PROMPT,
                        label="🤖 LLM System Prompt",
                        info="可自訂翻譯指令，留空則使用預設 Prompt",
                        lines=6,
                        visible=True,
                    )

                min_silence_slider = gr.Slider(
                    minimum=0.01, maximum=2.0, value=0.2, step=0.01,
                    label="VAD: Minimum Silence Duration (seconds)",
                )

                max_chars_slider = gr.Slider(
                    minimum=40, maximum=120, value=80, step=10,
                    label="Max Characters Per Line",
                )

                process_btn = gr.Button("🚀 Start", variant="primary", size="lg")

            # ── Right column: Output ────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Output")

                status_text = gr.HTML("Waiting for input...")

                # ── ASR output (always visible) ─────────────────────────
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

                # ── Translation output (visible only after LLM translation) ──
                with gr.Column(visible=False) as translated_col:
                    gr.Markdown("#### 🈯 Translation Result (繁體中文)")
                    translated_srt_output = gr.Textbox(
                        label="SRT Subtitle Content (Translated)",
                        lines=12,
                        max_lines=20,
                    )
                    with gr.Row():
                        trans_copy_btn    = gr.Button("Copy", elem_classes="copy-button")
                        trans_copy_status = gr.HTML("", elem_classes="copy-success")
                    translated_srt_file = gr.File(label="Download Translated SRT")

        # ── Event handlers ──────────────────────────────────────────────

        process_btn.click(
            fn=process_audio,
            inputs=[
                audio_input,
                youtube_input,
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
                use_enhancement_checkbox,
                enhancement_mix_slider,
            ],
            outputs=[
                status_text,
                asr_srt_output,
                asr_srt_file,
                translated_col,
                translated_srt_output,
                translated_srt_file,
            ],
        )

        _GENERAL_MODELS_IDS = ["large-v3", "large-v3-turbo"]
        _HAKKA_MODELS_IDS = [
            "formospeech/whisper-large-v2-taiwanese-hakka-v1",
            "formospeech/whisper-large-v3-taiwanese-hakka",
        ]

        def on_language_change(lang):
            """Update model choices and related controls when language selector changes."""
            if lang == "hakka":
                choices = [(MODEL_CONFIGS[m]["display_name"], m) for m in _HAKKA_MODELS_IDS]
                new_model = _HAKKA_MODELS_IDS[0]
                is_hakka = True
            else:
                choices = [(MODEL_CONFIGS[m]["display_name"], m) for m in _GENERAL_MODELS_IDS]
                new_model = "large-v3-turbo"
                is_hakka = False

            task_update = gr.update(
                value="transcribe", interactive=False,
                info="Note: large-v3-turbo only supports Transcribe",
            ) if new_model == "large-v3-turbo" else gr.update(interactive=True, info=None)

            return (
                gr.update(choices=choices, value=new_model),   # model_dropdown
                task_update,                                    # task_radio
                gr.update(visible=is_hakka),                   # llm_col
                gr.update(value=is_hakka),                     # translate_hakka_checkbox
                gr.update(visible=is_hakka),                   # llm_prompt_textbox
                gr.update(visible=False),                      # translated_col
            )

        language_selector.change(
            fn=on_language_change,
            inputs=[language_selector],
            outputs=[
                model_dropdown,
                task_radio,
                llm_col,
                translate_hakka_checkbox,
                llm_prompt_textbox,
                translated_col,
            ],
            queue=False,
        )

        def on_model_change(model_name):
            """Update task controls when a specific model is chosen within the filtered list."""
            is_hakka = any(m in model_name for m in _HAKKA_MODELS_IDS)

            task_update = gr.update(
                value="transcribe", interactive=False,
                info="Note: large-v3-turbo only supports Transcribe",
            ) if model_name == "large-v3-turbo" else gr.update(interactive=True, info=None)

            checkbox_update = gr.update(value=True) if is_hakka else gr.update(value=False)

            return (
                task_update,
                gr.update(visible=is_hakka),   # llm_col
                checkbox_update,               # translate_hakka_checkbox
                gr.update(visible=is_hakka),   # llm_prompt_textbox
                gr.update(visible=False),      # translated_col
            )

        model_dropdown.change(
            fn=on_model_change,
            inputs=[model_dropdown],
            outputs=[
                task_radio,
                llm_col,
                translate_hakka_checkbox,
                llm_prompt_textbox,
                translated_col,
            ],
            queue=False,
        )

        use_enhancement_checkbox.change(
            fn=lambda checked: gr.update(interactive=checked),
            inputs=[use_enhancement_checkbox],
            outputs=[enhancement_mix_slider],
            queue=False,
            show_progress="hidden",
        )

        audio_input.change(
            fn=lambda x: "" if x else gr.update(),
            inputs=[audio_input],
            outputs=[youtube_input],
            queue=False,
        )
        youtube_input.change(
            fn=lambda x: None if x else gr.update(),
            inputs=[youtube_input],
            outputs=[audio_input],
            queue=False,
        )

        merge_checkbox.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[merge_checkbox],
            outputs=[max_chars_slider],
            queue=False,
        )
        use_vad_checkbox.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[use_vad_checkbox],
            outputs=[min_silence_slider],
            queue=False,
        )

        # Copy buttons
        _COPY_JS = """(content) => {
            if (!content) return "⚠️ No content to copy";
            navigator.clipboard.writeText(content).then(
                () => "✅ Copied!",
                (err) => "❌ Failed: " + err
            );
            return "✅ Copied!";
        }"""

        asr_copy_btn.click(
            fn=None, inputs=[asr_srt_output], outputs=[asr_copy_status], js=_COPY_JS,
        )
        trans_copy_btn.click(
            fn=None, inputs=[translated_srt_output], outputs=[trans_copy_status], js=_COPY_JS,
        )

        # ── Examples ────────────────────────────────────────────────────
        gr.Markdown("### 📋 Examples")

        # Ground truth textbox — read-only, populated when an example is clicked
        ground_truth_textbox = gr.Textbox(
            label="Ground Truth (客語漢字)",
            info="此範例音檔的標準參考文字",
            interactive=False,
            lines=3,
        )

        _EXAMPLE_DEFAULTS = dict(
            youtube_url="",
            model_size="formospeech/whisper-large-v2-taiwanese-hakka-v1",
            language="hakka",
            task="transcribe",
        )

        gr.Examples(
            examples=[
                [
                    "samples/734a04794010481cb3eed411b6e005cc.wav",
                    _EXAMPLE_DEFAULTS["youtube_url"],
                    _EXAMPLE_DEFAULTS["model_size"],
                    _EXAMPLE_DEFAULTS["language"],
                    _EXAMPLE_DEFAULTS["task"],
                    "下二隻月就愛過年吔，魚仔相關个產品就開始起價，因為呢愛分民眾在防疫期間乜買得著萋萋个魚貨，苗栗魚市場就特別推出咧限量个過年禮盒，用網路，注文還過送貨到屋个服務，還過較便宜个價數，分苗栗鄉親在屋下裡肚，乜買得著萋萋又有保障个魚貨。",
                ],
                [
                    "samples/874062dc1657497b9ac996971c9ce4bb.wav",
                    _EXAMPLE_DEFAULTS["youtube_url"],
                    _EXAMPLE_DEFAULTS["model_size"],
                    _EXAMPLE_DEFAULTS["language"],
                    _EXAMPLE_DEFAULTS["task"],
                    "這隻世界項有當多人高不將愛摎自家个夢想放忒去，你既然做得追求你个夢想，你就愛認真煞猛分佢兜試著當見笑啊。",
                ],
            ],
            inputs=[
                audio_input,
                youtube_input,
                model_dropdown,
                language_selector,
                task_radio,
                ground_truth_textbox,
            ],
            label="客語辨識 + LLM 翻譯範例",
        )

    return app


def main():
    print("\n" + "=" * 60)
    print("🚀 Starting Whisper ASR Service (Improved Version)")
    print("=" * 60 + "\n")

    print("🧹 Cleaning up temporary files...")
    for path in ["/tmp/whisper-downloads", "/tmp/whisper-sessions"]:
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
                print(f"  ✓ Cleaned {path}")
            except Exception as e:
                print(f"  ⚠️  Failed to clean {path}: {e}")

    output_dir = "/app/outputs"
    if os.path.exists(output_dir):
        try:
            for f in glob.glob(os.path.join(output_dir, "*.srt")):
                os.unlink(f)
            print(f"  ✓ Cleaned old SRT files in {output_dir}")
        except Exception as e:
            print(f"  ⚠️  Failed to clean {output_dir}: {e}")

    if LLM_ENABLED:
        print("🤖 LLM enabled — checking Ollama model availability...")
        pull_model_if_needed()
    else:
        print("ℹ️  LLM disabled (ENABLE_LLM=false) — skipping Ollama setup")

    default_model = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
    if os.environ.get("PRELOAD_MODEL", "false").lower() == "true":
        print(f"🔄 Pre-loading model: {default_model}")
        trans, gpu_id = transcriber_pool.get_single_gpu_transcriber(default_model, True, 0.1)
        transcriber_pool.release_single_gpu_transcriber(gpu_id)
        print("✅ Model pre-loaded")

    fastapi_app = FastAPI()

    @fastapi_app.get("/terms-and-privacy")
    async def serve_pdf():
        pdf_path = (
            "/app/docs/Terms_and_Privacy.pdf"
            if os.path.exists("/app/docs/Terms_and_Privacy.pdf")
            else "docs/Terms_and_Privacy.pdf"
        )
        if os.path.exists(pdf_path):
            return FileResponse(
                pdf_path,
                media_type="application/pdf",
                headers={"Content-Disposition": "inline; filename=Terms_and_Privacy.pdf"},
            )
        return {"error": "File not found"}

    gradio_app = create_interface()
    gradio_app.queue(max_size=10, default_concurrency_limit=2)

    # Mount with optional password and API disabled
    _password = os.environ.get("GRADIO_PASSWORD", "").strip()
    mount_kwargs = dict(
        app=fastapi_app,
        blocks=gradio_app,
        path="/",
    )
    if _password:
        mount_kwargs["auth"] = ("admin", _password)
        mount_kwargs["auth_message"] = "請輸入密碼以使用本系統"
        print(f"🔒 Password protection enabled")
    else:
        print("⚠️  GRADIO_PASSWORD not set — running without authentication")
    fastapi_app = gr.mount_gradio_app(**mount_kwargs)

    import uvicorn
    uvicorn.run(
        fastapi_app,
        host=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        port=int(os.environ.get("GRADIO_SERVER_PORT", 7860)),
    )


if __name__ == "__main__":
    main()
