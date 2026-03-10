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
from parallel_transcriber import ParallelWhisperTranscriber
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
    DEFAULT_SYSTEM_PROMPT,
)

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
    Thread-safe pool for managing transcriber instances.
    Ensures each concurrent request can use an isolated transcriber.
    """

    def __init__(self, max_workers: int = 2):
        self.max_workers = max_workers
        self.lock = Lock()
        self.single_gpu_pool: Dict[str, WhisperTranscriber] = {}
        self.parallel_gpu_pool: Dict[str, ParallelWhisperTranscriber] = {}
        self.available_single = []
        self.available_parallel = []

    def get_single_gpu_transcriber(
        self,
        model_size: str,
        use_vad: bool,
        min_silence_duration_s: float,
    ) -> Tuple[WhisperTranscriber, str]:
        min_silence_duration_ms = int(min_silence_duration_s * 1000)

        with self.lock:
            for worker_id in self.available_single[:]:
                trans = self.single_gpu_pool.get(worker_id)
                if trans and trans.model_size == model_size:
                    self.available_single.remove(worker_id)
                    print(f"♻️  Reusing single-GPU transcriber: {worker_id}")
                    return trans, worker_id

            if len(self.single_gpu_pool) < self.max_workers:
                worker_id = f"single_{uuid.uuid4().hex[:8]}"

                device = os.environ.get("WHISPER_DEVICE", "cuda")
                if device == "cuda" and torch.cuda.is_available():
                    torch.cuda.set_device(0)

                trans = WhisperTranscriber(
                    model_size=model_size,
                    device=device,
                    compute_type=os.environ.get("WHISPER_COMPUTE_TYPE", "float16"),
                    use_vad=use_vad,
                    min_silence_duration_ms=min_silence_duration_ms,
                )

                self.single_gpu_pool[worker_id] = trans
                print(f"✨ Created new single-GPU transcriber: {worker_id}")
                return trans, worker_id

            print("⏳ Waiting for available transcriber...")
            if self.available_single:
                worker_id = self.available_single.pop(0)
                return self.single_gpu_pool[worker_id], worker_id

            worker_id = list(self.single_gpu_pool.keys())[0]
            return self.single_gpu_pool[worker_id], worker_id

    def release_single_gpu_transcriber(self, worker_id: str):
        with self.lock:
            if (
                worker_id in self.single_gpu_pool
                and worker_id not in self.available_single
            ):
                self.available_single.append(worker_id)
                print(f"✅ Released single-GPU transcriber: {worker_id}")

    def get_parallel_transcriber(
        self,
        model_size: str,
        min_silence_duration_s: float,
    ) -> Tuple[ParallelWhisperTranscriber, str]:
        min_silence_duration_ms = int(min_silence_duration_s * 1000)

        with self.lock:
            for worker_id in self.available_parallel[:]:
                trans = self.parallel_gpu_pool.get(worker_id)
                if trans and trans.model_size == model_size:
                    self.available_parallel.remove(worker_id)
                    print(f"♻️  Reusing parallel transcriber: {worker_id}")
                    return trans, worker_id

            worker_id = f"parallel_{uuid.uuid4().hex[:8]}"

            gpu_ids_str = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3")
            gpu_ids = [int(x.strip()) for x in gpu_ids_str.split(",") if x.strip()]

            trans = ParallelWhisperTranscriber(
                model_size=model_size,
                compute_type=os.environ.get("WHISPER_COMPUTE_TYPE", "float16"),
                gpu_ids=gpu_ids,
                min_silence_duration_ms=min_silence_duration_ms,
            )

            self.parallel_gpu_pool[worker_id] = trans
            print(f"✨ Created new parallel transcriber: {worker_id}")
            return trans, worker_id

    def release_parallel_transcriber(self, worker_id: str):
        with self.lock:
            if (
                worker_id in self.parallel_gpu_pool
                and worker_id not in self.available_parallel
            ):
                self.available_parallel.append(worker_id)
                print(f"✅ Released parallel transcriber: {worker_id}")


# Global transcriber pool
transcriber_pool = TranscriberPool(max_workers=2)


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
    use_multi_gpu: bool,
    translate_hakka: bool = False,
    llm_system_prompt: str = "",
) -> Generator[Tuple[str, str, Optional[str]], None, None]:
    session_id = uuid.uuid4().hex[:12]
    session_dir = os.path.join("/tmp/whisper-sessions", session_id)
    os.makedirs(session_dir, exist_ok=True)

    start_time = time.time()
    audio_path = None
    temp_files = []
    video_title = "output"
    audio_duration = 0.0
    worker_id = None
    is_parallel = False

    print(f"\n{'=' * 60}")
    print(f"🎬 Starting session: {session_id}")
    print(f"{'=' * 60}\n")

    try:
        if youtube_url and youtube_url.strip():
            if not is_youtube_url(youtube_url):
                yield "❌ Invalid YouTube URL", "", None
                return

            yield format_progress_html(5, "Fetching video information..."), "", None
            info = get_video_info(youtube_url)
            if info:
                video_title = info.get("title", "youtube_audio")
                yield (
                    format_progress_html(10, f"Downloading: {video_title[:40]}..."),
                    "", None,
                )

            download_dir = os.path.join(session_dir, "downloads")
            os.makedirs(download_dir, exist_ok=True)

            audio_path, title = download_audio_with_progress(
                youtube_url, output_dir=download_dir, progress_callback=None,
            )

            yield format_progress_html(30, "Download complete"), "", None

            if audio_path is None:
                yield "❌ Download failed. Please check the URL.", "", None
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
            yield (
                format_progress_html(10, "Audio file loaded and copied to session"),
                "", None,
            )
            print(f"📁 Uploaded file copied to session: {upload_copy}")
        else:
            yield "❌ Please upload an audio file or enter a YouTube URL", "", None
            return

        try:
            audio_info = sf.info(audio_path)
            audio_duration = audio_info.duration
            print(f"⏱️  Audio duration: {audio_duration:.1f}s")
        except Exception as e:
            print(f"Warning: Could not get audio duration: {e}")
            audio_duration = 0.0

        use_parallel = use_multi_gpu and audio_duration >= 300
        num_gpus_used = 1

        if use_parallel:
            is_parallel = True
            yield (format_progress_html(35, "Loading models on multiple GPUs..."), "", None)

            para_trans, worker_id = transcriber_pool.get_parallel_transcriber(
                model_size, min_silence_duration_s
            )
            num_gpus_used = para_trans.num_gpus

            yield (
                format_progress_html(40, f"Starting parallel transcription on {num_gpus_used} GPUs..."),
                "", None,
            )
            print(f"🚀 Using parallel transcriber: {worker_id} ({num_gpus_used} GPUs)")

            def transcribe_progress(pct, msg):
                pass

            segments = para_trans.transcribe_parallel(
                audio_path,
                language=language if language != "auto" else None,
                task=task,
                progress_callback=transcribe_progress,
            )
        else:
            is_parallel = False
            yield (format_progress_html(35, "Loading Whisper model on GPU 0..."), "", None)

            trans, worker_id = transcriber_pool.get_single_gpu_transcriber(
                model_size, use_vad, min_silence_duration_s
            )

            yield (
                format_progress_html(40, "Model loaded on GPU 0. Starting transcription..."),
                "", None,
            )
            print(f"🔧 Using single-GPU transcriber: {worker_id}")

            last_progress = [40]

            def transcribe_progress(pct, msg):
                last_progress[0] = 40 + int(pct * 0.45)

            segments = trans.transcribe(
                audio_path,
                language=language if language != "auto" else None,
                task=task,
                progress_callback=transcribe_progress,
            )

        yield format_progress_html(85, "Transcription complete"), "", None

        if not segments:
            yield "⚠️ No speech detected", "", None
            return

        print(f"📝 Generated {len(segments)} segments")

        # Translate Hakka to Mandarin via LLM (formospeech models only)
        is_hakka_model = any(m in model_size for m in [
            "formospeech/whisper-large-v2-taiwanese-hakka-v1",
            "formospeech/whisper-large-v3-taiwanese-hakka",
        ])
        if translate_hakka and is_hakka_model and LLM_ENABLED:
            yield (format_progress_html(86, "Translating Hakka → Mandarin via LLM..."), "", None)
            segments = translate_segments(segments, system_prompt=llm_system_prompt)
            print("✅ Hakka translation complete")
        elif translate_hakka and not LLM_ENABLED:
            print("⚠️  LLM translation requested but ENABLE_LLM=false — skipping")

        # Convert to Traditional Chinese if requested
        if language == "zh" and convert_to_traditional:
            converter = get_converter()
            if converter.is_available():
                yield (format_progress_html(88, "Converting to Traditional Chinese..."), "", None)
                segments = convert_segments_to_traditional(segments)
                print("✅ Converted to Traditional Chinese")
            else:
                print("⚠️  Chinese converter not available, skipping conversion")

        # Merge segments if requested
        if merge_subtitles:
            yield format_progress_html(90, "Merging subtitle segments..."), "", None
            original_count = len(segments)
            segments = merge_segments(segments, max_chars=max_chars)
            print(f"🔗 Merged from {original_count} to {len(segments)} segments")

        # Generate SRT
        yield format_progress_html(95, "Generating SRT file..."), "", None
        srt_content = segments_to_srt(segments)

        output_dir = (
            "/app/outputs" if os.path.exists("/app/outputs") else tempfile.gettempdir()
        )
        os.makedirs(output_dir, exist_ok=True)

        safe_title = "".join(
            c for c in video_title if c.isalnum() or c in " -_"
        ).strip()[:40]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:6]
        srt_filename = f"{safe_title}_{timestamp}_{unique_id}.srt"
        srt_path = os.path.join(output_dir, srt_filename)

        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        print(f"💾 SRT saved: {srt_path}")

        processing_time = time.time() - start_time
        gpu_info = f"{num_gpus_used} GPUs" if use_parallel else "GPU 0 (single)"
        status_parts = [f"✅ Transcription complete! {len(segments)} subtitle segments generated.\n"]
        status_parts.append(f"Session: {session_id}")
        status_parts.append(f"Mode: {gpu_info}")
        if audio_duration > 0:
            status_parts.append(f"Audio duration: {audio_duration:.1f}s")
        status_parts.append(f"Processing time: {processing_time:.1f}s")
        if audio_duration > 0 and processing_time > 0:
            status_parts.append(f"Speed: {audio_duration / processing_time:.2f}x realtime")

        print(f"\n{'=' * 60}")
        print(f"✅ Session completed: {session_id}")
        print(f"⏱️  Total time: {processing_time:.1f}s")
        print(f"{'=' * 60}\n")

        yield " | ".join(status_parts), srt_content, srt_path

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n❌ Session failed: {session_id}\nError: {str(e)}\n")
        yield f"❌ Error in session {session_id}: {str(e)}", "", None

    finally:
        if worker_id:
            if is_parallel:
                transcriber_pool.release_parallel_transcriber(worker_id)
            else:
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


# Build Gradio interface
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

            ### Developers
            - **[李鴻欣 Hung-Shin Lee](https://www.linkedin.com/in/hungshinlee)**（聯和科創股份有限公司）
            - **[陳力瑋 Li-Wei Chen](mailto:wayne900619@gmail.com)**（國立清華大學資訊工程研究所）
            ### Contributors
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
                '<a href="/terms-and-privacy" target="_blank">使用者條款、資訊安全與隱私權政策 (Terms and Privacy Policy)</a>'
            )

        with gr.Row():
            # ── Left column: Input & Settings ──────────────────────────────
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

                model_dropdown = gr.Dropdown(
                    choices=[
                        (MODEL_CONFIGS[model_id]["display_name"], model_id)
                        for model_id in MODEL_SIZES
                    ],
                    value=os.environ.get("WHISPER_MODEL", "large-v3-turbo"),
                    label="Model",
                )

                default_model = os.environ.get("WHISPER_MODEL", "large-v3-turbo")

                if any(m in default_model for m in [
                    "formospeech/whisper-large-v2-taiwanese-hakka-v1",
                    "formospeech/whisper-large-v3-taiwanese-hakka",
                ]):
                    language_interactive = False
                    language_value = "zh"
                    language_info = "Note: This model only supports Mandarin"
                else:
                    language_interactive = True
                    language_value = "auto"
                    language_info = None

                with gr.Row():
                    language_radio = gr.Radio(
                        choices=[(name, code) for code, name in SUPPORTED_LANGUAGES.items()],
                        value=language_value,
                        label="Language",
                        interactive=language_interactive,
                        info=language_info,
                    )

                task_interactive = default_model != "large-v3-turbo"

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
                    use_vad_checkbox = gr.Checkbox(value=True, label="Enable VAD")
                    merge_checkbox = gr.Checkbox(value=True, label="Merge Short Subtitles")
                    zh_conv_checkbox = gr.Checkbox(value=False, label="Convert to zh-TW")

                # ── LLM controls — wrapped in a Column so visibility is
                #    controlled at the container level. This avoids Gradio's
                #    unreliable event handling on hidden Checkbox components.
                with gr.Column(visible=False) as llm_col:
                    translate_hakka_checkbox = gr.Checkbox(
                        value=False,
                        label="🤖 Translate Hakka → Mandarin (via Ollama LLM)",
                        interactive=LLM_ENABLED,
                        info=None if LLM_ENABLED else "LLM not deployed (ENABLE_LLM=false)",
                    )
                    llm_prompt_textbox = gr.Textbox(
                        value=DEFAULT_SYSTEM_PROMPT,
                        label="🤖 LLM System Prompt",
                        info="可自訂翻譯指令，留空則使用預設 Prompt",
                        lines=6,
                        visible=False,
                    )

                min_silence_slider = gr.Slider(
                    minimum=0.01, maximum=2.0, value=0.1, step=0.01,
                    label="VAD: Minimum Silence Duration (seconds)",
                    visible=True,
                )

                multi_gpu_checkbox = gr.Checkbox(
                    value=True,
                    label="Use Multi-GPU Parallel Processing (for audio > 5 min)",
                )

                max_chars_slider = gr.Slider(
                    minimum=40, maximum=120, value=80, step=10,
                    label="Max Characters Per Line",
                    visible=True,
                )

                process_btn = gr.Button("🚀 Start", variant="primary", size="lg")

            # ── Right column: Output ────────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Output")

                status_text = gr.HTML("Waiting for input...")

                srt_output = gr.Textbox(
                    label="SRT Subtitle Content",
                    lines=20,
                    max_lines=30,
                )

                with gr.Row():
                    copy_btn = gr.Button("Copy to Clipboard", elem_classes="copy-button")
                    copy_status = gr.HTML("", elem_classes="copy-success")

                srt_file = gr.File(label="Download SRT File")

        # ── Event handlers ──────────────────────────────────────────────────

        process_btn.click(
            fn=process_audio,
            inputs=[
                audio_input,
                youtube_input,
                model_dropdown,
                language_radio,
                task_radio,
                use_vad_checkbox,
                min_silence_slider,
                merge_checkbox,
                zh_conv_checkbox,
                max_chars_slider,
                multi_gpu_checkbox,
                translate_hakka_checkbox,
                llm_prompt_textbox,
            ],
            outputs=[status_text, srt_output, srt_file],
        )

        HAKKA_MODELS = [
            "formospeech/whisper-large-v2-taiwanese-hakka-v1",
            "formospeech/whisper-large-v3-taiwanese-hakka",
        ]

        def on_model_change(model_name):
            is_hakka = any(m in model_name for m in HAKKA_MODELS)

            language_update = gr.update(
                value="zh", interactive=False,
                info="Note: This model only supports Mandarin",
            ) if is_hakka else gr.update(interactive=True, info=None)

            task_update = gr.update(
                value="transcribe", interactive=False,
                info="Note: large-v3-turbo only supports Transcribe",
            ) if model_name == "large-v3-turbo" else gr.update(interactive=True, info=None)

            llm_col_update = gr.update(visible=is_hakka)
            checkbox_reset = gr.update(value=False)
            prompt_hide    = gr.update(visible=False)

            return language_update, task_update, llm_col_update, checkbox_reset, prompt_hide

        model_dropdown.change(
            fn=on_model_change,
            inputs=[model_dropdown],
            outputs=[language_radio, task_radio, llm_col, translate_hakka_checkbox, llm_prompt_textbox],
        )

        translate_hakka_checkbox.change(
            fn=lambda checked: gr.update(visible=checked),
            inputs=[translate_hakka_checkbox],
            outputs=[llm_prompt_textbox],
        )

        audio_input.change(
            fn=lambda x: "" if x else gr.update(),
            inputs=[audio_input],
            outputs=[youtube_input],
        )
        youtube_input.change(
            fn=lambda x: None if x else gr.update(),
            inputs=[youtube_input],
            outputs=[audio_input],
        )

        merge_checkbox.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[merge_checkbox],
            outputs=[max_chars_slider],
        )
        use_vad_checkbox.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[use_vad_checkbox],
            outputs=[min_silence_slider],
        )

        copy_btn.click(
            fn=None,
            inputs=[srt_output],
            outputs=[copy_status],
            js="""(srt_content) => {
                if (!srt_content) return "⚠️ No content to copy";
                navigator.clipboard.writeText(srt_content).then(
                    () => "✅ Copied to clipboard!",
                    (err) => "❌ Failed to copy: " + err
                );
                return "✅ Copied to clipboard!";
            }""",
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
        trans, worker_id = transcriber_pool.get_single_gpu_transcriber(default_model, True, 0.1)
        transcriber_pool.release_single_gpu_transcriber(worker_id)
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
    fastapi_app = gr.mount_gradio_app(fastapi_app, gradio_app, path="/")

    import uvicorn
    uvicorn.run(
        fastapi_app,
        host=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        port=int(os.environ.get("GRADIO_SERVER_PORT", 7860)),
    )


if __name__ == "__main__":
    main()
