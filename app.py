"""
Gradio-based web interface for Whisper ASR service.
"""

import os
import glob
import tempfile
import time
import shutil
from datetime import datetime, timedelta
from typing import Optional, Tuple, Generator

import gradio as gr
import soundfile as sf

from transcriber import (
    WhisperTranscriber,
    SUPPORTED_LANGUAGES,
    MODEL_SIZES,
    get_gpu_info,
)
from youtube_downloader import (
    is_youtube_url,
    download_audio_with_progress,
    get_video_info,
)
from srt_utils import segments_to_srt, merge_segments


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
"""


# Global transcriber instance
transcriber: Optional[WhisperTranscriber] = None


def cleanup_old_files(max_age_hours: int = 24):
    """Clean up old temporary files and outputs."""
    now = datetime.now()
    
    # Clean /tmp/whisper-downloads
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
    
    # Clean /app/outputs (keep files for 24 hours)
    output_dir = "/app/outputs"
    if os.path.exists(output_dir):
        for f in glob.glob(os.path.join(output_dir, "*.srt")):
            try:
                mtime = datetime.fromtimestamp(os.path.getmtime(f))
                if now - mtime > timedelta(hours=max_age_hours):
                    os.unlink(f)
            except Exception:
                pass


def get_transcriber(
    model_size: str = "large-v3",
    use_vad: bool = True,
) -> WhisperTranscriber:
    """Get or create transcriber instance."""
    global transcriber
    
    if transcriber is None or transcriber.model_size != model_size:
        transcriber = WhisperTranscriber(
            model_size=model_size,
            device=os.environ.get("WHISPER_DEVICE", "cuda"),
            compute_type=os.environ.get("WHISPER_COMPUTE_TYPE", "float16"),
            use_vad=use_vad,
        )
    
    return transcriber


def format_progress_html(percent: int, message: str) -> str:
    """Generate HTML for progress bar."""
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
    merge_subtitles: bool,
    max_chars: int,
) -> Generator[Tuple[str, str, Optional[str]], None, None]:
    """
    Process audio from file or YouTube URL.
    
    Yields:
        Tuple of (status message, SRT content, SRT file path)
    """
    # Record start time
    start_time = time.time()
    
    # Clean up old files periodically
    cleanup_old_files(max_age_hours=24)
    
    audio_path = None
    temp_files = []
    video_title = "output"
    audio_duration = 0.0
    
    try:
        # Determine input source
        if youtube_url and youtube_url.strip():
            if not is_youtube_url(youtube_url):
                yield "❌ Invalid YouTube URL", "", None
                return
            
            yield format_progress_html(5, "Fetching video information..."), "", None
            info = get_video_info(youtube_url)
            if info:
                video_title = info.get("title", "youtube_audio")
                yield format_progress_html(10, f"Downloading: {video_title[:40]}..."), "", None
            
            # Download audio
            audio_path, title = download_audio_with_progress(
                youtube_url,
                progress_callback=None,
            )
            
            yield format_progress_html(30, "Download complete"), "", None
            
            if audio_path is None:
                yield "❌ Download failed. Please check the URL.", "", None
                return
            
            if title:
                video_title = title
            temp_files.append(audio_path)
            
        elif audio_file:
            audio_path = audio_file
            video_title = os.path.splitext(os.path.basename(audio_file))[0]
            yield format_progress_html(10, "Audio file loaded"), "", None
        else:
            yield "❌ Please upload an audio file or enter a YouTube URL", "", None
            return
        
        # Get audio duration
        try:
            audio_info = sf.info(audio_path)
            audio_duration = audio_info.duration
        except Exception as e:
            print(f"Warning: Could not get audio duration: {e}")
            audio_duration = 0.0
        
        # Initialize transcriber
        yield format_progress_html(35, "Loading Whisper model..."), "", None
        trans = get_transcriber(model_size, use_vad)
        
        yield format_progress_html(40, "Model loaded. Starting transcription..."), "", None
        
        # Transcribe with progress updates
        last_progress = [40]  # Use list to allow modification in nested function
        
        def transcribe_progress(pct, msg):
            # Map transcriber progress (0-100) to our range (40-85)
            mapped = 40 + int(pct * 0.45)
            last_progress[0] = mapped
        
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
        
        # Merge segments if requested
        if merge_subtitles:
            yield format_progress_html(90, "Merging subtitle segments..."), "", None
            segments = merge_segments(segments, max_chars=max_chars)
        
        # Generate SRT
        yield format_progress_html(95, "Generating SRT file..."), "", None
        srt_content = segments_to_srt(segments)
        
        # Save SRT file
        output_dir = "/app/outputs" if os.path.exists("/app/outputs") else tempfile.gettempdir()
        
        # Clean filename
        safe_title = "".join(c for c in video_title if c.isalnum() or c in " -_").strip()[:50]
        # Add timestamp to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        srt_filename = f"{safe_title}_{timestamp}.srt"
        srt_path = os.path.join(output_dir, srt_filename)
        
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Format status message with duration and processing time
        status_parts = [f"✅ Transcription complete! {len(segments)} subtitle segments generated."]
        
        if audio_duration > 0:
            status_parts.append(f"Audio duration: {audio_duration:.1f}s")
        
        status_parts.append(f"Processing time: {processing_time:.1f}s")
        
        if audio_duration > 0 and processing_time > 0:
            speed_ratio = audio_duration / processing_time
            status_parts.append(f"Speed: {speed_ratio:.2f}x realtime")
        
        status = " | ".join(status_parts)
        yield status, srt_content, srt_path
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        yield f"❌ Error: {str(e)}", "", None
    
    finally:
        # Cleanup temp files
        for f in temp_files:
            if f and os.path.exists(f):
                try:
                    os.unlink(f)
                except:
                    pass


def get_system_info() -> str:
    """Get system and GPU information."""
    info_lines = ["**Source:** [王新民](https://homepage.iis.sinica.edu.tw/pages/whm/index_zh.html) 教授（中央研究院資訊科學研究所）\n"]
    
    gpu_info = get_gpu_info()
    if gpu_info:
        info_lines.append(f"**GPU Count:** {len(gpu_info)}\n")
        for gpu in gpu_info:
            info_lines.append(
                f"- GPU {gpu['index']}: {gpu['name']} "
                f"({gpu['memory_total']:.1f} GB)"
            )
    else:
        info_lines.append("**GPU:** No GPU available. Using CPU mode.")
    
    return "\n".join(info_lines)


# Build Gradio interface
def create_interface() -> gr.Blocks:
    """Create and return Gradio interface."""
    
    with gr.Blocks(
        title="Medical and Pharmaceutical ASR with Whisper",
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
    ) as app:
        
        gr.Markdown(
            """
            # 🎙️ Medical and Pharmaceutical ASR with Whisper
            
            Note: large-v3-turbo is for transcription only.
            """
        )
        
        with gr.Row():
            # Left column: Input
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
                
                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        choices=MODEL_SIZES,
                        value=os.environ.get("WHISPER_MODEL", "large-v3-turbo"),
                        label="Model Size",
                    )
                    
                    language_dropdown = gr.Dropdown(
                        choices=list(SUPPORTED_LANGUAGES.keys()),
                        value="en",
                        label="Language",
                    )
                
                with gr.Row():
                    task_radio = gr.Radio(
                        choices=[
                            ("Transcribe", "transcribe"),
                            ("Translate to English", "translate"),
                        ],
                        value="transcribe",
                        label="Task",
                    )
                
                with gr.Row():
                    use_vad_checkbox = gr.Checkbox(
                        value=True,
                        label="Enable VAD (Voice Activity Detection)",
                    )
                    merge_checkbox = gr.Checkbox(
                        value=True,
                        label="Merge Short Subtitles",
                    )
                
                max_chars_slider = gr.Slider(
                    minimum=40,
                    maximum=120,
                    value=80,
                    step=10,
                    label="Max Characters Per Line",
                    visible=True,
                )
                
                process_btn = gr.Button(
                    "🚀 Start Transcription",
                    variant="primary",
                    size="lg",
                )
            
            # Right column: Output
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Output")
                
                status_text = gr.HTML("Waiting for input...")
                
                srt_output = gr.Textbox(
                    label="SRT Subtitle Content",
                    lines=20,
                    max_lines=30,
                )
                
                srt_file = gr.File(
                    label="Download SRT File",
                )
        
        # System info
        with gr.Accordion("System Information", open=False):
            system_info = gr.Markdown(get_system_info())
        
        # Language mapping display
        with gr.Accordion("Supported Languages", open=False):
            lang_info = "\n".join(
                f"- `{code}`: {name}"
                for code, name in SUPPORTED_LANGUAGES.items()
            )
            gr.Markdown(lang_info)
        
        # Event handlers
        process_btn.click(
            fn=process_audio,
            inputs=[
                audio_input,
                youtube_input,
                model_dropdown,
                language_dropdown,
                task_radio,
                use_vad_checkbox,
                merge_checkbox,
                max_chars_slider,
            ],
            outputs=[status_text, srt_output, srt_file],
        )
        
        # Clear YouTube when audio uploaded and vice versa
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
        
        # Toggle max_chars visibility based on merge checkbox
        merge_checkbox.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[merge_checkbox],
            outputs=[max_chars_slider],
        )
    
    return app


def main():
    """Main entry point."""
    # Clean up old files on startup
    cleanup_old_files(max_age_hours=24)
    
    # Pre-load model if specified
    default_model = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
    preload = os.environ.get("PRELOAD_MODEL", "false").lower() == "true"
    
    if preload:
        print(f"Pre-loading model: {default_model}")
        get_transcriber(default_model)
    
    # Create and launch app with queue for concurrent requests
    app = create_interface()
    
    # Enable queue for handling multiple users
    # Note: Transcription is sequential due to GPU memory constraints
    app.queue(max_size=10)
    
    app.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", 7860)),
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
