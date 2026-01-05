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
import torch

from transcriber import (
    WhisperTranscriber,
    SUPPORTED_LANGUAGES,
    MODEL_SIZES,
    get_gpu_info,
)
from parallel_transcriber import (
    ParallelWhisperTranscriber,
    transcribe_with_multiple_gpus,
)
from youtube_downloader import (
    is_youtube_url,
    download_audio_with_progress,
    get_video_info,
)
from srt_utils import segments_to_srt, merge_segments
from chinese_converter import convert_segments_to_traditional, get_converter


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


# Global transcriber instances
transcriber: Optional[WhisperTranscriber] = None
parallel_transcriber: Optional[ParallelWhisperTranscriber] = None


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
    min_silence_duration_s: float = 0.1,
) -> WhisperTranscriber:
    """Get or create single-GPU transcriber instance (uses only GPU 0)."""
    global transcriber
    
    # Convert seconds to milliseconds
    min_silence_duration_ms = int(min_silence_duration_s * 1000)
    
    if transcriber is None or transcriber.model_size != model_size:
        # For single GPU mode, set GPU 0 as default device
        device = os.environ.get("WHISPER_DEVICE", "cuda")
        
        # Set PyTorch default GPU to 0 for single-GPU mode
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.set_device(0)  # Set GPU 0 as default
        
        transcriber = WhisperTranscriber(
            model_size=model_size,
            device=device,  # Use "cuda" not "cuda:0"
            compute_type=os.environ.get("WHISPER_COMPUTE_TYPE", "float16"),
            use_vad=use_vad,
            min_silence_duration_ms=min_silence_duration_ms,
        )
    
    return transcriber


def get_parallel_transcriber(
    model_size: str = "large-v3",
    min_silence_duration_s: float = 0.1,
) -> ParallelWhisperTranscriber:
    """Get or create multi-GPU transcriber instance."""
    global parallel_transcriber
    
    # Convert seconds to milliseconds
    min_silence_duration_ms = int(min_silence_duration_s * 1000)
    
    if parallel_transcriber is None or parallel_transcriber.model_size != model_size:
        # Parse GPU IDs from environment
        gpu_ids_str = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3")
        gpu_ids = [int(x.strip()) for x in gpu_ids_str.split(",") if x.strip()]
        
        parallel_transcriber = ParallelWhisperTranscriber(
            model_size=model_size,
            compute_type=os.environ.get("WHISPER_COMPUTE_TYPE", "float16"),
            gpu_ids=gpu_ids,
            min_silence_duration_ms=min_silence_duration_ms,
        )
    
    return parallel_transcriber


def load_vocabulary(vocab_file: Optional[str], max_words: int = 30) -> Tuple[Optional[str], str]:
    """
    Load vocabulary file and create initial prompt.
    
    Args:
        vocab_file: Path to vocabulary.txt file
        max_words: Maximum number of words to include (default: 30)
    
    Returns:
        Tuple of (initial_prompt, status_message)
    """
    if not vocab_file or not os.path.exists(vocab_file):
        return None, ""
    
    try:
        with open(vocab_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Extract valid words (non-empty, non-comment lines)
        words = []
        for line in lines:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            # Take only the first word if line contains multiple fields
            word = line.split('|')[0].strip()
            if word:
                words.append(word)
        
        if not words:
            return None, "⚠️ Vocabulary file is empty"
        
        # Take first max_words
        selected_words = words[:max_words]
        
        # Create initial prompt
        initial_prompt = f"重要詞彙：{', '.join(selected_words)}"
        
        status_msg = f"✅ Loaded {len(selected_words)} words from vocabulary (total: {len(words)} words)"
        print(f"📚 {status_msg}")
        print(f"📝 Initial prompt: {initial_prompt[:100]}...")
        
        return initial_prompt, status_msg
        
    except Exception as e:
        error_msg = f"❌ Error loading vocabulary: {str(e)}"
        print(error_msg)
        return None, error_msg


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
    min_silence_duration_s: float,
    merge_subtitles: bool,
    max_chars: int,
    use_multi_gpu: bool,
    vocabulary_file: Optional[str] = None,
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
    
    # Load vocabulary if provided
    initial_prompt, vocab_status = load_vocabulary(vocabulary_file, max_words=30)
    if vocab_status:
        yield format_progress_html(2, vocab_status), "", None
    
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
        
        # Decide whether to use multi-GPU based on audio duration and user choice
        use_parallel = use_multi_gpu and audio_duration >= 300  # 5+ minutes
        num_gpus_used = 1
        
        if use_parallel:
            # Multi-GPU parallel processing
            yield format_progress_html(35, "Loading models on multiple GPUs..."), "", None
            para_trans = get_parallel_transcriber(model_size, min_silence_duration_s)
            num_gpus_used = para_trans.num_gpus
            
            yield format_progress_html(40, f"Starting parallel transcription on {num_gpus_used} GPUs..."), "", None
            
            def transcribe_progress(pct, msg):
                pass  # Progress handled internally
            
            segments = para_trans.transcribe_parallel(
                audio_path,
                language=language if language != "auto" else None,
                task=task,
                initial_prompt=initial_prompt,
                progress_callback=transcribe_progress,
            )
        else:
            # Single GPU processing (uses only GPU 0)
            yield format_progress_html(35, "Loading Whisper model on GPU 0..."), "", None
            trans = get_transcriber(model_size, use_vad, min_silence_duration_s)
            
            yield format_progress_html(40, "Model loaded on GPU 0. Starting transcription..."), "", None
            
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
                initial_prompt=initial_prompt,
                progress_callback=transcribe_progress,
            )
        
        yield format_progress_html(85, "Transcription complete"), "", None
        
        if not segments:
            yield "⚠️ No speech detected", "", None
            return
        
        # Convert to Traditional Chinese if language is Chinese
        if language == "zh":
            converter = get_converter()
            if converter.is_available():
                yield format_progress_html(87, "Converting to Traditional Chinese..."), "", None
                segments = convert_segments_to_traditional(segments)
                print("✅ Converted to Traditional Chinese")
            else:
                print("⚠️  Chinese converter not available, skipping conversion")
        
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
        gpu_info = f"{num_gpus_used} GPUs" if use_parallel else "GPU 0 (single)"
        status_parts = [f"✅ Transcription complete! {len(segments)} subtitle segments generated.\n"]
        
        status_parts.append(f"Mode: {gpu_info}")
        
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
            
            Note: large-v3-turbo is for "**transcribe**" only.
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
                
                min_silence_slider = gr.Slider(
                    minimum=0.01,
                    maximum=2.0,
                    value=0.1,
                    step=0.01,
                    label="VAD: Min Silence Duration (seconds)",
                    info="Minimum silence duration to split segments (default: 0.1s)",
                    visible=True,
                )
                
                multi_gpu_checkbox = gr.Checkbox(
                    value=True,
                    label="🚀 Use Multi-GPU Parallel Processing (for audio > 5 min)",
                    info="Automatically enables for long audio files",
                )
                
                max_chars_slider = gr.Slider(
                    minimum=40,
                    maximum=120,
                    value=80,
                    step=10,
                    label="Max Characters Per Line",
                    visible=True,
                )
                
                gr.Markdown("### 📚 Vocabulary (Optional)")
                
                vocabulary_input = gr.File(
                    label="Upload Vocabulary File (vocabulary.txt)",
                    file_types=[".txt"],
                    type="filepath",
                )
                
                gr.Markdown(
                    """💡 **Tip:** Upload a vocabulary.txt file with one word per line.
                    The first 30 words will be used as hints for Whisper.
                    
                    Example:
                    ```
                    受洗
                    禱告
                    聖經
                    見證
                    恩典
                    ```
                    """
                )
                
                process_btn = gr.Button(
                    "🚀 Start",
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
                
                with gr.Row():
                    copy_btn = gr.Button(
                        "📋 Copy to Clipboard",
                        elem_classes="copy-button",
                    )
                    copy_status = gr.HTML("", elem_classes="copy-success")
                
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
                min_silence_slider,
                merge_checkbox,
                max_chars_slider,
                multi_gpu_checkbox,
                vocabulary_input,
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
        
        # Toggle min_silence visibility based on VAD checkbox
        use_vad_checkbox.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[use_vad_checkbox],
            outputs=[min_silence_slider],
        )
        
        # Copy to clipboard functionality
        copy_btn.click(
            fn=None,
            inputs=[srt_output],
            outputs=[copy_status],
            js="""(srt_content) => {
                if (!srt_content) {
                    return "⚠️ No content to copy";
                }
                navigator.clipboard.writeText(srt_content).then(
                    () => {
                        return "✅ Copied to clipboard!";
                    },
                    (err) => {
                        return "❌ Failed to copy: " + err;
                    }
                );
                // Return immediately for UI feedback
                return "✅ Copied to clipboard!";
            }"""
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
    app.queue(
        max_size=10,
        default_concurrency_limit=2,  # Allow 2 concurrent uploads
    )
    
    app.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", 7860)),
        share=False,
        show_error=True,
        max_file_size="500mb",  # Increase max file size to 500MB
    )


if __name__ == "__main__":
    main()
