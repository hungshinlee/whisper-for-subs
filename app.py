"""
Gradio-based web interface for Whisper ASR service.
"""

import os
import tempfile
import time
from typing import Optional, Tuple

import gradio as gr

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


# Global transcriber instance
transcriber: Optional[WhisperTranscriber] = None


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


def process_audio(
    audio_file: Optional[str],
    youtube_url: str,
    model_size: str,
    language: str,
    task: str,
    use_vad: bool,
    merge_subtitles: bool,
    max_chars: int,
    progress=gr.Progress(),
) -> Tuple[str, Optional[str], str]:
    """
    Process audio from file or YouTube URL.
    
    Returns:
        Tuple of (SRT content, SRT file path, status message)
    """
    audio_path = None
    temp_files = []
    video_title = "output"
    
    try:
        # Determine input source
        if youtube_url and youtube_url.strip():
            if not is_youtube_url(youtube_url):
                return "", None, "❌ 無效的 YouTube 網址"
            
            progress(0.05, desc="取得影片資訊...")
            info = get_video_info(youtube_url)
            if info:
                video_title = info.get("title", "youtube_audio")
                progress(0.1, desc=f"下載中: {video_title[:50]}...")
            
            # Download audio
            def download_progress(percent, msg):
                progress(0.1 + percent * 0.2 / 100, desc=msg)
            
            audio_path, title = download_audio_with_progress(
                youtube_url,
                progress_callback=download_progress,
            )
            
            if audio_path is None:
                return "", None, "❌ 下載失敗，請確認網址是否正確"
            
            if title:
                video_title = title
            temp_files.append(audio_path)
            
        elif audio_file:
            audio_path = audio_file
            video_title = os.path.splitext(os.path.basename(audio_file))[0]
        else:
            return "", None, "❌ 請上傳音檔或輸入 YouTube 網址"
        
        # Initialize transcriber
        progress(0.3, desc="載入模型中...")
        trans = get_transcriber(model_size, use_vad)
        
        # Transcribe
        def transcribe_progress(pct, msg):
            progress(0.3 + pct * 0.6 / 100, desc=msg)
        
        segments = trans.transcribe(
            audio_path,
            language=language if language != "auto" else None,
            task=task,
            progress_callback=transcribe_progress,
        )
        
        if not segments:
            return "", None, "⚠️ 未偵測到語音內容"
        
        # Merge segments if requested
        if merge_subtitles:
            progress(0.92, desc="合併字幕段落...")
            segments = merge_segments(segments, max_chars=max_chars)
        
        # Generate SRT
        progress(0.95, desc="生成 SRT 檔案...")
        srt_content = segments_to_srt(segments)
        
        # Save SRT file
        output_dir = "/app/outputs" if os.path.exists("/app/outputs") else tempfile.gettempdir()
        
        # Clean filename
        safe_title = "".join(c for c in video_title if c.isalnum() or c in " -_").strip()[:50]
        srt_filename = f"{safe_title}.srt"
        srt_path = os.path.join(output_dir, srt_filename)
        
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)
        
        progress(1.0, desc="完成！")
        
        status = f"✅ 轉錄完成！共 {len(segments)} 個字幕段落"
        return srt_content, srt_path, status
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return "", None, f"❌ 錯誤: {str(e)}"
    
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
    info_lines = ["### 系統資訊\n"]
    
    gpu_info = get_gpu_info()
    if gpu_info:
        info_lines.append(f"**GPU 數量:** {len(gpu_info)}\n")
        for gpu in gpu_info:
            info_lines.append(
                f"- GPU {gpu['index']}: {gpu['name']} "
                f"({gpu['memory_total']:.1f} GB)"
            )
    else:
        info_lines.append("**GPU:** 無可用 GPU，使用 CPU 模式")
    
    return "\n".join(info_lines)


# Build Gradio interface
def create_interface() -> gr.Blocks:
    """Create and return Gradio interface."""
    
    with gr.Blocks(
        title="Whisper ASR 字幕生成服務",
        theme=gr.themes.Soft(),
    ) as app:
        
        gr.Markdown(
            """
            # 🎙️ Whisper ASR 字幕生成服務
            
            上傳音檔、影片，或輸入 YouTube 網址，自動生成 SRT 字幕檔。
            """
        )
        
        with gr.Row():
            # Left column: Input
            with gr.Column(scale=1):
                gr.Markdown("### 📥 輸入")
                
                audio_input = gr.Audio(
                    label="上傳音檔或影片",
                    type="filepath",
                    sources=["upload", "microphone"],
                )
                
                gr.Markdown("**或**")
                
                youtube_input = gr.Textbox(
                    label="YouTube 網址",
                    placeholder="https://www.youtube.com/watch?v=...",
                )
                
                gr.Markdown("### ⚙️ 設定")
                
                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        choices=MODEL_SIZES,
                        value=os.environ.get("WHISPER_MODEL", "large-v3"),
                        label="模型大小",
                    )
                    
                    language_dropdown = gr.Dropdown(
                        choices=list(SUPPORTED_LANGUAGES.keys()),
                        value="auto",
                        label="語言",
                    )
                
                with gr.Row():
                    task_radio = gr.Radio(
                        choices=[
                            ("轉錄 (Transcribe)", "transcribe"),
                            ("翻譯成英文 (Translate)", "translate"),
                        ],
                        value="transcribe",
                        label="功能",
                    )
                
                with gr.Row():
                    use_vad_checkbox = gr.Checkbox(
                        value=True,
                        label="使用 VAD 語音偵測",
                    )
                    merge_checkbox = gr.Checkbox(
                        value=True,
                        label="合併短字幕",
                    )
                
                max_chars_slider = gr.Slider(
                    minimum=40,
                    maximum=120,
                    value=80,
                    step=10,
                    label="每行最大字數",
                    visible=True,
                )
                
                process_btn = gr.Button(
                    "🚀 開始轉錄",
                    variant="primary",
                    size="lg",
                )
            
            # Right column: Output
            with gr.Column(scale=1):
                gr.Markdown("### 📤 輸出")
                
                status_text = gr.Markdown("等待輸入...")
                
                srt_output = gr.Textbox(
                    label="SRT 字幕內容",
                    lines=20,
                    max_lines=30,
                )
                
                srt_file = gr.File(
                    label="下載 SRT 檔案",
                )
        
        # System info
        with gr.Accordion("系統資訊", open=False):
            system_info = gr.Markdown(get_system_info())
        
        # Language mapping display
        with gr.Accordion("支援語言列表", open=False):
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
            outputs=[srt_output, srt_file, status_text],
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
    # Pre-load model if specified
    default_model = os.environ.get("WHISPER_MODEL", "large-v3")
    preload = os.environ.get("PRELOAD_MODEL", "false").lower() == "true"
    
    if preload:
        print(f"Pre-loading model: {default_model}")
        get_transcriber(default_model)
    
    # Create and launch app
    app = create_interface()
    
    app.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", 7860)),
        share=True,
        show_error=True,
    )


if __name__ == "__main__":
    main()
