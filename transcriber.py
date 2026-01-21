"""
Whisper transcription module using faster-whisper for efficient inference.
"""

import os
import tempfile
import subprocess
from typing import List, Optional, Generator
import numpy as np
import torch

from faster_whisper import WhisperModel
from vad import SileroVAD


def ensure_model_ready(model_name: str) -> str:
    """
    Ensure the model is in CTranslate2 format.
    If it's a known non-CT2 model, convert it automatically.

    Args:
        model_name: Name of the model (e.g., "large-v3" or HF repo ID)

    Returns:
        Path to the usable model (CT2 format)
    """
    # Map of models that need conversion -> target directory name
    CUSTOM_MODELS = {
        "formospeech/whisper-large-v2-taiwanese-hakka-v1": "whisper-large-v2-taiwanese-hakka-v1-ct2"
    }

    if model_name not in CUSTOM_MODELS:
        return model_name

    # Get cache directory
    cache_dir = os.environ.get("HF_HOME", "/root/.cache/huggingface")
    models_dir = os.path.join(cache_dir, "ct2_converted")
    target_dir = os.path.join(models_dir, CUSTOM_MODELS[model_name])

    # Check if already converted
    if os.path.exists(os.path.join(target_dir, "model.bin")):
        print(f"✅ Found converted model at: {target_dir}")
        return target_dir

    print(f"⚠️  Model {model_name} needs conversion to CTranslate2 format.")
    print(f"   Converting to {target_dir}...")

    # Ensure directory exists
    os.makedirs(target_dir, exist_ok=True)

    try:
        # Run conversion using ct2-transformers-converter
        # We use float16 by default as it's the standard for GPU
        cmd = [
            "ct2-transformers-converter",
            "--model",
            model_name,
            "--output_dir",
            target_dir,
            "--quantization",
            "float16",
            "--force",
        ]

        print(f"   Running: {' '.join(cmd)}")
        subprocess.check_call(cmd)
        print("✅ Conversion complete!")
        return target_dir

    except subprocess.CalledProcessError as e:
        print(f"❌ Conversion failed with code {e.returncode}")
        # Cleanup
        if os.path.exists(target_dir):
            import shutil

            shutil.rmtree(target_dir)
        # Fallback to original name (will likely fail, but we tried)
        return model_name
    except Exception as e:
        print(f"❌ Conversion error: {str(e)}")
        return model_name


# Supported languages for Whisper
SUPPORTED_LANGUAGES = {"auto": "Auto", "zh": "Mandarin", "en": "English"}

# Model configurations with labels
MODEL_CONFIGS = {
    "large-v3": {
        "label": "General",
        "display_name": "[General] large-v3",
    },
    "large-v3-turbo": {
        "label": "General",
        "display_name": "[General] large-v3-turbo",
    },
    "formospeech/whisper-large-v2-taiwanese-hakka-v1": {
        "label": "Hakka",
        "display_name": "[Hakka] formospeech/whisper-large-v2-taiwanese-hakka-v1",
    },
}

# Model IDs list (for backward compatibility)
MODEL_SIZES = list(MODEL_CONFIGS.keys())


class WhisperTranscriber:
    """Whisper-based transcription with VAD support."""

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        use_vad: bool = True,
        vad_threshold: float = 0.5,
        min_silence_duration_ms: int = 100,
    ):
        """
        Initialize transcriber.

        Args:
            model_size: Whisper model size
            device: Device to use (cuda/cpu)
            compute_type: Compute type (float16/int8/float32)
            use_vad: Whether to use VAD for segmentation
            vad_threshold: VAD speech detection threshold
            min_silence_duration_ms: Minimum silence duration in ms to split segments
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.use_vad = use_vad

        # Auto-detect device
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            self.device = "cpu"
            self.compute_type = "float32"

        # Determine GPU index for logging
        self.gpu_index = None
        if self.device == "cuda" and torch.cuda.is_available():
            self.gpu_index = torch.cuda.current_device()
            print(f"🎯 Single-GPU mode: Using GPU {self.gpu_index}")

        # Load Whisper model
        print(f"Loading Whisper model: {model_size} on {self.device}")

        # Ensure model is ready (convert if necessary)
        actual_model_path = ensure_model_ready(model_size)
        if actual_model_path != model_size:
            print(f"   Using converted model path: {actual_model_path}")

        self.model = WhisperModel(
            actual_model_path,
            device=self.device,
            compute_type=self.compute_type,
        )
        print("✅ Model loaded successfully")

        # Load VAD if enabled
        self.vad = None
        if use_vad:
            print(
                f"Loading Silero VAD (min_silence_duration={min_silence_duration_ms}ms)..."
            )
            self.vad = SileroVAD(
                threshold=vad_threshold,
                min_silence_duration_ms=min_silence_duration_ms,
            )
            print("✅ VAD loaded successfully")

    def load_audio(self, file_path: str, sample_rate: int = 16000) -> np.ndarray:
        """
        Load audio file and convert to proper format.

        Args:
            file_path: Path to audio/video file
            sample_rate: Target sample rate

        Returns:
            Audio array (mono, float32)
        """
        # Create temp file for converted audio
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_wav.close()

        try:
            # Use ffmpeg to convert to WAV
            cmd = [
                "ffmpeg",
                "-i",
                file_path,
                "-ar",
                str(sample_rate),
                "-ac",
                "1",
                "-f",
                "wav",
                "-y",
                temp_wav.name,
            ]
            subprocess.run(cmd, capture_output=True, check=True)

            # Load with soundfile
            import soundfile as sf

            audio, sr = sf.read(temp_wav.name, dtype="float32")

            return audio

        finally:
            # Cleanup temp file
            if os.path.exists(temp_wav.name):
                os.unlink(temp_wav.name)

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe",
        initial_prompt: Optional[str] = None,
        word_timestamps: bool = False,
        progress_callback=None,
    ) -> List[dict]:
        """
        Transcribe audio file.

        Args:
            audio_path: Path to audio file
            language: Source language code (None for auto-detect)
            task: "transcribe" or "translate"
            initial_prompt: Initial prompt to guide transcription
            word_timestamps: Whether to include word-level timestamps
            progress_callback: Callback function(progress, status)

        Returns:
            List of segments with start, end, text
        """
        import time

        start_time = time.time()

        if progress_callback:
            progress_callback(0, "Loading audio...")

        # Load audio
        audio = self.load_audio(audio_path)
        duration = len(audio) / 16000  # seconds

        print(f"📊 Audio loaded: {duration:.1f}s ({len(audio)} samples @ 16000Hz)")

        if progress_callback:
            progress_callback(5, f"Audio duration: {duration:.1f} seconds")

        # Use VAD for segmentation if enabled
        if self.use_vad and self.vad is not None:
            if progress_callback:
                progress_callback(10, "Detecting speech segments with VAD...")
            segments = self._transcribe_with_vad(
                audio,
                duration,
                language,
                task,
                initial_prompt,
                word_timestamps,
                progress_callback,
            )
        else:
            segments = self._transcribe_direct(
                audio_path,
                language,
                task,
                initial_prompt,
                word_timestamps,
                progress_callback,
            )

        # Print summary
        elapsed = time.time() - start_time
        speed_ratio = duration / elapsed if elapsed > 0 else 0
        gpu_info = f"GPU {self.gpu_index}" if self.gpu_index is not None else "CPU"

        print("✅ Transcription complete!")
        print(f"   Device: {gpu_info}")
        print(f"   Segments: {len(segments)}")
        print(f"   Duration: {duration:.1f}s")
        print(f"   Time: {elapsed:.1f}s")
        print(f"   Speed: {speed_ratio:.1f}x realtime")

        return segments

    def _transcribe_with_vad(
        self,
        audio: np.ndarray,
        duration: float,
        language: Optional[str],
        task: str,
        initial_prompt: Optional[str],
        word_timestamps: bool,
        progress_callback=None,
    ) -> List[dict]:
        """Transcribe using VAD segmentation."""
        # Get speech segments
        chunks = self.vad.segment_audio(
            audio,
            merge=True,
            min_duration=0.5,
            max_duration=30.0,
        )

        if not chunks:
            print("⚠ No speech detected in audio")
            if progress_callback:
                progress_callback(100, "No speech detected")
            return []

        print(f"🎯 VAD detected {len(chunks)} speech segments")

        if progress_callback:
            progress_callback(15, f"Detected {len(chunks)} speech segments")

        segments = []
        gpu_label = f"GPU {self.gpu_index}" if self.gpu_index is not None else "CPU"

        for i, (start_time, end_time, chunk_audio) in enumerate(chunks):
            chunk_duration = end_time - start_time
            print(
                f"[{gpu_label}] ▶ Processing chunk {i + 1}/{len(chunks)} ({chunk_duration:.1f}s)"
            )

            # Update progress
            progress = 15 + (i / len(chunks)) * 80
            if progress_callback:
                progress_callback(progress, f"Transcribing ({i + 1}/{len(chunks)})...")

            # Save chunk to temp file for transcription
            temp_chunk = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_chunk.close()

            try:
                import soundfile as sf

                sf.write(temp_chunk.name, chunk_audio, 16000)

                # Transcribe chunk
                result, info = self.model.transcribe(
                    temp_chunk.name,
                    language=language if language != "auto" else None,
                    task=task,
                    initial_prompt=initial_prompt,
                    word_timestamps=word_timestamps,
                    vad_filter=False,  # We already did VAD
                )

                # Collect segments with adjusted timestamps
                chunk_segments = []
                for seg in result:
                    chunk_segments.append(
                        {
                            "start": start_time + seg.start,
                            "end": start_time + seg.end,
                            "text": seg.text,
                        }
                    )
                    segments.append(chunk_segments[-1])

                print(
                    f"[{gpu_label}] ✓ Chunk {i + 1} complete: {len(chunk_segments)} text segments"
                )

            finally:
                if os.path.exists(temp_chunk.name):
                    os.unlink(temp_chunk.name)

        if progress_callback:
            progress_callback(100, f"Complete! {len(segments)} segments")

        return segments

    def _transcribe_direct(
        self,
        audio_path: str,
        language: Optional[str],
        task: str,
        initial_prompt: Optional[str],
        word_timestamps: bool,
        progress_callback=None,
    ) -> List[dict]:
        """Transcribe without VAD (use Whisper's built-in VAD)."""
        if progress_callback:
            progress_callback(20, "Starting transcription...")

        result, info = self.model.transcribe(
            audio_path,
            language=language if language != "auto" else None,
            task=task,
            initial_prompt=initial_prompt,
            word_timestamps=word_timestamps,
            vad_filter=True,
        )

        segments = []
        for seg in result:
            segments.append(
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                }
            )

            if progress_callback:
                # Estimate progress based on timestamp
                if info.duration > 0:
                    progress = 20 + (seg.end / info.duration) * 75
                    progress_callback(progress, f"Transcribing... {seg.end:.1f}s")

        if progress_callback:
            progress_callback(100, f"Complete! {len(segments)} segments")

        return segments

    def transcribe_streaming(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe",
    ) -> Generator[dict, None, None]:
        """
        Transcribe audio with streaming output.

        Yields:
            Segment dictionaries as they are transcribed
        """
        result, info = self.model.transcribe(
            audio_path,
            language=language if language != "auto" else None,
            task=task,
            vad_filter=True,
        )

        for seg in result:
            yield {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
            }


def get_available_devices() -> List[str]:
    """Get list of available compute devices."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devices.append(f"cuda:{i}")
        devices.insert(1, "cuda")  # Default CUDA device
    return devices


def get_gpu_info() -> List[dict]:
    """Get information about available GPUs."""
    info = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info.append(
                {
                    "index": i,
                    "name": props.name,
                    "memory_total": props.total_memory / (1024**3),  # GB
                    "memory_free": torch.cuda.memory_reserved(i) / (1024**3),
                }
            )
    return info
