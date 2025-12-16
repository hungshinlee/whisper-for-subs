"""
Whisper transcription module using faster-whisper for efficient inference.
"""

import os
import tempfile
import subprocess
from typing import List, Optional, Tuple, Generator
import numpy as np
import torch

from faster_whisper import WhisperModel
from vad import SileroVAD


# Supported languages for Whisper
SUPPORTED_LANGUAGES = {
    "auto": "Auto Detect",
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
    "hi": "Hindi",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "ms": "Malay",
    "tl": "Filipino",
    "nan": "Taiwanese (Hokkien)",
    "yue": "Cantonese",
}

# Model sizes
MODEL_SIZES = [
    "large-v2",
    "large-v3",
    "large-v3-turbo",
]


class WhisperTranscriber:
    """Whisper-based transcription with VAD support."""

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        use_vad: bool = True,
        vad_threshold: float = 0.5,
    ):
        """
        Initialize transcriber.

        Args:
            model_size: Whisper model size
            device: Device to use (cuda/cpu)
            compute_type: Compute type (float16/int8/float32)
            use_vad: Whether to use VAD for segmentation
            vad_threshold: VAD speech detection threshold
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

        # Load Whisper model
        print(f"Loading Whisper model: {model_size} on {self.device}")
        self.model = WhisperModel(
            model_size,
            device=self.device,
            compute_type=self.compute_type,
        )

        # Load VAD if enabled
        self.vad = None
        if use_vad:
            print("Loading Silero VAD...")
            self.vad = SileroVAD(threshold=vad_threshold)

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
                "-i", file_path,
                "-ar", str(sample_rate),
                "-ac", "1",
                "-f", "wav",
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
        if progress_callback:
            progress_callback(0, "Loading audio...")

        # Load audio
        audio = self.load_audio(audio_path)
        duration = len(audio) / 16000  # seconds

        if progress_callback:
            progress_callback(5, f"Audio duration: {duration:.1f} seconds")

        # Use VAD for segmentation if enabled
        if self.use_vad and self.vad is not None:
            if progress_callback:
                progress_callback(10, "Detecting speech segments with VAD...")
            return self._transcribe_with_vad(
                audio,
                duration,
                language,
                task,
                initial_prompt,
                word_timestamps,
                progress_callback,
            )
        else:
            return self._transcribe_direct(
                audio_path,
                language,
                task,
                initial_prompt,
                word_timestamps,
                progress_callback,
            )

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
            if progress_callback:
                progress_callback(100, "No speech detected")
            return []

        if progress_callback:
            progress_callback(15, f"Detected {len(chunks)} speech segments")

        segments = []
        
        for i, (start_time, end_time, chunk_audio) in enumerate(chunks):
            # Update progress
            progress = 15 + (i / len(chunks)) * 80
            if progress_callback:
                progress_callback(
                    progress,
                    f"Transcribing ({i+1}/{len(chunks)})..."
                )

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
                for seg in result:
                    segments.append({
                        "start": start_time + seg.start,
                        "end": start_time + seg.end,
                        "text": seg.text,
                    })

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
            segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
            })
            
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
            info.append({
                "index": i,
                "name": props.name,
                "memory_total": props.total_memory / (1024**3),  # GB
                "memory_free": torch.cuda.memory_reserved(i) / (1024**3),
            })
    return info
