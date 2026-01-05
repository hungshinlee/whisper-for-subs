"""
Silero VAD (Voice Activity Detection) module for audio segmentation.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")


class SileroVAD:
    """Silero VAD wrapper for detecting speech segments in audio."""

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
        sampling_rate: int = 16000,
    ):
        """
        Initialize Silero VAD.

        Args:
            threshold: Speech probability threshold (0-1)
            min_speech_duration_ms: Minimum speech segment duration in ms
            min_silence_duration_ms: Minimum silence duration to split segments
            speech_pad_ms: Padding to add around speech segments
            sampling_rate: Audio sampling rate (must be 8000 or 16000)
        """
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.sampling_rate = sampling_rate

        # Load Silero VAD model
        self.model, self.utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        
        (
            self.get_speech_timestamps,
            self.save_audio,
            self.read_audio,
            self.VADIterator,
            self.collect_chunks,
        ) = self.utils

    def detect_speech_segments(
        self, audio: np.ndarray, return_seconds: bool = True
    ) -> List[dict]:
        """
        Detect speech segments in audio.

        Args:
            audio: Audio array (mono, float32, normalized to [-1, 1])
            return_seconds: If True, return timestamps in seconds; otherwise samples

        Returns:
            List of dictionaries with 'start' and 'end' timestamps
        """
        # Convert to torch tensor if needed
        if isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio).float()
        else:
            audio_tensor = audio.float()

        # Ensure 1D (convert stereo to mono if needed)
        if audio_tensor.dim() > 1:
            if audio_tensor.shape[1] > 1:
                # Multiple channels - average them to mono
                audio_tensor = audio_tensor.mean(dim=1)
            else:
                # Single channel - just squeeze
                audio_tensor = audio_tensor.squeeze()

        # Get speech timestamps
        speech_timestamps = self.get_speech_timestamps(
            audio_tensor,
            self.model,
            threshold=self.threshold,
            sampling_rate=self.sampling_rate,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            speech_pad_ms=self.speech_pad_ms,
            return_seconds=return_seconds,
        )

        return speech_timestamps

    def merge_short_segments(
        self,
        segments: List[dict],
        min_duration: float = 1.0,
        max_duration: float = 30.0,
        max_gap: float = 0.5,
    ) -> List[dict]:
        """
        Merge short segments and split long ones for better transcription.

        Args:
            segments: List of speech segments with 'start' and 'end'
            min_duration: Minimum segment duration in seconds
            max_duration: Maximum segment duration in seconds
            max_gap: Maximum gap between segments to merge

        Returns:
            Merged and adjusted segments
        """
        if not segments:
            return []

        merged = []
        current = segments[0].copy()

        for next_seg in segments[1:]:
            gap = next_seg["start"] - current["end"]
            current_duration = current["end"] - current["start"]
            combined_duration = next_seg["end"] - current["start"]

            # Merge if gap is small and combined duration is acceptable
            if gap <= max_gap and combined_duration <= max_duration:
                current["end"] = next_seg["end"]
            else:
                # Save current segment if long enough
                if current_duration >= min_duration:
                    merged.append(current)
                current = next_seg.copy()

        # Don't forget the last segment
        if current["end"] - current["start"] >= min_duration:
            merged.append(current)

        return merged

    def segment_audio(
        self,
        audio: np.ndarray,
        merge: bool = True,
        min_duration: float = 1.0,
        max_duration: float = 30.0,
    ) -> List[Tuple[float, float, np.ndarray]]:
        """
        Segment audio based on VAD and return audio chunks.

        Args:
            audio: Audio array
            merge: Whether to merge short segments
            min_duration: Minimum segment duration
            max_duration: Maximum segment duration

        Returns:
            List of tuples (start_time, end_time, audio_chunk)
        """
        segments = self.detect_speech_segments(audio, return_seconds=True)

        if merge:
            segments = self.merge_short_segments(
                segments,
                min_duration=min_duration,
                max_duration=max_duration,
            )

        chunks = []
        for seg in segments:
            start_sample = int(seg["start"] * self.sampling_rate)
            end_sample = int(seg["end"] * self.sampling_rate)
            chunk = audio[start_sample:end_sample]
            chunks.append((seg["start"], seg["end"], chunk))

        return chunks


def load_vad(
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
) -> SileroVAD:
    """
    Convenience function to load VAD with common settings.
    """
    return SileroVAD(
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
    )
