"""
Multi-GPU parallel transcription for efficient processing of long audio files.
"""

import os
import time
import tempfile
from typing import List, Dict, Optional, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import soundfile as sf

from transcriber import WhisperTranscriber
from vad import SileroVAD


def transcribe_segment_on_gpu(
    args: tuple,
) -> Dict:
    """
    Transcribe a single audio segment on a specific GPU.
    
    This function runs in a separate process with its own GPU assignment.
    
    Args:
        args: Tuple of (segment_index, audio_segment, start_time, end_time, 
                       gpu_id, model_size, language, task)
    
    Returns:
        Dictionary with segment results
    """
    (
        segment_idx,
        audio_data,
        start_time,
        end_time,
        gpu_id,
        model_size,
        language,
        task,
        compute_type,
    ) = args
    
    # Set this process to use specific GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    try:
        # Create temporary file for this segment
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            sf.write(temp_path, audio_data, 16000)
        
        # Initialize transcriber for this GPU
        transcriber = WhisperTranscriber(
            model_size=model_size,
            device="cuda",
            compute_type=compute_type,
            use_vad=False,  # VAD already done
        )
        
        # Transcribe segment
        segments = transcriber.transcribe(
            temp_path,
            language=language,
            task=task,
            progress_callback=None,
        )
        
        # Adjust timestamps to global timeline
        adjusted_segments = []
        for seg in segments:
            adjusted_segments.append({
                "start": start_time + seg["start"],
                "end": start_time + seg["end"],
                "text": seg["text"],
            })
        
        return {
            "segment_idx": segment_idx,
            "success": True,
            "segments": adjusted_segments,
            "gpu_id": gpu_id,
            "duration": end_time - start_time,
        }
        
    except Exception as e:
        return {
            "segment_idx": segment_idx,
            "success": False,
            "error": str(e),
            "gpu_id": gpu_id,
        }
    
    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)


class ParallelWhisperTranscriber:
    """
    Multi-GPU parallel Whisper transcriber for efficient processing.
    """
    
    def __init__(
        self,
        model_size: str = "large-v3-turbo",
        compute_type: str = "float16",
        gpu_ids: List[int] = None,
        vad_threshold: float = 0.5,
    ):
        """
        Initialize parallel transcriber.
        
        Args:
            model_size: Whisper model size
            compute_type: Compute type (float16/int8/float32)
            gpu_ids: List of GPU IDs to use (e.g., [0, 1, 2, 3])
            vad_threshold: VAD detection threshold
        """
        self.model_size = model_size
        self.compute_type = compute_type
        self.gpu_ids = gpu_ids or [0, 1, 2, 3]
        self.num_gpus = len(self.gpu_ids)
        
        print(f"Initialized ParallelWhisperTranscriber with {self.num_gpus} GPUs: {self.gpu_ids}")
        
        # Initialize VAD (runs on CPU, shared across all processes)
        self.vad = SileroVAD(threshold=vad_threshold)
    
    def transcribe_parallel(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe",
        min_segment_duration: float = 10.0,  # Minimum 10 seconds per segment
        max_segment_duration: float = 60.0,  # Maximum 60 seconds per segment
        progress_callback: Optional[Callable] = None,
    ) -> List[Dict]:
        """
        Transcribe audio file using multiple GPUs in parallel.
        
        Args:
            audio_path: Path to audio file
            language: Source language code
            task: "transcribe" or "translate"
            min_segment_duration: Minimum duration for each segment
            max_segment_duration: Maximum duration for each segment
            progress_callback: Callback function(progress, message)
        
        Returns:
            List of transcription segments with timestamps
        """
        start_time = time.time()
        
        if progress_callback:
            progress_callback(0, "Loading audio file...")
        
        # Load audio
        audio, sample_rate = sf.read(audio_path, dtype="float32")
        total_duration = len(audio) / sample_rate
        
        if progress_callback:
            progress_callback(5, f"Audio loaded: {total_duration:.1f}s")
        
        # Use VAD to detect speech segments
        if progress_callback:
            progress_callback(10, "Detecting speech with VAD...")
        
        vad_segments = self.vad.detect_speech_segments(audio, return_seconds=True)
        
        if not vad_segments:
            if progress_callback:
                progress_callback(100, "No speech detected")
            return []
        
        # Merge VAD segments to optimal size for parallel processing
        if progress_callback:
            progress_callback(15, "Optimizing segments for parallel processing...")
        
        optimized_segments = self._optimize_segments(
            vad_segments,
            audio,
            sample_rate,
            min_duration=min_segment_duration,
            max_duration=max_segment_duration,
        )
        
        num_segments = len(optimized_segments)
        if progress_callback:
            progress_callback(
                20,
                f"Split into {num_segments} segments for {self.num_gpus} GPUs"
            )
        
        # Prepare tasks for parallel processing
        tasks = []
        for idx, (start, end, audio_segment) in enumerate(optimized_segments):
            gpu_id = self.gpu_ids[idx % self.num_gpus]  # Round-robin GPU assignment
            tasks.append((
                idx,
                audio_segment,
                start,
                end,
                gpu_id,
                self.model_size,
                language,
                task,
                self.compute_type,
            ))
        
        # Process segments in parallel using multiple GPUs
        if progress_callback:
            progress_callback(25, f"Starting parallel transcription on {self.num_gpus} GPUs...")
        
        results = []
        completed = 0
        
        with ProcessPoolExecutor(max_workers=self.num_gpus) as executor:
            # Submit all tasks
            future_to_segment = {
                executor.submit(transcribe_segment_on_gpu, task): task[0]
                for task in tasks
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_segment):
                result = future.result()
                results.append(result)
                completed += 1
                
                if progress_callback:
                    progress_pct = 25 + (completed / num_segments) * 70
                    gpu_info = f"GPU {result.get('gpu_id', '?')}"
                    progress_callback(
                        progress_pct,
                        f"Processing ({completed}/{num_segments}) on {gpu_info}..."
                    )
        
        # Sort results by segment index and merge
        if progress_callback:
            progress_callback(95, "Merging results...")
        
        results.sort(key=lambda x: x["segment_idx"])
        
        # Collect all segments
        all_segments = []
        failed_segments = []
        
        for result in results:
            if result["success"]:
                all_segments.extend(result["segments"])
            else:
                failed_segments.append(result["segment_idx"])
        
        if failed_segments:
            print(f"Warning: {len(failed_segments)} segments failed to transcribe")
        
        # Sort by timestamp
        all_segments.sort(key=lambda x: x["start"])
        
        elapsed = time.time() - start_time
        speed_ratio = total_duration / elapsed if elapsed > 0 else 0
        
        if progress_callback:
            progress_callback(
                100,
                f"Complete! {len(all_segments)} segments | "
                f"Speed: {speed_ratio:.1f}x realtime | "
                f"Time: {elapsed:.1f}s"
            )
        
        return all_segments
    
    def _optimize_segments(
        self,
        vad_segments: List[Dict],
        audio: np.ndarray,
        sample_rate: int,
        min_duration: float,
        max_duration: float,
    ) -> List[tuple]:
        """
        Optimize VAD segments for parallel processing.
        
        Merges short segments and splits long ones to balance workload across GPUs.
        
        Returns:
            List of tuples (start_time, end_time, audio_chunk)
        """
        # First, merge very short segments
        merged = []
        current = None
        
        for seg in vad_segments:
            duration = seg["end"] - seg["start"]
            
            if current is None:
                current = seg.copy()
            else:
                gap = seg["start"] - current["end"]
                combined_duration = seg["end"] - current["start"]
                
                # Merge if gap is small and combined duration is reasonable
                if gap < 1.0 and combined_duration < max_duration:
                    current["end"] = seg["end"]
                else:
                    merged.append(current)
                    current = seg.copy()
        
        if current:
            merged.append(current)
        
        # Now split any segments that are too long
        optimized = []
        
        for seg in merged:
            start = seg["start"]
            end = seg["end"]
            duration = end - start
            
            if duration <= max_duration:
                # Segment is good as-is
                start_sample = int(start * sample_rate)
                end_sample = int(end * sample_rate)
                chunk = audio[start_sample:end_sample]
                optimized.append((start, end, chunk))
            else:
                # Split long segment into smaller chunks
                num_chunks = int(np.ceil(duration / max_duration))
                chunk_duration = duration / num_chunks
                
                for i in range(num_chunks):
                    chunk_start = start + i * chunk_duration
                    chunk_end = min(start + (i + 1) * chunk_duration, end)
                    
                    start_sample = int(chunk_start * sample_rate)
                    end_sample = int(chunk_end * sample_rate)
                    chunk = audio[start_sample:end_sample]
                    
                    optimized.append((chunk_start, chunk_end, chunk))
        
        return optimized
    
    def get_stats(self) -> Dict:
        """Get statistics about GPU usage."""
        return {
            "num_gpus": self.num_gpus,
            "gpu_ids": self.gpu_ids,
            "model_size": self.model_size,
            "compute_type": self.compute_type,
        }


# Convenience function for easy use
def transcribe_with_multiple_gpus(
    audio_path: str,
    model_size: str = "large-v3-turbo",
    language: Optional[str] = None,
    task: str = "transcribe",
    gpu_ids: List[int] = None,
    compute_type: str = "float16",
    progress_callback: Optional[Callable] = None,
) -> List[Dict]:
    """
    Transcribe audio using multiple GPUs.
    
    This is a convenience function that handles the entire parallel transcription process.
    
    Args:
        audio_path: Path to audio file
        model_size: Whisper model size
        language: Source language (None for auto-detect)
        task: "transcribe" or "translate"
        gpu_ids: List of GPU IDs to use (default: [0, 1, 2, 3])
        compute_type: Compute type (float16/int8/float32)
        progress_callback: Optional progress callback
    
    Returns:
        List of transcription segments
    """
    transcriber = ParallelWhisperTranscriber(
        model_size=model_size,
        compute_type=compute_type,
        gpu_ids=gpu_ids,
    )
    
    return transcriber.transcribe_parallel(
        audio_path=audio_path,
        language=language,
        task=task,
        progress_callback=progress_callback,
    )
