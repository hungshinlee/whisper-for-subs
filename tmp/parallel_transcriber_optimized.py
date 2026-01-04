"""
Multi-GPU parallel transcription for efficient processing of long audio files.
OPTIMIZED VERSION with persistent worker processes and model caching.
"""

import os
import time
import tempfile
import multiprocessing
from typing import List, Dict, Optional, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import soundfile as sf

from transcriber import WhisperTranscriber
from vad import SileroVAD

# CRITICAL: Set multiprocessing start method to 'spawn' for CUDA compatibility
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# Global variable to store transcriber instance in each worker process
_worker_transcriber = None
_worker_gpu_id = None


def _init_worker(gpu_id: int, model_size: str, compute_type: str):
    """
    Initialize worker process with a persistent transcriber instance.
    This function is called once when each worker process starts.
    
    Args:
        gpu_id: GPU ID for this worker
        model_size: Whisper model size
        compute_type: Compute type (float16/int8/float32)
    """
    global _worker_transcriber, _worker_gpu_id
    
    # Set GPU for this worker
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    _worker_gpu_id = gpu_id
    
    # Initialize transcriber once per worker
    print(f"[GPU {gpu_id}] 🔧 Initializing worker with model {model_size}...")
    _worker_transcriber = WhisperTranscriber(
        model_size=model_size,
        device="cuda",
        compute_type=compute_type,
        use_vad=False,  # VAD already done
    )
    print(f"[GPU {gpu_id}] ✅ Worker initialized and ready")


def transcribe_segment_on_gpu(args: tuple) -> Dict:
    """
    Transcribe a single audio segment using the pre-loaded model.
    
    This function reuses the transcriber instance created in _init_worker.
    
    Args:
        args: Tuple of (segment_index, audio_segment, start_time, end_time, language, task)
    
    Returns:
        Dictionary with segment results
    """
    global _worker_transcriber, _worker_gpu_id
    
    (
        segment_idx,
        audio_data,
        start_time,
        end_time,
        language,
        task,
    ) = args
    
    temp_path = None
    gpu_id = _worker_gpu_id
    
    try:
        # Validate audio data
        if len(audio_data) == 0:
            raise ValueError(f"Segment {segment_idx}: Empty audio data")
        
        duration = end_time - start_time
        
        # Skip very short segments (< 100ms)
        if duration < 0.1:
            print(f"[GPU {gpu_id}] ⊘ Segment {segment_idx} too short ({duration:.2f}s), skipping")
            return {
                "segment_idx": segment_idx,
                "success": True,
                "segments": [],
                "gpu_id": gpu_id,
                "duration": duration,
                "skipped": True,
            }
        
        # Create temporary file for this segment
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            sf.write(temp_path, audio_data, 16000)
        
        print(f"[GPU {gpu_id}] ▶ Processing segment {segment_idx} ({duration:.1f}s)")
        
        # Use the pre-loaded transcriber (no model loading overhead!)
        segments = _worker_transcriber.transcribe(
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
        
        print(f"[GPU {gpu_id}] ✓ Segment {segment_idx} complete: {len(adjusted_segments)} text segments")
        
        return {
            "segment_idx": segment_idx,
            "success": True,
            "segments": adjusted_segments,
            "gpu_id": gpu_id,
            "duration": duration,
        }
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"[GPU {gpu_id}] ✗ ERROR in segment {segment_idx}: {str(e)}")
        print(f"[GPU {gpu_id}] Traceback:\n{error_detail}")
        
        return {
            "segment_idx": segment_idx,
            "success": False,
            "error": str(e),
            "error_detail": error_detail,
            "gpu_id": gpu_id,
        }
    
    finally:
        # Cleanup temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                print(f"[GPU {gpu_id}] ⚠ Could not delete temp file {temp_path}: {e}")


class ParallelWhisperTranscriber:
    """
    Multi-GPU parallel Whisper transcriber with persistent worker processes.
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
        print(f"Using multiprocessing start method: {multiprocessing.get_start_method()}")
        print(f"💡 Using persistent workers (models loaded once per GPU)")
        
        # Initialize VAD (runs on CPU, shared across all processes)
        self.vad = SileroVAD(threshold=vad_threshold)
    
    def transcribe_parallel(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe",
        min_segment_duration: float = 15.0,
        max_segment_duration: float = 45.0,
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
        
        print(f"📊 Audio loaded: {total_duration:.1f}s ({len(audio)} samples @ {sample_rate}Hz)")
        
        if progress_callback:
            progress_callback(5, f"Audio loaded: {total_duration:.1f}s")
        
        # Use VAD to detect speech segments
        if progress_callback:
            progress_callback(10, "Detecting speech with VAD...")
        
        vad_segments = self.vad.detect_speech_segments(audio, return_seconds=True)
        
        if not vad_segments:
            print("⚠ No speech detected in audio")
            if progress_callback:
                progress_callback(100, "No speech detected")
            return []
        
        print(f"🎯 VAD detected {len(vad_segments)} speech segments")
        
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
        print(f"✂️  Optimized to {num_segments} segments for {self.num_gpus} GPUs")
        
        if progress_callback:
            progress_callback(
                20,
                f"Split into {num_segments} segments for {self.num_gpus} GPUs"
            )
        
        # Prepare tasks for parallel processing
        # Note: Simplified args since GPU assignment is handled by worker initialization
        tasks = []
        for idx, (start, end, audio_segment) in enumerate(optimized_segments):
            tasks.append((
                idx,
                audio_segment,
                start,
                end,
                language,
                task,
            ))
        
        # Process segments in parallel using persistent workers
        print(f"🚀 Starting parallel transcription with {self.num_gpus} persistent workers...")
        
        if progress_callback:
            progress_callback(25, f"Initializing {self.num_gpus} GPU workers...")
        
        results = []
        completed = 0
        skipped = 0
        failed = 0
        
        # Create executor with worker initialization
        mp_context = multiprocessing.get_context('spawn')
        
        # Create init args for each worker (one worker per GPU)
        # Worker i will use GPU self.gpu_ids[i]
        with ProcessPoolExecutor(
            max_workers=self.num_gpus,
            mp_context=mp_context,
            initializer=_init_worker,
            initargs=(self.gpu_ids[0], self.model_size, self.compute_type),  # Will be overridden
        ) as executor:
            
            # We need to manually create workers with different GPU IDs
            # This is a workaround for ProcessPoolExecutor's limitation
            # Instead, we'll use a different approach: create separate executors
            pass
        
        # Better approach: Create separate executor for each GPU
        executors = []
        for gpu_id in self.gpu_ids:
            executor = ProcessPoolExecutor(
                max_workers=1,
                mp_context=mp_context,
                initializer=_init_worker,
                initargs=(gpu_id, self.model_size, self.compute_type),
            )
            executors.append(executor)
        
        try:
            # Submit tasks round-robin to each executor
            future_to_segment = {}
            for idx, task in enumerate(tasks):
                executor_idx = idx % self.num_gpus
                future = executors[executor_idx].submit(transcribe_segment_on_gpu, task)
                future_to_segment[future] = task[0]  # segment_idx
            
            # Collect results as they complete
            for future in as_completed(future_to_segment):
                result = future.result()
                results.append(result)
                completed += 1
                
                if not result["success"]:
                    failed += 1
                elif result.get("skipped"):
                    skipped += 1
                
                if progress_callback:
                    progress_pct = 25 + (completed / num_segments) * 70
                    gpu_info = f"GPU {result.get('gpu_id', '?')}"
                    status = "✓" if result["success"] else "✗"
                    progress_callback(
                        progress_pct,
                        f"{status} ({completed}/{num_segments}) on {gpu_info}..."
                    )
        
        finally:
            # Shutdown all executors
            for executor in executors:
                executor.shutdown(wait=True)
        
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
                failed_segments.append((result["segment_idx"], result.get("error", "Unknown")))
        
        # Report statistics
        if failed_segments:
            print(f"⚠️  {len(failed_segments)} segments failed to transcribe:")
            for idx, error in failed_segments[:5]:
                print(f"   - Segment {idx}: {error}")
            if len(failed_segments) > 5:
                print(f"   ... and {len(failed_segments) - 5} more")
        
        if skipped > 0:
            print(f"⊘ {skipped} segments skipped (too short)")
        
        # Sort by timestamp
        all_segments.sort(key=lambda x: x["start"])
        
        elapsed = time.time() - start_time
        speed_ratio = total_duration / elapsed if elapsed > 0 else 0
        
        print(f"✅ Complete! {len(all_segments)} text segments | "
              f"Speed: {speed_ratio:.1f}x realtime | "
              f"Time: {elapsed:.1f}s")
        
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
            
            # Skip segments that are too short
            if duration < 0.5:  # Less than 500ms
                continue
            
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
