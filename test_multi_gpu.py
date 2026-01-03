"""
Test script for multi-GPU parallel transcription.

This script demonstrates the performance improvement of using multiple GPUs.
"""

import time
import sys
from parallel_transcriber import transcribe_with_multiple_gpus


def test_single_vs_multi_gpu(audio_file: str):
    """
    Compare single-GPU vs multi-GPU transcription performance.
    
    Args:
        audio_file: Path to audio file to test
    """
    print("=" * 80)
    print("Multi-GPU Performance Test")
    print("=" * 80)
    
    # Test with single GPU
    print("\n[Test 1] Single GPU (GPU 0 only)")
    print("-" * 40)
    start = time.time()
    
    segments_single = transcribe_with_multiple_gpus(
        audio_path=audio_file,
        model_size="large-v3-turbo",
        gpu_ids=[0],  # Only use GPU 0
    )
    
    single_gpu_time = time.time() - start
    print(f"✓ Completed in {single_gpu_time:.2f} seconds")
    print(f"✓ Generated {len(segments_single)} segments")
    
    # Test with multiple GPUs
    print("\n[Test 2] Multiple GPUs (0, 1, 2, 3)")
    print("-" * 40)
    start = time.time()
    
    segments_multi = transcribe_with_multiple_gpus(
        audio_path=audio_file,
        model_size="large-v3-turbo",
        gpu_ids=[0, 1, 2, 3],  # Use all 4 GPUs
    )
    
    multi_gpu_time = time.time() - start
    print(f"✓ Completed in {multi_gpu_time:.2f} seconds")
    print(f"✓ Generated {len(segments_multi)} segments")
    
    # Compare results
    print("\n" + "=" * 80)
    print("Results Comparison")
    print("=" * 80)
    print(f"Single GPU time:  {single_gpu_time:.2f}s")
    print(f"Multi GPU time:   {multi_gpu_time:.2f}s")
    
    speedup = single_gpu_time / multi_gpu_time if multi_gpu_time > 0 else 0
    print(f"Speedup:          {speedup:.2f}x")
    
    efficiency = (speedup / 4) * 100  # Theoretical max is 4x with 4 GPUs
    print(f"Parallel efficiency: {efficiency:.1f}%")
    
    print("\n" + "=" * 80)


def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: python test_multi_gpu.py <audio_file>")
        print("\nExample:")
        print("  python test_multi_gpu.py /path/to/long_audio.wav")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    try:
        test_single_vs_multi_gpu(audio_file)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
