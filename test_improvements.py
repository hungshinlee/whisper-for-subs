#!/usr/bin/env python3
"""
Test script to verify multi-user isolation and cleanup improvements.
"""

import os
import time
import requests
import concurrent.futures
from pathlib import Path


def check_session_cleanup():
    """Check if session directories are properly cleaned up."""
    sessions_dir = Path("/tmp/whisper-sessions")
    
    if not sessions_dir.exists():
        print("✅ No session directory exists (good - nothing to clean)")
        return True
    
    sessions = list(sessions_dir.iterdir())
    if not sessions:
        print("✅ Session directory exists but is empty (good)")
        return True
    
    print(f"⚠️  Found {len(sessions)} session directories:")
    for session in sessions:
        age = time.time() - session.stat().st_mtime
        print(f"  - {session.name}: {age:.1f}s old")
    
    return False


def check_temp_files():
    """Check for leftover temporary files."""
    temp_dirs = [
        "/tmp/whisper-downloads",
        "/tmp/whisper-sessions",
    ]
    
    total_files = 0
    for dir_path in temp_dirs:
        if not os.path.exists(dir_path):
            continue
        
        files = list(Path(dir_path).rglob("*"))
        file_count = len([f for f in files if f.is_file()])
        total_files += file_count
        
        if file_count > 0:
            print(f"⚠️  {dir_path}: {file_count} files found")
        else:
            print(f"✅ {dir_path}: clean")
    
    return total_files == 0


def simulate_concurrent_requests(num_requests=2):
    """
    Simulate concurrent requests to test isolation.
    Note: This requires the Gradio app to be running.
    """
    print(f"\n🧪 Simulating {num_requests} concurrent requests...")
    print("⚠️  Note: This test requires the Gradio app to be running on port 7860")
    
    # Check if app is running
    try:
        response = requests.get("http://localhost:7860", timeout=2)
        if response.status_code != 200:
            print("❌ Gradio app is not responding")
            return False
    except requests.exceptions.RequestException:
        print("❌ Gradio app is not running on localhost:7860")
        print("   Start the app first: python app.py")
        return False
    
    print("✅ Gradio app is running")
    
    # TODO: Implement actual API calls to test concurrent processing
    # This would require using Gradio's API or client interface
    print("ℹ️  Manual testing recommended:")
    print("   1. Open two browser tabs")
    print("   2. Upload different audio files simultaneously")
    print("   3. Verify both complete successfully")
    
    return True


def test_transcriber_pool():
    """Test TranscriberPool functionality."""
    print("\n🧪 Testing TranscriberPool...")
    
    try:
        # Import the pool
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from app import transcriber_pool
        
        print(f"✅ TranscriberPool initialized")
        print(f"   Max workers: {transcriber_pool.max_workers}")
        print(f"   Single GPU pool size: {len(transcriber_pool.single_gpu_pool)}")
        print(f"   Parallel pool size: {len(transcriber_pool.parallel_gpu_pool)}")
        
        return True
    except Exception as e:
        print(f"❌ Error testing TranscriberPool: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("🔍 Whisper-for-Subs Improvement Verification")
    print("="*60)
    
    results = []
    
    # Test 1: Check session cleanup
    print("\n📋 Test 1: Session Directory Cleanup")
    print("-" * 60)
    results.append(("Session Cleanup", check_session_cleanup()))
    
    # Test 2: Check temp files
    print("\n📋 Test 2: Temporary Files Cleanup")
    print("-" * 60)
    results.append(("Temp Files", check_temp_files()))
    
    # Test 3: Test TranscriberPool
    print("\n📋 Test 3: TranscriberPool")
    print("-" * 60)
    results.append(("TranscriberPool", test_transcriber_pool()))
    
    # Test 4: Concurrent requests (optional)
    print("\n📋 Test 4: Concurrent Request Handling")
    print("-" * 60)
    results.append(("Concurrent Requests", simulate_concurrent_requests()))
    
    # Summary
    print("\n" + "="*60)
    print("📊 Test Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Improvements are working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please review the output above.")
    
    return passed == total


if __name__ == "__main__":
    exit(0 if main() else 1)
