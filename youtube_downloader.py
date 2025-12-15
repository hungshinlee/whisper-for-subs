"""
YouTube audio downloader using yt-dlp.
"""

import os
import re
import tempfile
from typing import Optional, Tuple
import yt_dlp


def is_youtube_url(url: str) -> bool:
    """
    Check if URL is a valid YouTube URL.
    
    Args:
        url: URL string to check
        
    Returns:
        True if valid YouTube URL
    """
    youtube_patterns = [
        r"(https?://)?(www\.)?youtube\.com/watch\?v=[\w-]+",
        r"(https?://)?(www\.)?youtu\.be/[\w-]+",
        r"(https?://)?(www\.)?youtube\.com/shorts/[\w-]+",
        r"(https?://)?(www\.)?youtube\.com/embed/[\w-]+",
    ]
    
    for pattern in youtube_patterns:
        if re.match(pattern, url):
            return True
    return False


def extract_video_id(url: str) -> Optional[str]:
    """
    Extract video ID from YouTube URL.
    
    Args:
        url: YouTube URL
        
    Returns:
        Video ID or None
    """
    patterns = [
        r"(?:v=|/)([a-zA-Z0-9_-]{11})(?:[&?/]|$)",
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
        r"embed/([a-zA-Z0-9_-]{11})",
        r"shorts/([a-zA-Z0-9_-]{11})",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def get_video_info(url: str) -> Optional[dict]:
    """
    Get video metadata without downloading.
    
    Args:
        url: YouTube URL
        
    Returns:
        Video info dictionary or None
    """
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return {
                "id": info.get("id"),
                "title": info.get("title"),
                "duration": info.get("duration"),
                "channel": info.get("channel"),
                "upload_date": info.get("upload_date"),
                "description": info.get("description", "")[:500],
            }
    except Exception as e:
        print(f"Error getting video info: {e}")
        return None


def download_audio(
    url: str,
    output_dir: Optional[str] = None,
    audio_format: str = "wav",
    sample_rate: int = 16000,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Download audio from YouTube video.
    
    Args:
        url: YouTube URL
        output_dir: Output directory (uses temp dir if None)
        audio_format: Output audio format (wav, mp3, etc.)
        sample_rate: Audio sample rate
        
    Returns:
        Tuple of (audio_file_path, video_title) or (None, None) on error
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure yt-dlp options
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": audio_format,
                "preferredquality": "192",
            }
        ],
        "postprocessor_args": [
            "-ar", str(sample_rate),
            "-ac", "1",  # Mono
        ],
        "quiet": True,
        "no_warnings": True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_id = info.get("id")
            video_title = info.get("title", "Unknown")
            
            # Find the downloaded file
            audio_path = os.path.join(output_dir, f"{video_id}.{audio_format}")
            
            if os.path.exists(audio_path):
                return audio_path, video_title
            
            # Try to find any audio file with the video ID
            for f in os.listdir(output_dir):
                if f.startswith(video_id):
                    return os.path.join(output_dir, f), video_title
            
            return None, None
            
    except Exception as e:
        print(f"Error downloading audio: {e}")
        return None, None


def download_audio_with_progress(
    url: str,
    output_dir: Optional[str] = None,
    progress_callback=None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Download audio with progress callback.
    
    Args:
        url: YouTube URL
        output_dir: Output directory
        progress_callback: Callback function(progress_percent, status_message)
        
    Returns:
        Tuple of (audio_file_path, video_title)
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp()
    
    os.makedirs(output_dir, exist_ok=True)
    
    def progress_hook(d):
        if progress_callback is None:
            return
            
        if d["status"] == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate", 0)
            downloaded = d.get("downloaded_bytes", 0)
            if total > 0:
                percent = (downloaded / total) * 100
                progress_callback(percent, f"下載中... {percent:.1f}%")
        elif d["status"] == "finished":
            progress_callback(100, "下載完成，正在轉換格式...")
    
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "postprocessor_args": [
            "-ar", "16000",
            "-ac", "1",
        ],
        "progress_hooks": [progress_hook],
        "quiet": True,
        "no_warnings": True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_id = info.get("id")
            video_title = info.get("title", "Unknown")
            
            audio_path = os.path.join(output_dir, f"{video_id}.wav")
            
            if os.path.exists(audio_path):
                return audio_path, video_title
            
            for f in os.listdir(output_dir):
                if f.startswith(video_id):
                    return os.path.join(output_dir, f), video_title
            
            return None, None
            
    except Exception as e:
        if progress_callback:
            progress_callback(0, f"錯誤: {str(e)}")
        return None, None
