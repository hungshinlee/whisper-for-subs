"""
SRT (SubRip Subtitle) format utilities.
"""

from typing import List, Optional
from dataclasses import dataclass
import re


@dataclass
class Subtitle:
    """Represents a single subtitle entry."""
    index: int
    start: float  # Start time in seconds
    end: float    # End time in seconds
    text: str


def format_timestamp(seconds: float) -> str:
    """
    Convert seconds to SRT timestamp format (HH:MM:SS,mmm).
    
    Args:
        seconds: Time in seconds
        
    Returns:
        SRT formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def parse_timestamp(timestamp: str) -> float:
    """
    Parse SRT timestamp to seconds.
    
    Args:
        timestamp: SRT timestamp string (HH:MM:SS,mmm)
        
    Returns:
        Time in seconds
    """
    match = re.match(r"(\d+):(\d+):(\d+)[,.](\d+)", timestamp)
    if not match:
        raise ValueError(f"Invalid timestamp format: {timestamp}")
    
    hours, minutes, seconds, millis = map(int, match.groups())
    return hours * 3600 + minutes * 60 + seconds + millis / 1000


def segments_to_srt(segments: List[dict]) -> str:
    """
    Convert Whisper segments to SRT format string.
    
    Args:
        segments: List of segment dictionaries with 'start', 'end', 'text' keys
        
    Returns:
        SRT formatted string
    """
    srt_lines = []
    
    for i, segment in enumerate(segments, 1):
        start = format_timestamp(segment["start"])
        end = format_timestamp(segment["end"])
        text = segment["text"].strip()
        
        srt_lines.append(f"{i}")
        srt_lines.append(f"{start} --> {end}")
        srt_lines.append(text)
        srt_lines.append("")  # Empty line between entries
    
    return "\n".join(srt_lines)


def parse_srt(srt_content: str) -> List[Subtitle]:
    """
    Parse SRT content into list of Subtitle objects.
    
    Args:
        srt_content: SRT file content as string
        
    Returns:
        List of Subtitle objects
    """
    subtitles = []
    
    # Split by double newlines (subtitle blocks)
    blocks = re.split(r"\n\n+", srt_content.strip())
    
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) >= 3:
            try:
                index = int(lines[0])
                timestamps = lines[1]
                text = "\n".join(lines[2:])
                
                # Parse timestamps
                match = re.match(
                    r"(.+?)\s*-->\s*(.+)",
                    timestamps
                )
                if match:
                    start = parse_timestamp(match.group(1).strip())
                    end = parse_timestamp(match.group(2).strip())
                    
                    subtitles.append(Subtitle(
                        index=index,
                        start=start,
                        end=end,
                        text=text
                    ))
            except (ValueError, IndexError):
                continue
    
    return subtitles


def merge_segments(
    segments: List[dict],
    max_chars: int = 80,
    max_duration: float = 5.0,
) -> List[dict]:
    """
    Merge short segments for better readability.
    
    Args:
        segments: List of segments with 'start', 'end', 'text'
        max_chars: Maximum characters per subtitle
        max_duration: Maximum duration per subtitle
        
    Returns:
        Merged segments
    """
    if not segments:
        return []
    
    merged = []
    current = {
        "start": segments[0]["start"],
        "end": segments[0]["end"],
        "text": segments[0]["text"].strip(),
    }
    
    for seg in segments[1:]:
        text = seg["text"].strip()
        combined_text = current["text"] + " " + text
        combined_duration = seg["end"] - current["start"]
        
        # Merge if within limits
        if len(combined_text) <= max_chars and combined_duration <= max_duration:
            current["end"] = seg["end"]
            current["text"] = combined_text
        else:
            merged.append(current)
            current = {
                "start": seg["start"],
                "end": seg["end"],
                "text": text,
            }
    
    merged.append(current)
    return merged


def adjust_timestamps(
    segments: List[dict],
    offset: float = 0.0,
) -> List[dict]:
    """
    Adjust segment timestamps by adding an offset.
    
    Args:
        segments: List of segments
        offset: Time offset in seconds to add
        
    Returns:
        Adjusted segments
    """
    return [
        {
            **seg,
            "start": seg["start"] + offset,
            "end": seg["end"] + offset,
        }
        for seg in segments
    ]
