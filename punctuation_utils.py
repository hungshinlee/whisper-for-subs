"""
Punctuation post-processing for ASR segments.

Adds Chinese punctuation to Whisper output:
  - 。 at the end of each segment (sentence terminator)
  - ，at natural pause positions within a segment

TWO MODES (selected automatically):

  1. Gap-based  (preferred)
     Uses word-level timing from faster-whisper (word_timestamps=True).
     A ，is inserted between two adjacent words when the silence gap
     between them exceeds a threshold (default 300 ms).  This places
     commas exactly where the speaker paused — far more accurate than
     counting characters.

  2. Char-count fallback
     Used when a segment has no word-level data (e.g. streaming mode,
     or very short segments where Whisper didn't align words).
     Inserts ，approximately every N CJK characters.
"""

from typing import List, Dict, Optional

# Characters that already end a sentence — skip adding 。 after these.
_SENTENCE_ENDINGS = set("。！？…⋯♪～")

# Characters that already mark a mid-phrase break — never insert ，here.
_EXISTING_BREAKS = set("，、；：。！？…⋯")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _already_ends_sentence(text: str) -> bool:
    stripped = text.rstrip()
    return bool(stripped) and stripped[-1] in _SENTENCE_ENDINGS


def _insert_commas_by_gap(
    words: List[Dict],
    gap_threshold_s: float,
) -> str:
    """
    Reconstruct segment text and insert ，wherever the inter-word silence
    gap exceeds *gap_threshold_s* seconds.

    Args:
        words:            List of word dicts: {word, start, end, probability}.
        gap_threshold_s:  Minimum gap (seconds) to trigger a comma.

    Returns:
        Reconstructed text with commas inserted at natural pause points.
    """
    if not words:
        return ""

    parts = []
    for i, w in enumerate(words):
        word_text = w["word"]

        # Remove leading/trailing whitespace that faster-whisper often adds
        # (it prefixes Chinese tokens with a space).
        word_text = word_text.strip()

        if not word_text:
            continue

        parts.append(word_text)

        # Check gap to the next word
        if i < len(words) - 1:
            next_w = words[i + 1]
            gap = next_w["start"] - w["end"]

            if gap >= gap_threshold_s:
                # Only insert if neither the current nor next word already
                # starts / ends with a break character.
                last_char = word_text[-1] if word_text else ""
                next_text = words[i + 1]["word"].strip()
                first_next = next_text[0] if next_text else ""

                if last_char not in _EXISTING_BREAKS and first_next not in _EXISTING_BREAKS:
                    parts.append("，")

    return "".join(parts)


def _insert_commas_by_charcount(text: str, comma_every: int) -> str:
    """
    Insert ，roughly every *comma_every* CJK characters (fallback mode).

    ASCII words, whitespace, and existing punctuation reset the counter.
    """
    if len(text) <= comma_every:
        return text

    result = []
    char_count = 0
    i = 0

    while i < len(text):
        ch = text[i]

        if ch in _EXISTING_BREAKS:
            result.append(ch)
            char_count = 0
            i += 1
            continue

        if ch.isascii() and (ch.isalpha() or ch.isdigit()):
            result.append(ch)
            i += 1
            continue

        if ch.isspace():
            result.append(ch)
            char_count = 0
            i += 1
            continue

        # CJK or other non-ASCII character
        result.append(ch)
        char_count += 1
        i += 1

        if char_count >= comma_every:
            next_ch = text[i] if i < len(text) else ""
            if next_ch and next_ch not in _EXISTING_BREAKS and not next_ch.isspace():
                result.append("，")
            char_count = 0

    return "".join(result)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_punctuation(
    text: str,
    words: Optional[List[Dict]] = None,
    add_period: bool = True,
    add_comma: bool = True,
    gap_threshold_s: float = 0.3,
    comma_every: int = 15,
) -> str:
    """
    Add Chinese punctuation to a single segment.

    Args:
        text:            Raw ASR text for the segment.
        words:           Word-level timing list from faster-whisper
                         (each dict: {word, start, end, probability}).
                         If provided and non-empty, gap-based comma insertion
                         is used.  Otherwise falls back to char-count mode.
        add_period:      Append 。 if text doesn't already end with a
                         sentence terminator.
        add_comma:       Insert ，at natural pause positions.
        gap_threshold_s: Silence gap (seconds) that triggers a comma
                         (gap-based mode).  Default 0.3 s (300 ms).
        comma_every:     Char-count fallback threshold (CJK characters).

    Returns:
        Punctuated text string.
    """
    text = text.strip()
    if not text:
        return text

    if add_comma:
        if words:
            # Gap-based: reconstruct text from word list with comma injection.
            # Prefer this over the raw `text` because it avoids any whitespace
            # artefacts that faster-whisper sometimes leaves in the segment text.
            text = _insert_commas_by_gap(words, gap_threshold_s)
            if not text:
                # Reconstruction produced nothing — fall back to original text
                text = text or ""
        else:
            # Fallback: character-count heuristic
            text = _insert_commas_by_charcount(text, comma_every)

    if add_period and not _already_ends_sentence(text):
        text = text + "。"

    return text


def add_punctuation_to_segments(
    segments: List[Dict],
    add_period: bool = True,
    add_comma: bool = True,
    gap_threshold_s: float = 0.3,
    comma_every: int = 15,
) -> List[Dict]:
    """
    Apply punctuation to every segment in a list.

    Segments that contain a ``"words"`` key with word-level timing use
    gap-based comma insertion.  Segments without ``"words"`` fall back to
    the character-count heuristic.

    Args:
        segments:        List of segment dicts (must have ``"text"`` key).
        add_period:      See :func:`add_punctuation`.
        add_comma:       See :func:`add_punctuation`.
        gap_threshold_s: Silence gap threshold for comma insertion (seconds).
        comma_every:     Char-count fallback threshold.

    Returns:
        New list of segment dicts with punctuated text.
    """
    result = []
    gap_count = 0
    fallback_count = 0

    for seg in segments:
        new_seg = seg.copy()
        if "text" in new_seg:
            words = new_seg.get("words") or []
            if words:
                gap_count += 1
            else:
                fallback_count += 1

            new_seg["text"] = add_punctuation(
                new_seg["text"],
                words=words,
                add_period=add_period,
                add_comma=add_comma,
                gap_threshold_s=gap_threshold_s,
                comma_every=comma_every,
            )
        result.append(new_seg)

    if gap_count or fallback_count:
        print(
            f"✅ Punctuation added: {gap_count} gap-based, "
            f"{fallback_count} char-count fallback"
        )

    return result


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Simulate what faster-whisper returns for a 10-second segment
    # with a 400 ms pause after the 4th character.
    mock_words = [
        {"word": "今天", "start": 0.0,  "end": 0.5,  "probability": 0.99},
        {"word": "天氣", "start": 0.5,  "end": 1.0,  "probability": 0.98},
        # 400 ms gap here — should insert ，
        {"word": "很好", "start": 1.4,  "end": 1.9,  "probability": 0.97},
        {"word": "我們", "start": 1.9,  "end": 2.4,  "probability": 0.96},
        # 50 ms gap — too short, no comma
        {"word": "去公園", "start": 2.45, "end": 3.1, "probability": 0.95},
        # 600 ms gap — should insert ，
        {"word": "散步", "start": 3.7,  "end": 4.2,  "probability": 0.94},
    ]
    mock_text = "今天天氣很好我們去公園散步"

    print("=" * 60)
    print("punctuation_utils self-test")
    print("=" * 60)

    print("\n[Gap-based mode]")
    out = add_punctuation(mock_text, words=mock_words, gap_threshold_s=0.3)
    print(f"  IN : {mock_text!r}")
    print(f"  OUT: {out!r}")

    print("\n[Char-count fallback mode (no words)]")
    tests = [
        "今天天氣很好我們去公園散步然後吃冰淇淋",
        "這是一段短文",
        "已經有標點了，不需要再加。",
        "Hello this is English text",
        "超過門檻的長句子應該要在適當的地方加入逗號讓整體閱讀起來更加流暢自然",
    ]
    for t in tests:
        out = add_punctuation(t, words=None, comma_every=15)
        print(f"  IN : {t!r}")
        print(f"  OUT: {out!r}")
        print()
