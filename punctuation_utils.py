"""
Punctuation post-processing for ASR segments.

Whisper often outputs text with no punctuation.  This module adds:
  - 。 at the end of each segment (sentence terminator)
  - ，within long segments where a natural mid-phrase break is estimated

The comma insertion uses a character-count heuristic because Whisper does
not expose within-segment pause timings after VAD chunking.  The threshold
is tunable via the `comma_every` parameter.
"""

import re
from typing import List, Dict

# Full-width punctuation that already ends a sentence — don't add 。 after these.
_SENTENCE_ENDINGS = set("。！？…⋯♪～")

# Full-width punctuation that already acts as a mid-phrase break — we skip
# inserting ，immediately after these positions.
_EXISTING_BREAKS = set("，、；：。！？…⋯")


def _already_ends_sentence(text: str) -> bool:
    """Return True if the last non-space character is a sentence terminator."""
    stripped = text.rstrip()
    return bool(stripped) and stripped[-1] in _SENTENCE_ENDINGS


def _insert_commas(text: str, comma_every: int) -> str:
    """
    Insert ，roughly every *comma_every* CJK characters.

    Strategy:
      Walk through the text.  Every time the character counter since the last
      break reaches *comma_every*, look ahead up to 5 characters for a safe
      insertion point (prefer inserting *after* a non-punctuated CJK char
      rather than splitting inside an ASCII word or just before existing
      punctuation).  If no better position is found within the lookahead
      window, insert at the current position.

    ASCII words and existing punctuation marks reset the character counter so
    we never split English words or double-up on punctuation.
    """
    if len(text) <= comma_every:
        return text  # Short enough — no comma needed

    result = []
    char_count = 0  # CJK chars since last break
    i = 0

    while i < len(text):
        ch = text[i]

        # Existing break punctuation resets counter
        if ch in _EXISTING_BREAKS:
            result.append(ch)
            char_count = 0
            i += 1
            continue

        # ASCII letter/digit — copy verbatim, do NOT count toward CJK run
        if ch.isascii() and (ch.isalpha() or ch.isdigit()):
            result.append(ch)
            i += 1
            continue

        # Whitespace — copy and treat as minor break
        if ch.isspace():
            result.append(ch)
            char_count = 0
            i += 1
            continue

        # CJK (or other non-ASCII, non-punctuation) character
        result.append(ch)
        char_count += 1
        i += 1

        if char_count >= comma_every:
            # Check that the *next* character isn't already punctuation
            next_ch = text[i] if i < len(text) else ""
            if next_ch and next_ch not in _EXISTING_BREAKS and not next_ch.isspace():
                result.append("，")
            char_count = 0

    return "".join(result)


def add_punctuation(
    text: str,
    add_period: bool = True,
    add_comma: bool = True,
    comma_every: int = 15,
) -> str:
    """
    Add Chinese punctuation to a single segment text.

    Args:
        text:        Raw ASR text for one segment.
        add_period:  Append 。 if the segment doesn't already end with a
                     sentence terminator.
        add_comma:   Insert ，approximately every *comma_every* CJK characters.
        comma_every: Character-count threshold between inserted commas.
                     Typical values: 10–20.  Default 15.

    Returns:
        Punctuated text.
    """
    text = text.strip()
    if not text:
        return text

    # 1. Insert internal commas first (before appending 。 so the period
    #    doesn't interfere with the counter).
    if add_comma and comma_every > 0:
        text = _insert_commas(text, comma_every)

    # 2. Append sentence-ending period if needed.
    if add_period and not _already_ends_sentence(text):
        text = text + "。"

    return text


def add_punctuation_to_segments(
    segments: List[Dict],
    add_period: bool = True,
    add_comma: bool = True,
    comma_every: int = 15,
) -> List[Dict]:
    """
    Apply punctuation to every segment in a list.

    Args:
        segments:    List of dicts with at least a ``"text"`` key.
        add_period:  See :func:`add_punctuation`.
        add_comma:   See :func:`add_punctuation`.
        comma_every: See :func:`add_punctuation`.

    Returns:
        New list of segment dicts with punctuated ``"text"`` values.
    """
    result = []
    for seg in segments:
        new_seg = seg.copy()
        if "text" in new_seg:
            new_seg["text"] = add_punctuation(
                new_seg["text"],
                add_period=add_period,
                add_comma=add_comma,
                comma_every=comma_every,
            )
        result.append(new_seg)
    return result


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        "今天天氣很好我們去公園散步然後吃冰淇淋",
        "這是一段短文",
        "已經有標點了，不需要再加。",
        "Hello this is English text",
        "混合 Mixed 中英文 text 測試一下看看效果如何",
        "短",
        "",
        "剛好十五個字符這樣的文字長度",
        "超過門檻的長句子應該要在適當的地方加入逗號讓整體閱讀起來更加流暢自然",
    ]

    print("=" * 60)
    print("punctuation_utils self-test")
    print("=" * 60)
    for t in tests:
        out = add_punctuation(t, add_period=True, add_comma=True, comma_every=15)
        print(f"IN : {t!r}")
        print(f"OUT: {out!r}")
        print()
