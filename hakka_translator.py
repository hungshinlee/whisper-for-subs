"""
Hakka-to-Mandarin translation module using a local Ollama LLM.

Translates Hakka Chinese characters (客語漢字) to Traditional Mandarin Chinese (繁體中文).
Only active when ENABLE_LLM=true is set in the environment.
"""

import os
import requests
from typing import List, Dict, Optional

# Ollama service endpoint — resolved via Docker internal DNS when running in compose
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://ollama:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")

# Timeout per API call.
# Batch calls may take longer than single-line calls, so set generously.
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "300"))  # seconds

# Number of lines per batch. 5 is the sweet spot:
# - Fast enough (~4x vs line-by-line)
# - Small enough that Gemma 3 rarely merges lines
# - Easy to retry individually on mismatch
BATCH_SIZE = int(os.environ.get("OLLAMA_BATCH_SIZE", "5"))

# Default system prompt — also used as the UI placeholder / default value
DEFAULT_SYSTEM_PROMPT = (
    "你是一位專業的客語翻譯員。"
    "使用者會提供客語漢字字幕，請將每一句翻譯成自然流暢的繁體中文（台灣用語）。"
    "規則：\n"
    "1. 只輸出翻譯結果，不要解釋、不要加注音或標點以外的任何內容。\n"
    "2. 保持與原文相近的句子長度。\n"
    "3. 若原文已是繁體中文，則原文照傳回。\n"
    "4. 一行輸入對應一行輸出，行數必須完全相同。"
)


def is_llm_enabled() -> bool:
    """Check if LLM translation is enabled via environment variable."""
    return os.environ.get("ENABLE_LLM", "false").lower() == "true"


def check_ollama_available() -> bool:
    """
    Ping the Ollama service to verify it's reachable and the model is loaded.
    Returns False gracefully if anything fails.
    """
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        if resp.status_code != 200:
            return False
        models = [m["name"] for m in resp.json().get("models", [])]
        model_base = OLLAMA_MODEL.split(":")[0]
        available = any(model_base in m for m in models)
        if not available:
            print(
                f"⚠️  Ollama is running but model '{OLLAMA_MODEL}' is not loaded. "
                f"Available: {models}"
            )
        return available
    except Exception as e:
        print(f"⚠️  Ollama not reachable at {OLLAMA_HOST}: {e}")
        return False


def _call_ollama(system_prompt: str, user_content: str) -> Optional[str]:
    """
    Single raw call to Ollama chat API.
    Returns the assistant's reply string, or None on failure.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 512,
        },
    }
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except Exception as e:
        print(f"❌ Ollama call error: {e}")
        return None


def _translate_one(text: str, system_prompt: str) -> str:
    """
    Translate a single line. Falls back to original text on failure.
    Only the first non-empty reply line is used to avoid stray commentary.
    """
    reply = _call_ollama(system_prompt, text)
    if reply is None:
        return text
    first_line = next(
        (l.strip() for l in reply.splitlines() if l.strip()),
        None,
    )
    return first_line if first_line else text


def _translate_batch(texts: List[str], system_prompt: str) -> List[str]:
    """
    Translate a small batch of lines in one API call.

    Strategy:
    1. Send all lines joined by newline (fast path).
    2. If the reply line count mismatches, fall back to line-by-line (slow path).

    Always returns a list the same length as `texts` — never raises.
    """
    if not texts:
        return []

    if len(texts) == 1:
        return [_translate_one(texts[0], system_prompt)]

    reply = _call_ollama(system_prompt, "\n".join(texts))

    if reply is not None:
        lines = [l for l in reply.splitlines() if l.strip()]
        if len(lines) == len(texts):
            return lines   # ✅ perfect match — fast path succeeded

        print(
            f"  ⚠️  Batch mismatch: sent {len(texts)} lines, got {len(lines)}. "
            f"Falling back to line-by-line for this batch..."
        )

    # Slow-path fallback: translate each line individually
    return [_translate_one(t, system_prompt) for t in texts]


def translate_segments(
    segments: List[Dict],
    system_prompt: Optional[str] = None,
    batch_size: int = BATCH_SIZE,
    progress_callback=None,
) -> List[Dict]:
    """
    Translate all segment texts from Hakka to Traditional Mandarin.

    Uses small-batch mode (default batch_size=5) for speed, with automatic
    line-by-line fallback when the LLM returns the wrong number of lines.

    Args:
        segments:          List of segment dicts with 'start', 'end', 'text'.
        system_prompt:     Custom system prompt; falls back to DEFAULT_SYSTEM_PROMPT
                           when None or empty.
        batch_size:        Lines per LLM call (default 5; override via
                           OLLAMA_BATCH_SIZE env var).
        progress_callback: Optional callback(percent, message).

    Returns:
        List of segments with translated text.
    """
    if not segments:
        return segments

    if not is_llm_enabled():
        print("ℹ️  LLM translation disabled (ENABLE_LLM != true)")
        return segments

    if not check_ollama_available():
        print("⚠️  Ollama unavailable — skipping translation, keeping original text")
        return segments

    effective_prompt = (system_prompt or "").strip() or DEFAULT_SYSTEM_PROMPT
    if effective_prompt != DEFAULT_SYSTEM_PROMPT:
        print("ℹ️  Using custom system prompt")

    total = len(segments)
    total_batches = (total + batch_size - 1) // batch_size
    print(
        f"🈯 Starting Hakka → Mandarin translation "
        f"({total} segments, batch_size={batch_size}, "
        f"{total_batches} batches, timeout={OLLAMA_TIMEOUT}s)..."
    )

    translated_segments = [seg.copy() for seg in segments]

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end   = min(start + batch_size, total)
        batch_texts = [seg["text"].strip() for seg in segments[start:end]]

        if progress_callback:
            pct = int((batch_idx / total_batches) * 100)
            progress_callback(pct, f"Translating batch {batch_idx + 1}/{total_batches}...")

        translated = _translate_batch(batch_texts, system_prompt=effective_prompt)

        for i, text in enumerate(translated):
            translated_segments[start + i]["text"] = text
            print(f"  [{start + i + 1}/{total}] {batch_texts[i]!r} → {text!r}")

    if progress_callback:
        progress_callback(100, "Translation complete")

    print(f"✅ Translation complete: {total} segments")
    return translated_segments


def pull_model_if_needed():
    """
    Pull the Ollama model if it hasn't been downloaded yet.
    Called once at startup.
    """
    if not is_llm_enabled():
        return

    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
        if resp.status_code != 200:
            print("⚠️  Could not reach Ollama to check model status")
            return

        models = [m["name"] for m in resp.json().get("models", [])]
        model_base = OLLAMA_MODEL.split(":")[0]

        if any(model_base in m for m in models):
            print(f"✅ Ollama model '{OLLAMA_MODEL}' already available")
            return

        print(f"📥 Pulling Ollama model '{OLLAMA_MODEL}'... (this may take a while)")
        pull_resp = requests.post(
            f"{OLLAMA_HOST}/api/pull",
            json={"name": OLLAMA_MODEL, "stream": False},
            timeout=600,
        )
        if pull_resp.status_code == 200:
            print(f"✅ Model '{OLLAMA_MODEL}' pulled successfully")
        else:
            print(f"❌ Failed to pull model: {pull_resp.text}")

    except Exception as e:
        print(f"⚠️  Model pull failed: {e}")
