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

# Timeout for a single translation batch request.
# gemma3:12b cold-load on a 2080 Ti can take 60–120 s on first inference;
# subsequent calls are much faster once the model is warm in VRAM.
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "300"))  # seconds

# Default system prompt — also used as the UI placeholder / default value
DEFAULT_SYSTEM_PROMPT = (
    "你是一位專業的客語翻譯員。"
    "使用者會提供客語漢字字幕，請將每一句翻譯成自然流暢的繁體中文（台灣用語）。"
    "規則：\n"
    "1. 只輸出翻譯結果，不要解釋、不要加注音或標點以外的任何內容。\n"
    "2. 保持與原文相近的句子長度。\n"
    "3. 若原文已是繁體中文，則原文照傳回。\n"
    "4. 一行輸入對應一行輸出，行數必須相同。"
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


def _translate_batch(
    texts: List[str],
    system_prompt: str,
) -> Optional[List[str]]:
    """
    Translate a batch of lines.

    Strategy:
    1. Send all lines at once (fast path).
    2. If the LLM returns the wrong number of lines, fall back to
       translating each line individually (slow but reliable).

    Returns a list of translated lines the same length as `texts`,
    or None if every attempt fails.
    """
    if not texts:
        return []

    # ── Fast path: batch ──────────────────────────────────────────────
    user_content = "\n".join(texts)
    reply = _call_ollama(system_prompt, user_content)

    if reply is not None:
        lines = reply.splitlines()
        if len(lines) == len(texts):
            return lines                        # ✅ perfect match

        print(
            f"⚠️  Batch line count mismatch: sent {len(texts)}, got {len(lines)}. "
            f"Retrying line-by-line..."
        )

    # ── Slow path: one line at a time ─────────────────────────────────
    results: List[Optional[str]] = []
    for i, text in enumerate(texts):
        single_reply = _call_ollama(system_prompt, text)
        if single_reply is not None:
            # Take only the first non-empty line to avoid stray commentary
            first_line = next(
                (l.strip() for l in single_reply.splitlines() if l.strip()),
                None,
            )
            results.append(first_line if first_line else text)
        else:
            print(f"⚠️  Line {i + 1} translation failed — keeping original")
            results.append(text)        # fallback to original for this line

    # If every single line kept its original, treat the whole batch as failed
    if results == list(texts):
        return None

    return results


def translate_segments(
    segments: List[Dict],
    system_prompt: Optional[str] = None,
    batch_size: int = 20,
    progress_callback=None,
) -> List[Dict]:
    """
    Translate all segment texts from Hakka to Traditional Mandarin.

    Args:
        segments:          List of segment dicts with 'start', 'end', 'text'.
        system_prompt:     Custom system prompt; falls back to DEFAULT_SYSTEM_PROMPT
                           when None or empty.
        batch_size:        Number of lines per LLM call (default 20).
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

    print(
        f"🈯 Starting Hakka → Mandarin translation "
        f"({len(segments)} segments, timeout={OLLAMA_TIMEOUT}s)..."
    )

    translated_segments = [seg.copy() for seg in segments]
    total_batches = (len(segments) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end   = min(start + batch_size, len(segments))
        batch_texts = [seg["text"].strip() for seg in segments[start:end]]

        if progress_callback:
            pct = int((batch_idx / total_batches) * 100)
            progress_callback(pct, f"Translating batch {batch_idx + 1}/{total_batches}...")

        translated = _translate_batch(batch_texts, system_prompt=effective_prompt)

        if translated is not None:
            for i, text in enumerate(translated):
                translated_segments[start + i]["text"] = text
        else:
            print(f"⚠️  Batch {batch_idx + 1} translation failed entirely — keeping original")

    if progress_callback:
        progress_callback(100, "Translation complete")

    print(f"✅ Translation complete: {len(translated_segments)} segments")
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
