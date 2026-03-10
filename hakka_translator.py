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

# Timeout per line. Line-by-line mode means each call is short,
# but first call on a cold model can still be slow.
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "300"))  # seconds

# Default system prompt — also used as the UI placeholder / default value
DEFAULT_SYSTEM_PROMPT = (
    "你是一位專業的客語翻譯員。"
    "使用者會提供一句客語漢字字幕，請將它翻譯成自然流暢的繁體中文（台灣用語）。"
    "規則：\n"
    "1. 只輸出翻譯結果，不要解釋、不要加注音或標點以外的任何內容。\n"
    "2. 保持與原文相近的句子長度。\n"
    "3. 若原文已是繁體中文，則原文照傳回。\n"
    "4. 只輸出一行。"
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


def _translate_one(text: str, system_prompt: str) -> Optional[str]:
    """
    Translate a single line. Returns the translated string, or None on failure.
    Only the first non-empty line of the reply is used to avoid stray commentary.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": text},
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 256,
        },
    }
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        reply = resp.json()["message"]["content"].strip()
        # Take only the first non-empty line
        first_line = next(
            (l.strip() for l in reply.splitlines() if l.strip()),
            None,
        )
        return first_line if first_line else None
    except Exception as e:
        print(f"❌ Ollama call error: {e}")
        return None


def translate_segments(
    segments: List[Dict],
    system_prompt: Optional[str] = None,
    batch_size: int = 20,       # kept for API compatibility, not used
    progress_callback=None,
) -> List[Dict]:
    """
    Translate all segment texts from Hakka to Traditional Mandarin,
    one line at a time for maximum reliability.

    Args:
        segments:          List of segment dicts with 'start', 'end', 'text'.
        system_prompt:     Custom system prompt; falls back to DEFAULT_SYSTEM_PROMPT
                           when None or empty.
        batch_size:        Unused (kept for backwards compatibility).
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
    print(
        f"🈯 Starting Hakka → Mandarin translation "
        f"({total} segments, line-by-line, timeout={OLLAMA_TIMEOUT}s/line)..."
    )

    translated_segments = [seg.copy() for seg in segments]

    for i, seg in enumerate(segments):
        if progress_callback:
            pct = int((i / total) * 100)
            progress_callback(pct, f"Translating segment {i + 1}/{total}...")

        text = seg["text"].strip()
        result = _translate_one(text, effective_prompt)

        if result is not None:
            translated_segments[i]["text"] = result
            print(f"  [{i + 1}/{total}] {text!r} → {result!r}")
        else:
            print(f"  [{i + 1}/{total}] ⚠️  failed — keeping original: {text!r}")

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
