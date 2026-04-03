"""
Hakka-to-Mandarin translation module using a local vLLM server.

Translates Hakka Chinese characters (客語漢字) to Traditional Mandarin Chinese (繁體中文).
Only active when ENABLE_LLM=true is set in the environment.

vLLM exposes an OpenAI-compatible REST API at /v1/chat/completions.
"""

import os
import csv
import re
from collections import defaultdict
from typing import List, Dict, Optional
import requests

# vLLM service endpoint — resolved via Docker internal DNS when running in compose
VLLM_HOST = os.environ.get("VLLM_HOST", "http://vllm:8000")
# Full HuggingFace model ID, must match the --model argument passed to vLLM at startup
VLLM_MODEL = os.environ.get("VLLM_MODEL", "google/gemma-4-26B-A4B-it")

# Timeout per API call.
# Batch calls may take longer than single-line calls, so set generously.
VLLM_TIMEOUT = int(os.environ.get("VLLM_TIMEOUT", "300"))  # seconds

# Number of lines per batch. 5 is the sweet spot:
# - Fast enough (~4x vs line-by-line)
# - Small enough that the model rarely merges lines
# - Easy to retry individually on mismatch
BATCH_SIZE = int(os.environ.get("VLLM_BATCH_SIZE", "5"))

# Path to the Hakka–Mandarin lexicon CSV (relative to working dir or absolute)
LEXICON_PATH = os.environ.get(
    "HAKKA_LEXICON_PATH",
    os.path.join(os.path.dirname(__file__), "lexicon", "hakka_to_mandarin.csv"),
)

# Maximum number of matched lexicon entries to inject per batch
LEXICON_MAX_HINTS = int(os.environ.get("LEXICON_MAX_HINTS", "20"))

# Minimum Hakka term length — single chars are too ambiguous to be useful hints
LEXICON_MIN_TERM_LEN = int(os.environ.get("LEXICON_MIN_TERM_LEN", "2"))


# Internal type: terms grouped by length for O(1) bucket lookup
# { char_length: { hakka_term: [mandarin, ...] } }
_LexiconIndex = Dict[int, Dict[str, List[str]]]


def load_lexicon(path: str = LEXICON_PATH) -> "_LexiconIndex":
    """
    Load the Hakka→Mandarin CSV and return a length-indexed structure.

    CSV format (no header): 客語漢字,華語漢字
    - Terms shorter than LEXICON_MIN_TERM_LEN chars are skipped (too noisy).
    - Duplicate Mandarin entries for the same Hakka term are silently dropped.
    - Returns an empty dict if the file is missing or unreadable.

    Index structure: { term_length: { hakka_term: [mandarin, ...] } }
    Pre-grouping by length lets build_lexicon_hint() do longest-match-first
    without re-sorting on every call.
    """
    flat: Dict[str, List[str]] = defaultdict(list)
    if not os.path.exists(path):
        print(f"⚠️  Lexicon file not found: {path}")
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) < 2:
                    continue
                hakka, mandarin = row[0].strip(), row[1].strip()
                if len(hakka) < LEXICON_MIN_TERM_LEN:
                    continue
                if hakka and mandarin and mandarin not in flat[hakka]:
                    flat[hakka].append(mandarin)
    except Exception as e:
        print(f"⚠️  Failed to load lexicon: {e}")
        return {}

    index: Dict[int, Dict[str, List[str]]] = defaultdict(dict)
    for hakka, mandarin_list in flat.items():
        index[len(hakka)][hakka] = mandarin_list

    total = sum(len(v) for v in index.values())
    print(f"✅ Lexicon loaded: {total} entries ({len(index)} length buckets) from {path}")
    return dict(index)


def build_lexicon_hint(texts: List[str], lexicon: "_LexiconIndex") -> str:
    """
    Scan the given texts for Hakka terms present in the length-indexed lexicon.
    Returns a formatted hint block to append to the system prompt,
    or an empty string when nothing matches.

    Iterates length buckets from longest to shortest (longest-match-first),
    so multi-character terms (e.g. "殺人放火") take priority over
    their substrings (e.g. "殺人") without an extra sort on every call.
    """
    combined = "\n".join(texts)

    matched: Dict[str, List[str]] = {}
    for length in sorted(lexicon.keys(), reverse=True):
        if len(matched) >= LEXICON_MAX_HINTS:
            break
        for term, mandarin_list in lexicon[length].items():
            if len(matched) >= LEXICON_MAX_HINTS:
                break
            if term in combined and term not in matched:
                matched[term] = mandarin_list

    if not matched:
        return ""

    lines = [f"{hakka} → {'／'.join(mandarin)}" for hakka, mandarin in matched.items()]
    hint = (
        "\n【客華詞彙對照表】"
        # "下列詞彙為標準對等譯法，請優先採用，並依上下文調整、選擇為最符合華語語意習慣的方式：\n"
        + "\n".join(lines)
    )
    return hint


# Default system prompt — also used as the UI placeholder / default value
DEFAULT_SYSTEM_PROMPT = (
    "你是一位專業的臺灣客語翻譯員。"
    "使用者會提供客語漢字，請將每一句翻譯成自然流暢的繁體中文（臺灣用語）。"
    "規則：\n"
    "1. 只輸出翻譯結果，不要解釋，適當地加上標點符號（逗號或句號）。\n"
    "2. 保持與原文相近的句子長度。\n"
    "3. 若原文已是繁體中文，則原文照傳回。\n"
    "4. 一行輸入對應一行輸出，行數必須完全相同。\n"
    "5. 若提供了【客華詞彙對照表】，總譯時必須優先採用其中的詞彙，非必要時才可依上下文微調。"
)


def is_llm_enabled() -> bool:
    """Check if LLM translation is enabled via environment variable."""
    return os.environ.get("ENABLE_LLM", "false").lower() == "true"


def check_vllm_available() -> bool:
    """
    Ping the vLLM service to verify it's reachable and the expected model is loaded.
    Uses the OpenAI-compatible GET /v1/models endpoint.
    Returns False gracefully if anything fails.
    """
    try:
        resp = requests.get(f"{VLLM_HOST}/v1/models", timeout=5)
        if resp.status_code != 200:
            print(f"⚠️  vLLM /v1/models returned HTTP {resp.status_code}")
            return False
        model_ids = [m["id"] for m in resp.json().get("data", [])]
        available = VLLM_MODEL in model_ids
        if not available:
            print(
                f"⚠️  vLLM is running but model '{VLLM_MODEL}' is not loaded. "
                f"Available: {model_ids}"
            )
        return available
    except Exception as e:
        print(f"⚠️  vLLM not reachable at {VLLM_HOST}: {e}")
        return False


def _call_vllm(system_prompt: str, user_content: str) -> Optional[str]:
    """
    Single raw call to the vLLM OpenAI-compatible chat completions API.
    Returns the assistant's reply string, or None on failure.
    """
    payload = {
        "model": VLLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ],
        "stream": False,
        "temperature": 0.1,
        "max_tokens": 512,
    }
    try:
        resp = requests.post(
            f"{VLLM_HOST}/v1/chat/completions",
            json=payload,
            timeout=VLLM_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"❌ vLLM call error: {e}")
        return None


def _translate_one(text: str, system_prompt: str) -> str:
    """
    Translate a single line. Falls back to original text on failure.
    Only the first non-empty reply line is used to avoid stray commentary.
    """
    reply = _call_vllm(system_prompt, text)
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

    reply = _call_vllm(system_prompt, "\n".join(texts))

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
    use_lexicon: bool = False,
    lexicon: Optional[Dict[str, List[str]]] = None,
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
        use_lexicon:       If True, inject matched lexicon terms into each batch's
                           system prompt as translation hints.
        lexicon:           Pre-loaded lexicon dict; loaded from LEXICON_PATH if None
                           and use_lexicon is True.
        progress_callback: Optional callback(percent, message).

    Returns:
        List of segments with translated text.
    """
    if not segments:
        return segments

    if not is_llm_enabled():
        print("ℹ️  LLM translation disabled (ENABLE_LLM != true)")
        return segments

    if not check_vllm_available():
        print("⚠️  vLLM unavailable — skipping translation, keeping original text")
        return segments

    effective_prompt = (system_prompt or "").strip() or DEFAULT_SYSTEM_PROMPT
    if effective_prompt != DEFAULT_SYSTEM_PROMPT:
        print("ℹ️  Using custom system prompt")

    # Load lexicon once if lexicon-augmented prompting is requested
    _lexicon: Dict[str, List[str]] = {}
    if use_lexicon:
        _lexicon = lexicon if lexicon is not None else load_lexicon()
        print(f"📚 Lexicon augmentation enabled ({len(_lexicon)} entries)")

    total = len(segments)
    total_batches = (total + batch_size - 1) // batch_size
    print(
        f"🈯 Starting Hakka → Mandarin translation "
        f"({total} segments, batch_size={batch_size}, "
        f"{total_batches} batches, timeout={VLLM_TIMEOUT}s)..."
    )

    translated_segments = [seg.copy() for seg in segments]

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end   = min(start + batch_size, total)
        batch_texts = [seg["text"].strip() for seg in segments[start:end]]

        if progress_callback:
            pct = int((batch_idx / total_batches) * 100)
            progress_callback(pct, f"Translating batch {batch_idx + 1}/{total_batches}...")

        # Build per-batch prompt: base prompt + lexicon hints for this batch
        batch_prompt = effective_prompt
        if use_lexicon and _lexicon:
            hint = build_lexicon_hint(batch_texts, _lexicon)
            if hint:
                batch_prompt = effective_prompt + hint

        translated = _translate_batch(batch_texts, system_prompt=batch_prompt)

        for i, text in enumerate(translated):
            translated_segments[start + i]["text"] = text
            print(f"  [{start + i + 1}/{total}] {batch_texts[i]!r} → {text!r}")

    if progress_callback:
        progress_callback(100, "Translation complete")

    print(f"✅ Translation complete: {total} segments")
    return translated_segments


def check_vllm_ready():
    """
    Verify vLLM is reachable and the expected model is loaded.
    Called once at startup. Unlike Ollama, vLLM downloads the model
    at container start via the --model flag — no pull step needed here.
    """
    if not is_llm_enabled():
        return

    print(f"🤖 Checking vLLM availability at {VLLM_HOST} (model: {VLLM_MODEL})...")
    if check_vllm_available():
        print(f"✅ vLLM model '{VLLM_MODEL}' is ready")
    else:
        print(
            f"⚠️  vLLM not ready. Ensure the vLLM container has started and "
            f"--model {VLLM_MODEL} is specified in docker-compose.yml."
        )
