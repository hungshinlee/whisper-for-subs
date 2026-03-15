"""
Whisper transcription module using faster-whisper for efficient inference.
"""

import warnings
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*", category=UserWarning)

import os
import re
import tempfile
import subprocess
from typing import List, Optional, Generator
import numpy as np
import torch

from faster_whisper import WhisperModel
from vad import SileroVAD


def _patch_model_config(model_dir: str, n_mels: int) -> None:
    """
    Ensure the CT2 model's config files have the correct n_mels value.

    Patches both config.json and preprocessor_config.json because different
    versions of faster-whisper / transformers read from different files to
    construct the FeatureExtractor.  Patching both is safe and guarantees the
    correct mel-bin count regardless of the library version in use.
    """
    import json

    files_to_patch = {
        "config.json": ["num_mel_bins", "n_mels"],
        "preprocessor_config.json": ["num_mel_bins", "feature_size"],
    }

    for filename, keys in files_to_patch.items():
        config_path = os.path.join(model_dir, filename)

        if not os.path.exists(config_path):
            config = {}
        else:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

        needs_patch = any(config.get(k) != n_mels for k in keys)
        if not needs_patch:
            print(f"✅ {filename} already has n_mels={n_mels}")
            continue

        for k in keys:
            config[k] = n_mels

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        print(f"✅ Patched {filename}: {keys} = {n_mels} in {model_dir}")


# ---------------------------------------------------------------------------
# Private model IDs — loaded from environment variables so they are never
# committed to the repository.  Set these in your .env file.
# ---------------------------------------------------------------------------
_HAKKA_V2_MODEL  = os.environ.get("HAKKA_V2_MODEL",  "")
_HAKKA_V3_MODEL  = os.environ.get("HAKKA_V3_MODEL",  "")
_TAIGI_MODEL     = os.environ.get("TAIGI_MODEL",     "")


def ensure_model_ready(model_name: str) -> str:
    """
    Ensure the model is in CTranslate2 format.
    If it's a known non-CT2 model, convert it automatically.

    Args:
        model_name: Name of the model (e.g., "large-v3" or HF repo ID)

    Returns:
        Path to the usable model (CT2 format)
    """
    # Models that need conversion from HuggingFace transformers → CTranslate2
    CUSTOM_MODELS: dict = {}
    if _HAKKA_V2_MODEL:
        CUSTOM_MODELS[_HAKKA_V2_MODEL] = ("whisper-large-v2-taiwanese-hakka-v1-ct2", 80)
    if _HAKKA_V3_MODEL:
        CUSTOM_MODELS[_HAKKA_V3_MODEL] = ("whisper-large-v3-taiwanese-hakka-ct2", 128)

    # Models already in CT2 format on HuggingFace — no conversion needed,
    # but we still patch n_mels in the cached snapshot after download.
    NATIVE_CT2_MODELS: dict = {}
    if _TAIGI_MODEL:
        NATIVE_CT2_MODELS[_TAIGI_MODEL] = 80

    if model_name in NATIVE_CT2_MODELS:
        n_mels = NATIVE_CT2_MODELS[model_name]
        import glob
        hf_cache = os.environ.get("HF_HOME", "/root/.cache/huggingface")
        owner, repo = model_name.split("/", 1)
        pattern = os.path.join(
            hf_cache, "hub",
            f"models--{owner}--{repo}",
            "snapshots", "*",
        )
        for snapshot_dir in glob.glob(pattern):
            _patch_model_config(snapshot_dir, n_mels=n_mels)
        return model_name  # load directly from HF (already CT2)

    if model_name not in CUSTOM_MODELS:
        is_v3 = "v3" in model_name.lower()
        if is_v3:
            import glob
            hf_cache = os.environ.get("HF_HOME", "/root/.cache/huggingface")
            pattern = os.path.join(
                hf_cache, "hub",
                f"models--Systran--faster-{model_name.replace('/', '-')}",
                "snapshots", "*",
            )
            for snapshot_dir in glob.glob(pattern):
                _patch_model_config(snapshot_dir, n_mels=128)
        return model_name

    target_dirname, n_mels = CUSTOM_MODELS[model_name]

    cache_dir = os.environ.get("HF_HOME", "/root/.cache/huggingface")
    models_dir = os.path.join(cache_dir, "ct2_converted")
    target_dir = os.path.join(models_dir, target_dirname)

    if os.path.exists(os.path.join(target_dir, "model.bin")):
        print(f"✅ Found converted model at: {target_dir}")
        _patch_model_config(target_dir, n_mels=n_mels)
        return target_dir

    print(f"⚠️  Model {model_name} needs conversion to CTranslate2 format.")
    print(f"   Converting to {target_dir}...")

    os.makedirs(target_dir, exist_ok=True)

    try:
        cmd = [
            "ct2-transformers-converter",
            "--model", model_name,
            "--output_dir", target_dir,
            "--quantization", "float16",
            "--force",
        ]
        print(f"   Running: {' '.join(cmd)}")
        subprocess.check_call(cmd)
        print("✅ Conversion complete!")

        _patch_model_config(target_dir, n_mels=n_mels)
        return target_dir

    except subprocess.CalledProcessError as e:
        print(f"❌ Conversion failed with code {e.returncode}")
        if os.path.exists(target_dir):
            import shutil
            shutil.rmtree(target_dir)
        return model_name
    except Exception as e:
        print(f"❌ Conversion error: {str(e)}")
        return model_name


# Supported languages for Whisper
SUPPORTED_LANGUAGES = {"auto": "Auto", "zh": "Mandarin", "en": "English"}

# Model configurations with labels
# General models are always available; language-specific models are added
# only when the corresponding environment variable is set.
MODEL_CONFIGS: dict = {
    "large-v3": {
        "label": "General",
        "display_name": "whisper-large-v3",
    },
    "large-v3-turbo": {
        "label": "General",
        "display_name": "whisper-large-v3-turbo",
    },
}
if _HAKKA_V2_MODEL:
    MODEL_CONFIGS[_HAKKA_V2_MODEL] = {"label": "Hakka", "display_name": "whisper-large-v2-hakka"}
if _HAKKA_V3_MODEL:
    MODEL_CONFIGS[_HAKKA_V3_MODEL] = {"label": "Hakka", "display_name": "whisper-large-v3-hakka"}
if _TAIGI_MODEL:
    MODEL_CONFIGS[_TAIGI_MODEL]    = {"label": "Taigi", "display_name": "whisper-large-v2-taigi"}

MODEL_SIZES = list(MODEL_CONFIGS.keys())

# ---------------------------------------------------------------------------
# Hallucination filter
# ---------------------------------------------------------------------------
# Whisper (especially fine-tuned models) sometimes generates these short filler
# tokens at the end of audio when there is silence or background noise.
# We strip segments whose *entire* text matches one of these patterns (after
# removing punctuation and whitespace).
_HALLUCINATION_PATTERNS: List[re.Pattern] = [re.compile(p) for p in [
    r"^好+[。！？.!?]*$",           # 好 / 好好 / 好。好。
    r"^謝謝.*",                      # 謝謝 / 謝謝您 / 謝謝收看
    r"^字幕.*",                      # 字幕由…
    r"^請.*訂閱.*",                  # 請訂閱 / 請記得訂閱
    r"^Thank[s]?\b.*",               # Thanks / Thank you
    r"^Subtitle[s]?\b.*",
    r"^\s*$",                        # blank
]]

# no_speech_prob threshold — segments above this are very likely silence/noise
_NO_SPEECH_THRESHOLD = float(os.environ.get("WHISPER_NO_SPEECH_THRESHOLD", "0.8"))


def _is_hallucination(text: str, no_speech_prob: float) -> bool:
    """Return True if a segment looks like a Whisper hallucination."""
    if no_speech_prob > _NO_SPEECH_THRESHOLD:
        return True
    stripped = text.strip()
    for pat in _HALLUCINATION_PATTERNS:
        if pat.match(stripped):
            return True
    return False


def _normalize(text: str) -> str:
    """Strip punctuation and whitespace for repetition comparison."""
    return re.sub(r'[\s\u3000\u0020。！？，、；：.!?,;:\-…⋯♪～]+', '', text)


def filter_repetition_loops(segments: List[dict], max_token_chars: int = 4) -> List[dict]:
    """
    Remove Whisper "repetition loop" hallucinations — consecutive segments
    whose normalized text is identical.

    Whisper sometimes gets stuck on a short token (e.g. 好 / 細義 / 啊)
    in low-energy or noisy audio and emits it repeatedly.  Unlike
    filter_hallucinations(), this filter operates on ALL positions in the
    segment list, not just the tail.

    Rules:
    - Normalize each segment text (strip punctuation + whitespace).
    - Find runs of ≥ 2 consecutive segments with the same normalized text.
    - If the normalized text is ≤ max_token_chars characters (short token):
        remove the ENTIRE run.
    - If the normalized text is > max_token_chars characters (long text):
        keep only the FIRST occurrence, remove the rest.  This is the
        conservative path that avoids accidentally deleting legitimate
        repeated sentences.

    Args:
        segments:        List of segment dicts with 'text' key.
        max_token_chars: Char threshold between "remove all" and
                         "keep first" strategy.  Default 4.

    Returns:
        Filtered segment list.
    """
    if not segments:
        return segments

    # Group segments into consecutive runs by normalized text
    runs: List[List[int]] = []   # each inner list = indices of one run
    current_run = [0]
    current_norm = _normalize(segments[0]["text"])

    for i in range(1, len(segments)):
        norm = _normalize(segments[i]["text"])
        if norm == current_norm:
            current_run.append(i)
        else:
            runs.append(current_run)
            current_run = [i]
            current_norm = norm
    runs.append(current_run)

    keep = set()
    removed_texts = []

    for run in runs:
        if len(run) == 1:
            keep.add(run[0])
            continue

        norm = _normalize(segments[run[0]]["text"])
        is_short = len(norm) <= max_token_chars

        if is_short:
            # Short token repeated ≥ 2 times → remove entire run
            for idx in run:
                removed_texts.append(segments[idx]["text"].strip())
        else:
            # Long text repeated → keep first, remove rest
            keep.add(run[0])
            for idx in run[1:]:
                removed_texts.append(segments[idx]["text"].strip())

    if removed_texts:
        print(f"🔁 Removed {len(removed_texts)} repetition-loop segment(s): "
              f"{removed_texts[:6]}{'…' if len(removed_texts) > 6 else ''}")

    return [s for i, s in enumerate(segments) if i in keep]


def filter_short_token_bursts(segments: List[dict], max_token_chars: int = 2, min_burst: int = 2) -> List[dict]:
    """
    Remove bursts of consecutive very-short segments that signal a Whisper
    "counting / enumerating" hallucination (e.g. 一。二。三。 or 𠊎。一。二。).

    This complements filter_repetition_loops(): that filter catches runs of
    *identical* short tokens; this one catches runs of *different* short tokens.

    Rules:
    - Normalize each segment text (strip punctuation + whitespace).
    - Find runs of >= min_burst consecutive segments whose normalized text is
      <= max_token_chars characters.
    - Remove the entire run.
    - Single isolated short segments are left untouched (conservative).

    Args:
        segments:        List of segment dicts with 'text' key.
        max_token_chars: Normalized-text length threshold.  Default 2.
        min_burst:       Minimum consecutive short segments to trigger removal.
                         Default 2 (i.e. two or more in a row).

    Returns:
        Filtered segment list.
    """
    if not segments:
        return segments

    is_short = [len(_normalize(s["text"])) <= max_token_chars for s in segments]

    # Build runs of consecutive short segments
    to_remove: set = set()
    i = 0
    while i < len(segments):
        if is_short[i]:
            run_start = i
            while i < len(segments) and is_short[i]:
                i += 1
            run_len = i - run_start
            if run_len >= min_burst:
                for j in range(run_start, run_start + run_len):
                    to_remove.add(j)
        else:
            i += 1

    if to_remove:
        removed_texts = [segments[i]["text"].strip() for i in sorted(to_remove)]
        print(f"🔢 Removed {len(to_remove)} short-token-burst segment(s): "
              f"{removed_texts[:6]}{'…' if len(removed_texts) > 6 else ''}")

    return [s for i, s in enumerate(segments) if i not in to_remove]


def filter_hallucinations(segments: List[dict]) -> List[dict]:
    """
    Remove hallucinated segments produced by Whisper at end-of-audio silence.

    Only removes segments that are flagged AND appear after the last
    non-hallucinated segment, so genuine content is never stripped.
    """
    if not segments:
        return segments

    # Mark each segment
    flags = [
        _is_hallucination(s["text"], s.get("no_speech_prob", 0.0))
        for s in segments
    ]

    # Find the last non-hallucinated segment
    last_real = -1
    for i in range(len(segments) - 1, -1, -1):
        if not flags[i]:
            last_real = i
            break

    if last_real == -1:
        # Everything was flagged — return all (avoid empty output)
        return segments

    kept = segments[: last_real + 1]
    removed = len(segments) - len(kept)
    if removed:
        removed_texts = [s["text"].strip() for s in segments[last_real + 1:]]
        print(f"🧹 Removed {removed} hallucinated segment(s): {removed_texts}")
    return kept


# ---------------------------------------------------------------------------

def _words_from_seg(seg, time_offset: float = 0.0) -> List[dict]:
    """
    Extract word-level timing from a faster-whisper segment object.

    Returns a list of dicts with keys: word, start, end, probability.
    Returns an empty list if the segment has no word data (word_timestamps
    was not requested, or the model didn't produce alignment).
    """
    if not hasattr(seg, "words") or seg.words is None:
        return []
    return [
        {
            "word": w.word,
            "start": time_offset + w.start,
            "end": time_offset + w.end,
            "probability": getattr(w, "probability", 1.0),
        }
        for w in seg.words
    ]


class WhisperTranscriber:
    """Whisper-based transcription with VAD support."""

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        device_index: int = 0,
        compute_type: str = "float16",
        use_vad: bool = True,
        vad_threshold: float = 0.5,
        min_silence_duration_ms: int = 100,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.use_vad = use_vad

        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            self.device = "cpu"
            self.compute_type = "float32"

        self.gpu_index = None
        if self.device == "cuda" and torch.cuda.is_available():
            self.gpu_index = device_index
            print(f"🎯 Single-GPU mode: Using GPU {self.gpu_index}")

        print(f"Loading Whisper model: {model_size} on {self.device}:{device_index}")

        actual_model_path = ensure_model_ready(model_size)
        if actual_model_path != model_size:
            print(f"   Using converted model path: {actual_model_path}")

        self.model = WhisperModel(
            actual_model_path,
            device=self.device,
            device_index=device_index,
            compute_type=self.compute_type,
        )
        print("✅ Model loaded successfully")

        # Verify FeatureExtractor mel bins
        is_v3_model = "v3" in model_size.lower()
        expected_n_mels = 128 if is_v3_model else 80
        fe = getattr(self.model, "feature_extractor", None)
        actual_n_mels = None
        if fe is not None:
            if hasattr(fe, "mel_filters") and fe.mel_filters is not None:
                actual_n_mels = int(fe.mel_filters.shape[0])
            elif hasattr(fe, "feature_size"):
                actual_n_mels = int(fe.feature_size)
            elif hasattr(fe, "n_mels"):
                actual_n_mels = int(fe.n_mels)

        if actual_n_mels == expected_n_mels:
            print(f"✅ FeatureExtractor n_mels verified: {actual_n_mels} bins")
        elif actual_n_mels is not None:
            raise RuntimeError(
                f"FeatureExtractor has {actual_n_mels} mel bins but model "
                f"'{model_size}' requires {expected_n_mels}. "
                f"Delete the cached model and rebuild:\n"
                f"  docker compose down\n"
                f"  docker volume rm whisper-models\n"
                f"  docker compose build --no-cache\n"
                f"  docker compose up -d"
            )
        else:
            print("⚠️  Could not verify FeatureExtractor n_mels — proceeding")

        self.vad = None
        if use_vad:
            print(
                f"Loading Silero VAD (min_silence_duration={min_silence_duration_ms}ms)..."
            )
            self.vad = SileroVAD(
                threshold=vad_threshold,
                min_silence_duration_ms=min_silence_duration_ms,
            )
            print("✅ VAD loaded successfully")

    def load_audio(self, file_path: str, sample_rate: int = 16000) -> np.ndarray:
        """Load audio file and convert to proper format."""
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_wav.close()

        try:
            cmd = [
                "ffmpeg", "-i", file_path,
                "-ar", str(sample_rate),
                "-ac", "1",
                "-f", "wav",
                "-y", temp_wav.name,
            ]
            subprocess.run(cmd, capture_output=True, check=True)

            import soundfile as sf
            audio, sr = sf.read(temp_wav.name, dtype="float32")
            return audio

        finally:
            if os.path.exists(temp_wav.name):
                os.unlink(temp_wav.name)

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe",
        initial_prompt: Optional[str] = None,
        word_timestamps: bool = True,
        progress_callback=None,
        vad_chunks=None,
    ) -> List[dict]:
        """
        Transcribe audio file.

        Args:
            vad_chunks: Optional pre-computed VAD chunks from an external VAD step
                        (list of (start, end, audio_data) tuples).  When provided,
                        the internal VAD step is skipped so that speech detection
                        always runs on the (potentially enhanced) audio before this
                        method is called.

        Returns:
            List of segments with start, end, text, no_speech_prob, and words.
        """
        import time

        start_time = time.time()

        if progress_callback:
            progress_callback(0, "Loading audio...")

        audio = self.load_audio(audio_path)
        duration = len(audio) / 16000

        print(f"📊 Audio loaded: {duration:.1f}s ({len(audio)} samples @ 16000Hz)")

        if progress_callback:
            progress_callback(5, f"Audio duration: {duration:.1f} seconds")

        if self.use_vad and self.vad is not None:
            if vad_chunks is not None:
                print(f"🎯 Using pre-computed VAD: {len(vad_chunks)} chunk(s) — skipping internal VAD")
            elif progress_callback:
                progress_callback(10, "Detecting speech segments with VAD...")
            segments = self._transcribe_with_vad(
                audio, duration, language, task, initial_prompt, word_timestamps,
                progress_callback,
                precomputed_chunks=vad_chunks,
            )
        else:
            segments = self._transcribe_direct(
                audio_path, language, task, initial_prompt, word_timestamps,
                progress_callback,
            )

        segments = filter_repetition_loops(segments)
        segments = filter_short_token_bursts(segments)
        segments = filter_hallucinations(segments)

        elapsed = time.time() - start_time
        speed_ratio = duration / elapsed if elapsed > 0 else 0
        gpu_info = f"GPU {self.gpu_index}" if self.gpu_index is not None else "CPU"

        print("✅ Transcription complete!")
        print(f"   Device: {gpu_info}")
        print(f"   Segments: {len(segments)}")
        print(f"   Duration: {duration:.1f}s")
        print(f"   Time: {elapsed:.1f}s")
        print(f"   Speed: {speed_ratio:.1f}x realtime")

        return segments

    def _transcribe_with_vad(
        self,
        audio: np.ndarray,
        duration: float,
        language: Optional[str],
        task: str,
        initial_prompt: Optional[str],
        word_timestamps: bool,
        progress_callback=None,
        precomputed_chunks=None,
    ) -> List[dict]:
        """Transcribe using VAD segmentation."""
        if precomputed_chunks is not None:
            chunks = precomputed_chunks
        else:
            chunks = self.vad.segment_audio(
                audio,
                merge=True,
                min_duration=0.5,
                max_duration=30.0,
            )

        if not chunks:
            print("⚠ No speech detected in audio")
            if progress_callback:
                progress_callback(100, "No speech detected")
            return []

        print(f"🎯 VAD detected {len(chunks)} speech segments")

        if progress_callback:
            progress_callback(15, f"Detected {len(chunks)} speech segments")

        segments = []
        gpu_label = f"GPU {self.gpu_index}" if self.gpu_index is not None else "CPU"

        for i, (chunk_start, chunk_end, chunk_audio) in enumerate(chunks):
            chunk_duration = chunk_end - chunk_start
            print(
                f"[{gpu_label}] ▶ Processing chunk {i + 1}/{len(chunks)} ({chunk_duration:.1f}s)"
            )

            progress = 15 + (i / len(chunks)) * 80
            if progress_callback:
                progress_callback(progress, f"Transcribing ({i + 1}/{len(chunks)})...")

            temp_chunk = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_chunk.close()

            try:
                import soundfile as sf
                sf.write(temp_chunk.name, chunk_audio, 16000)

                result, info = self.model.transcribe(
                    temp_chunk.name,
                    language=language if language != "auto" else None,
                    task=task,
                    initial_prompt=initial_prompt,
                    word_timestamps=word_timestamps,
                    vad_filter=False,  # VAD already done above
                )

                chunk_segments = []
                for seg in result:
                    seg_dict = {
                        "start": chunk_start + seg.start,
                        "end": chunk_start + seg.end,
                        "text": seg.text,
                        "no_speech_prob": getattr(seg, "no_speech_prob", 0.0),
                        "words": _words_from_seg(seg, time_offset=chunk_start),
                    }
                    chunk_segments.append(seg_dict)
                    segments.append(seg_dict)

                print(
                    f"[{gpu_label}] ✓ Chunk {i + 1} complete: {len(chunk_segments)} text segments"
                )

            finally:
                if os.path.exists(temp_chunk.name):
                    os.unlink(temp_chunk.name)

        if progress_callback:
            progress_callback(100, f"Complete! {len(segments)} segments")

        return segments

    def _transcribe_direct(
        self,
        audio_path: str,
        language: Optional[str],
        task: str,
        initial_prompt: Optional[str],
        word_timestamps: bool,
        progress_callback=None,
    ) -> List[dict]:
        """Transcribe without VAD."""
        if progress_callback:
            progress_callback(20, "Starting transcription...")

        result, info = self.model.transcribe(
            audio_path,
            language=language if language != "auto" else None,
            task=task,
            initial_prompt=initial_prompt,
            word_timestamps=word_timestamps,
            vad_filter=False,  # Must be False — see parallel_transcriber docstring
        )

        segments = []
        for seg in result:
            segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "no_speech_prob": getattr(seg, "no_speech_prob", 0.0),
                "words": _words_from_seg(seg, time_offset=0.0),
            })

            if progress_callback:
                if info.duration > 0:
                    progress = 20 + (seg.end / info.duration) * 75
                    progress_callback(progress, f"Transcribing... {seg.end:.1f}s")

        if progress_callback:
            progress_callback(100, f"Complete! {len(segments)} segments")

        return segments

    def transcribe_streaming(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe",
    ) -> Generator[dict, None, None]:
        """Transcribe audio with streaming output (no word timestamps)."""
        result, info = self.model.transcribe(
            audio_path,
            language=language if language != "auto" else None,
            task=task,
            vad_filter=True,
        )

        for seg in result:
            yield {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
            }


def get_available_devices() -> List[str]:
    """Get list of available compute devices."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devices.append(f"cuda:{i}")
        devices.insert(1, "cuda")
    return devices


def get_gpu_info() -> List[dict]:
    """Get information about available GPUs."""
    info = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info.append({
                "index": i,
                "name": props.name,
                "memory_total": props.total_memory / (1024**3),
                "memory_free": torch.cuda.memory_reserved(i) / (1024**3),
            })
    return info
