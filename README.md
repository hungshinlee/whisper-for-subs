# FormoSTT

**Speech-to-Text System for Taiwanese Languages**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-green.svg)](https://developer.nvidia.com/cuda-toolkit)

Production-grade ASR service built on [faster-whisper](https://github.com/guillaumekln/faster-whisper), purpose-built for Taiwan's multilingual landscape. Supports Mandarin, Hakka, Taigi, and English; outputs SRT subtitle files; ships as a Docker image with a Gradio web UI and a Gradio-compatible REST API.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Audio Processing Pipeline](#2-audio-processing-pipeline)
3. [Hallucination Filtering](#3-hallucination-filtering)
4. [Multi-GPU Scheduling — TranscriberPool](#4-multi-gpu-scheduling--transcriberpool)
5. [Model Management](#5-model-management)
6. [Hakka Translation Pipeline](#6-hakka-translation-pipeline)
7. [Supported Models](#7-supported-models)
8. [Environment Variables](#8-environment-variables)
9. [Deployment](#9-deployment)
10. [API Reference](#10-api-reference)
11. [Project Structure](#11-project-structure)
12. [Hardware Requirements](#12-hardware-requirements)
13. [Acknowledgements](#13-acknowledgements)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Gradio UI / REST API  (app.py)                                 │
│                                                                 │
│  TranscriberPool  ←─ GPU semaphore-based serialisation          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  GPU 0       │  │  GPU 1       │  │  GPU N       │          │
│  │  Transcriber │  │  Transcriber │  │  Transcriber │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                 │
│  Per-request pipeline                                           │
│  Input → Enhancement → VAD → Whisper → Filters → SRT           │
│                    ↓                                            │
│             [Optional] Ollama LLM (Hakka → Mandarin)           │
└─────────────────────────────────────────────────────────────────┘
```

The application is a single Python process. Gradio's built-in queue handles HTTP concurrency (`max_size=10`, `default_concurrency_limit=2`); `TranscriberPool` distributes the resulting worker threads across available GPUs.

---

## 2. Audio Processing Pipeline

Every request runs the following stages in order.

### 2.1 Format Normalisation

Uploaded files and YouTube downloads are normalised to **mono WAV at 16 kHz** before any processing:

- Containers that `soundfile` cannot decode directly (MP3, AAC, M4A, OGG, OPUS, WMA, AMR) are converted via `ffmpeg -ar 16000 -ac 1`.
- YouTube audio is downloaded with `yt-dlp` using `FFmpegExtractAudio` post-processor, also targeting 16 kHz mono WAV.

### 2.2 Speech Enhancement (optional)

When enabled, [DeepFilterNet3](https://github.com/Rikorose/DeepFilterNet) suppresses background noise before VAD and transcription.

**Implementation details (`speech_enhancer.py`):**

- DeepFilterNet3 operates natively at **48 kHz**. The 16 kHz input is upsampled via `scipy.signal.resample`, processed, then downsampled back.
- The model is loaded lazily (first call only) and cached for the process lifetime.
- A `mix_factor` parameter (0.0–1.0) blends the enhanced and original signals:
  ```
  output = mix_factor × enhanced + (1 − mix_factor) × original
  ```
- The DeepFilterNet3 weights (~30 MB) are **baked into the Docker image** at build time (`preload_deepfilter.py`) so runtime inference never requires network access.
- `is_deepfilter_available()` uses `importlib.util.find_spec("df")` — no import cost, safe to call at module load time.

### 2.3 Voice Activity Detection

[Silero VAD](https://github.com/snakers4/silero-vad) segments the audio into speech chunks before Whisper inference. This step runs **after** speech enhancement so the detector always operates on clean audio.

**`SileroVAD.segment_audio()` parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold` | 0.5 | Speech probability threshold |
| `min_speech_duration_ms` | 250 ms | Ignore segments shorter than this |
| `min_silence_duration_ms` | 100–2000 ms | Configurable via UI slider |
| `speech_pad_ms` | 30 ms | Padding added around each detected segment |

After raw detection, `merge_short_segments()` applies secondary merging:
- Segments separated by a gap ≤ `max_gap` (0.5 s) and whose combined duration ≤ `max_duration` (30 s) are merged.
- This reduces chunk fragmentation and improves context continuity for the decoder.

The VAD model itself (`snakers4/silero-vad`) is downloaded once during Docker build and cached in a named volume.

### 2.4 Whisper Inference

Each VAD chunk is written to a temporary WAV file and passed to `WhisperModel.transcribe()` with `vad_filter=False` (VAD is already applied above). The model returns a lazy generator; segments are consumed and collected into dicts:

```python
{
    "start": float,          # absolute start time in seconds
    "end": float,            # absolute end time in seconds
    "text": str,             # decoded text
    "no_speech_prob": float, # CTranslate2 silence probability
    "words": [...],          # word-level timestamps (when available)
}
```

Chunk timestamps are offset by `chunk_start` so all segments share a common absolute timeline.

### 2.5 Post-processing

After transcription, segments pass through three hallucination filters (see §3), then optionally:

- **OpenCC s2tw** — Simplified → Traditional Chinese (Taiwan standard), applied when `language=zh` and the user enables the option.
- **Subtitle merging** — `merge_segments()` in `srt_utils.py` merges consecutive segments subject to:
  - Gaps ≤ 50 ms are **always** merged (avoids split artifacts from Whisper cutting mid-word).
  - For larger gaps: combined text ≤ `max_chars` (default 80) **and** combined duration ≤ 5 s.
- **LLM translation** — For Hakka models, segments are batched through an Ollama LLM (see §6).

---

## 3. Hallucination Filtering

Whisper, particularly when fine-tuned on domain-specific data, tends to generate spurious output during silence or low-energy audio. Three independent filters are applied in sequence after `transcribe()`:

### 3.1 `filter_repetition_loops`

Targets consecutive segments whose **normalised text** (punctuation and whitespace stripped) is identical.

```
Rule:
  normalised_len ≤ 4 chars → remove entire run
  normalised_len > 4 chars → keep first occurrence, remove duplicates
```

Example suppressed: `好 / 好 / 好 / 好` (stuck token in silence).

### 3.2 `filter_short_token_bursts`

Targets consecutive runs of ≥ 2 segments whose normalised text is ≤ 2 characters, regardless of whether they are identical. This catches "counting" hallucinations distinct from repetition loops.

Example suppressed: `一。/ 二。/ 三。` (different short tokens, equally meaningless).

### 3.3 `filter_hallucinations`

Uses `no_speech_prob` (from CTranslate2) and a regex pattern list to flag likely hallucinations. Critically, **only trailing flagged segments are removed** — flags in the middle of real content are ignored. This prevents accidental deletion of legitimate content that happens to match a pattern.

**Flag conditions:**
- `no_speech_prob > WHISPER_NO_SPEECH_THRESHOLD` (default 0.8, configurable via env var)
- Text matches any of:
  - `^好+[。！？.!?]*$` — repeated filler character
  - `^謝謝.*` — closing pleasantry
  - `^字幕.*` — subtitle watermark
  - `^請.*訂閱.*` — call-to-action
  - `^Thank[s]?\b.*` — English filler
  - `^Subtitle[s]?\b.*` — English watermark
  - `^\s*$` — blank

---

## 4. Multi-GPU Scheduling — TranscriberPool

`TranscriberPool` (in `app.py`) manages concurrent access to multiple GPU-resident `WhisperTranscriber` instances.

### 4.1 The Concurrency Problem

CTranslate2's `WhisperModel.generate()` is **not thread-safe**. Calling it concurrently from two threads on the same model instance corrupts internal CUDA state, producing `RuntimeError: CUDA failed with error invalid argument`. A naive "least-loaded" counter is insufficient because multiple threads can select the same GPU before any of them begins execution.

### 4.2 Solution: Per-GPU Binary Semaphore

Each GPU slot holds a `threading.Semaphore(1)`. The pool enforces **at-most-one concurrent inference per GPU**. The semaphore is acquired **outside** `self.lock` to prevent deadlocks:

```
Phase 1  (under self.lock):
  Select GPU → increment gpu_active counter → release lock

Phase 2  (without lock):
  sem[gpu_id].acquire()   ← blocking; waits if GPU is running another job

Phase 3  (caller executes transcription)

Phase 4  (release_single_gpu_transcriber):
  decrement gpu_active
  sem[gpu_id].release()   ← wakes next waiting thread for this GPU
```

With N GPUs, up to N requests run truly in parallel. Additional requests queue on the semaphore of their assigned GPU.

### 4.3 GPU Selection — Four-Priority Strategy

The pool does not simply prefer the least-loaded GPU, because that ignores model cache state and semaphore availability:

| Priority | Condition | Action |
|----------|-----------|--------|
| **1** | GPU has the requested model cached **and** semaphore is immediately available | Reuse — zero wait, no reload |
| **2** | Any GPU with `active == 0` | Load model there, run immediately — this is what makes multi-GPU spread work |
| **3** | GPU has the requested model cached but is busy | Queue behind it — avoids evicting and reloading weights |
| **4** | No cached match, all GPUs busy | Load on least-loaded GPU and queue |

Priority 1 uses a non-blocking `sem.acquire(blocking=False)` probe inside the lock. The semaphore is immediately released; the real acquisition happens in Phase 2 after the lock is released.

### 4.4 GPU Detection

```python
# SINGLE_GPU_DEVICES overrides CUDA_VISIBLE_DEVICES
_gpu_str = os.environ.get("SINGLE_GPU_DEVICES",
           os.environ.get("CUDA_VISIBLE_DEVICES", ""))
```

With `CUDA_VISIBLE_DEVICES=0,1`, the pool creates slots `{0, 1}` and corresponding semaphores `{0: Semaphore(1), 1: Semaphore(1)}`. Each slot holds at most one `WhisperTranscriber` (one model at a time per GPU).

---

## 5. Model Management

### 5.1 Model Format

`faster-whisper` requires models in **CTranslate2 format**. Three model categories are handled:

| Category | Example | Handling |
|----------|---------|----------|
| Official Whisper (CT2 on HF) | `large-v3`, `large-v3-turbo` | Downloaded directly by faster-whisper |
| Private HF models (HF Transformers format) | Hakka v2/v3 | Auto-converted at first use via `ct2-transformers-converter` |
| Private HF models (already CT2) | Taigi | Downloaded directly; config patched for correct mel bins |

### 5.2 The n_mels Problem

Whisper-v2 models use **80 mel bins**; Whisper-v3 models use **128**. Fine-tuned models hosted on HuggingFace sometimes ship with `config.json` or `preprocessor_config.json` that specifies the wrong value, causing a silent feature extraction mismatch.

`_patch_model_config()` ensures both config files have the correct value at load time. `WhisperTranscriber.__init__()` then verifies the live `FeatureExtractor` object:

```python
if actual_n_mels == expected_n_mels:
    pass  # ok
elif actual_n_mels is not None:
    raise RuntimeError(...)  # fail fast with actionable message
```

### 5.3 Model ID Confidentiality

Private HuggingFace repo IDs are never hardcoded. They are injected via environment variables (`HAKKA_V2_MODEL`, `HAKKA_V3_MODEL`, `TAIGI_MODEL`) and read at import time in `transcriber.py`. `MODEL_CONFIGS` is constructed dynamically — a language option only appears in the UI if the corresponding variable is set.

The UI-level model lists (`GENERAL_MODELS_IDS`, `HAKKA_MODELS_IDS`, `TAIGI_MODELS_IDS`) are derived from `MODEL_CONFIGS` at module load time, making them the single source of truth across the entire codebase.

---

## 6. Hakka Translation Pipeline

When a Hakka model is selected and `ENABLE_LLM=true`, ASR output is post-translated to Traditional Mandarin via a locally-deployed Ollama LLM.

### 6.1 Batch Translation

Segments are processed in batches of `OLLAMA_BATCH_SIZE` (default 5) lines per API call:

```
Fast path:  send N lines joined by \n → expect N lines back
Slow path:  if reply line count mismatches → translate each line individually
```

The slow path is automatic, transparent, and ensures the output segment list is always the same length as the input.

### 6.2 Lexicon Augmentation

`lexicon/hakka_to_mandarin.csv` contains a Hakka–Mandarin term mapping. When `use_lexicon=True`, matched terms from the current batch are injected into the system prompt as translation hints using a **longest-match-first** strategy:

```python
# lexicon indexed by term length for O(1) bucket lookup
{ term_length: { hakka_term: [mandarin_translation, ...] } }
```

Iterating buckets from longest to shortest ensures multi-character compounds take priority over their substrings. Up to `LEXICON_MAX_HINTS` (default 20) hints are injected per batch.

### 6.3 LLM Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_LLM` | `false` | Master switch |
| `OLLAMA_HOST` | `http://ollama:11434` | Service endpoint |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Model tag |
| `OLLAMA_BATCH_SIZE` | `5` | Lines per API call |
| `OLLAMA_TIMEOUT` | `300` | Per-call timeout (seconds) |

The Ollama service is deployed as a separate Docker Compose service on a dedicated `llm` profile:

```bash
docker compose --profile llm up -d
```

In the reference hardware setup, GPU 2–3 are assigned exclusively to Ollama, leaving GPU 0–1 for Whisper ASR.

---

## 7. Supported Models

| Model ID | Language | Task | n_mels | VRAM | Notes |
|----------|----------|------|--------|------|-------|
| `large-v3-turbo` | Auto / Mandarin / English | Transcribe | 128 | ~6 GB | Default; faster-whisper's Systran CT2 build |
| `large-v3` | Auto / Mandarin / English | Transcribe + Translate to EN | 128 | ~10 GB | Full model; enables translation task |
| `HAKKA_V2_MODEL` | Hakka → Mandarin | Transcribe | 80 | ~10 GB | HF Transformers; auto-converted to CT2 |
| `HAKKA_V3_MODEL` | Hakka → Mandarin | Transcribe | 128 | ~10 GB | HF Transformers; auto-converted to CT2 |
| `TAIGI_MODEL` | Taigi → Mandarin | Transcribe | 80 | ~10 GB | Already CT2 on HF; config patched |

> `large-v3-turbo` and all language-specific models support only `task=transcribe`. `large-v3` additionally supports `task=translate` (translate to English).

---

## 8. Environment Variables

All configuration is managed through `.env`. Copy `.env.example` and fill in values. The `.env` file is listed in `.gitignore` and must never be committed.

### Core

| Variable | Default | Description |
|----------|---------|-------------|
| `HUGGING_FACE_HUB_TOKEN` | — | Required for gated/private HF models |
| `WHISPER_MODEL` | `large-v3-turbo` | Default model at startup |
| `WHISPER_COMPUTE_TYPE` | `float16` | CTranslate2 precision (`float16` / `int8` / `float32`) |
| `WHISPER_DEVICE` | `cuda` | `cuda` or `cpu` |
| `CUDA_VISIBLE_DEVICES` | `0,1` | GPUs allocated to the Whisper service |
| `SINGLE_GPU_DEVICES` | *(same as above)* | Override TranscriberPool GPU list without changing CUDA visibility |
| `GRADIO_SERVER_NAME` | `0.0.0.0` | Bind address |
| `GRADIO_SERVER_PORT` | `7860` | HTTP port |
| `GRADIO_PASSWORD` | *(empty)* | Legacy: single-user Basic Auth password (account name `admin`) |
| `PRELOAD_MODEL` | `false` | Load default model into GPU at startup |
| `WHISPER_NO_SPEECH_THRESHOLD` | `0.8` | `no_speech_prob` cutoff for hallucination filter |

### LLM (Hakka Translation)

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_LLM` | `false` | Enable Ollama LLM translation |
| `OLLAMA_HOST` | `http://ollama:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Model tag |
| `OLLAMA_BATCH_SIZE` | `5` | Segments per LLM call |
| `OLLAMA_TIMEOUT` | `300` | Per-call timeout (s) |
| `HAKKA_LEXICON_PATH` | `lexicon/hakka_to_mandarin.csv` | Hakka–Mandarin term dictionary |
| `LEXICON_MAX_HINTS` | `20` | Max terms injected per batch |
| `LEXICON_MIN_TERM_LEN` | `2` | Ignore single-character terms (too ambiguous) |

### Private Model IDs

| Variable | Description |
|----------|-------------|
| `HAKKA_V2_MODEL` | HuggingFace repo ID for Hakka v2 model (Transformers format) |
| `HAKKA_V3_MODEL` | HuggingFace repo ID for Hakka v3 model (Transformers format) |
| `TAIGI_MODEL` | HuggingFace repo ID for Taigi model (CT2 format) |

Leaving a variable empty removes the corresponding language from the UI entirely.

### Multi-user Authentication

User credentials are managed via a `.users` file (one `username:password` per line, `#` comments supported). This file is bind-mounted read-only into the container and is not committed to the repository.

```
# .users
alice:s3cr3t
bob:hunter2
```

The legacy `GRADIO_PASSWORD` env var is still supported as a fallback (maps to account name `admin`).

---

## 9. Deployment

### Prerequisites

- Docker Engine 20.10+ and Docker Compose v2
- NVIDIA GPU with CUDA 12.x (≥11 GB VRAM recommended)
- NVIDIA Container Toolkit
- HuggingFace account with access to any gated models you intend to use

### Quick Start

```bash
# 1. Clone and configure
git clone https://github.com/hungshinlee/whisper-for-subs.git
cd whisper-for-subs
cp .env.example .env
# Edit .env: set HUGGING_FACE_HUB_TOKEN and any private model IDs

# 2. Build and run (Whisper only)
docker compose build
docker compose up -d

# 3. With Hakka LLM translation (requires ENABLE_LLM=true in .env)
docker compose --profile llm up -d

# 4. View logs
docker compose logs -f
```

Access the web UI at `http://<host>:7860`.

### Model Conversion on First Run

When a Hakka model (HF Transformers format) is used for the first time, `ensure_model_ready()` automatically runs `ct2-transformers-converter`:

```
⚠️  Model <repo_id> needs conversion to CTranslate2 format.
   Running: ct2-transformers-converter --model <repo_id> --output_dir <cache_dir> --quantization float16 --force
✅ Conversion complete!
```

Converted models are cached at `$HF_HOME/ct2_converted/` inside the `whisper-models` named volume and reused on subsequent starts.

### Named Volumes

| Volume | Mount | Content |
|--------|-------|---------|
| `whisper-models` | `/root/.cache/huggingface` | Whisper models (CT2 and HF snapshots) |
| `torch-hub` | `/root/.cache/torch/hub` | Silero VAD model |
| `ollama-models` | `/root/.ollama` | Ollama LLM weights |

DeepFilterNet3 weights are baked into the image and do not require a volume.

### GPU Allocation Example (4-GPU Server)

```yaml
# docker-compose.yml
whisper-asr:
  environment:
    - CUDA_VISIBLE_DEVICES=0,1     # Whisper uses GPU 0 and 1

ollama:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            device_ids: ['2', '3']  # Ollama uses GPU 2 and 3
```

### Rebuilding After Code Changes

```bash
docker compose down
docker compose build
docker compose up -d
```

Model volumes are preserved across rebuilds.

---

## 10. API Reference

The Gradio interface exposes a REST API compatible with `gradio_client`.

### Endpoint: `/process_audio`

```python
from gradio_client import Client

client = Client("http://<host>:7860", auth=("username", "password"))

status, asr_srt, asr_file, _, translated_srt, translated_file = client.predict(
    audio_file="/path/to/audio.wav",   # local file path OR None
    youtube_url="",                     # YouTube URL OR empty string
    model_size="large-v3-turbo",        # model ID from MODEL_CONFIGS
    language="auto",                    # auto | zh | en | hakka | taigi
    task="transcribe",                  # transcribe | translate
    use_vad=True,
    min_silence_duration_s=0.2,
    merge_subtitles=True,
    convert_to_traditional=False,
    max_chars=80,
    translate_hakka=False,             # requires ENABLE_LLM=true + Hakka model
    llm_system_prompt="",              # custom prompt; uses default if empty
    use_lexicon=False,
    use_enhancement=False,
    enhancement_mix=1.0,
    api_name="/process_audio",
)
```

### Return Values

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `status` | `str` (HTML) | Progress/completion status HTML |
| 1 | `asr_srt` | `str` | ASR SRT content |
| 2 | `asr_file` | `str` | Path to ASR SRT file |
| 3 | `_` | `gr.update` | Internal UI update (ignore) |
| 4 | `translated_srt` | `str` | Translated SRT content (empty if translation disabled) |
| 5 | `translated_file` | `str \| None` | Path to translated SRT file (None if translation disabled) |

### SRT Output Format

```
1
00:00:00,000 --> 00:00:03,240
下二隻月就愛過年吔，魚仔相關个產品就開始起價

2
00:00:03,500 --> 00:00:07,120
因為呢愛分民眾在防疫期間乜買得著萋萋个魚貨
```

---

## 11. Project Structure

```
whisper-for-subs/
│
├── app.py                  # Entry point: Gradio UI, FastAPI routes, TranscriberPool
├── transcriber.py          # WhisperTranscriber, model management, hallucination filters
├── vad.py                  # SileroVAD wrapper
├── speech_enhancer.py      # DeepFilterNet3 integration
├── hakka_translator.py     # Ollama LLM client, lexicon augmentation
├── srt_utils.py            # SRT generation, parsing, subtitle merging
├── chinese_converter.py    # OpenCC s2tw wrapper
├── youtube_downloader.py   # yt-dlp wrapper
├── preload_deepfilter.py   # Docker build-time DeepFilterNet3 pre-download
├── manage_users.py         # CLI helper for .users credential management
│
├── requirements.txt
├── Dockerfile              # CUDA 12.4 + Python 3.11; bakes in VAD and DF3 models
├── docker-compose.yml      # ASR service + optional Ollama LLM profile
├── .env.example
│
├── lexicon/
│   └── hakka_to_mandarin.csv   # Hakka–Mandarin term dictionary (CSV, no header)
│
├── samples/                    # Built-in example audio files for UI demo
│   ├── *.wav                   # Hakka × 2, Taigi × 1
│   └── *.txt                   # Ground truth transcriptions
│
└── docs/
    └── Terms_and_Privacy.pdf   # Served at /terms-and-privacy
```

---

## 12. Hardware Requirements

### Minimum (single GPU)

- Ubuntu 22.04 / 24.04
- NVIDIA GPU with ≥8 GB VRAM and CUDA 12.x
- 16 GB RAM
- 50 GB disk (OS + Docker image + model cache)

### Reference Configuration (4 × RTX 2080 Ti)

| Resource | Allocation |
|----------|------------|
| GPU 0–1 | Whisper ASR (`CUDA_VISIBLE_DEVICES=0,1`) |
| GPU 2–3 | Ollama LLM (`--profile llm`, `device_ids: ['2','3']`) |
| RAM | 32 GB |
| Disk | 200 GB (multiple model weights) |

### Throughput (RTX 2080 Ti × 4, large-v3-turbo, float16)

| Audio Length | 1 GPU | 2 GPUs | Speedup |
|-------------|-------|--------|---------|
| 10 min | ~60 s | ~35 s | 1.7× |
| 30 min | ~180 s | ~100 s | 1.8× |
| 60 min | ~360 s | ~200 s | 1.8× |

> Speedup is sub-linear due to model loading on first request. Warm speedup with both GPUs loaded approaches 2×.

---

## 13. Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper) — base ASR architecture
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) — CTranslate2 inference engine
- [Silero VAD](https://github.com/snakers4/silero-vad) — voice activity detection
- [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) — neural noise suppression
- [Gradio](https://gradio.app/) — web UI and API framework
- [Ollama](https://ollama.com/) — local LLM inference
- [OpenCC](https://github.com/BYVoid/OpenCC) — Simplified/Traditional Chinese conversion
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) — YouTube audio extraction

### Developers

- **[李鴻欣 Hung-Shin Lee](https://www.linkedin.com/in/hungshinlee)** — 聯和科創股份有限公司
- **[陳力瑋 Li-Wei Chen](mailto:wayne900619@gmail.com)** — 國立清華大學資訊工程學研究所

### Machine Providers (RTX 2080 Ti × 4)

- **[王新民 Hsin-Min Wang](https://homepage.iis.sinica.edu.tw/pages/whm/index_zh.html)** — 中央研究院資訊科學研究所
- **[廖沛俊 Pei-Jun Liao](mailto:newsboy3423@gmail.com)** — 中央研究院資訊科學研究所

---

**© 2024–2026 FormoSTT Team. Released under the MIT License.**
