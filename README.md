# FormoSTT: Speech-to-Text System for Taiwanese Languages 🎙️

**臺灣語音辨識暨翻譯系統**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![GPU](https://img.shields.io/badge/Multi--GPU-Supported-green.svg)](https://developer.nvidia.com/cuda-toolkit)

使用 OpenAI Whisper 模型的專業級自動語音辨識（ASR）服務，專為台灣語言優化，可將音檔、影片或 YouTube 影片轉換為高品質 SRT 字幕檔。

---

## ✨ 功能特色

### 🌏 台灣多語言支援
- **國語（Mandarin）**：使用 whisper-large-v3 / large-v3-turbo，支援自動偵測
- **客語（Hakka）**：使用客語專用微調模型，可選配 Ollama LLM 翻譯成繁體中文
- **台語（Taigi）**：使用台語專用微調模型，直接輸出繁體中文
- **英文（English）**：支援轉錄或翻譯成英文

### 🎯 智能音訊處理
- **Silero VAD**：精確偵測語音段落，剔除靜音區間，顯著提升辨識準確率
- **DeepFilterNet3 降噪**：可選的語音增強，支援原聲與增強音的混合比例（0.0 – 1.0）
- **幻覺過濾**：三層過濾機制，自動移除 Whisper 在靜音段產生的重複／無意義片段
- **字幕合併**：將過短字幕依最大字元數限制合併，提升閱讀體驗

### ⚡ 多使用者 GPU 調度（TranscriberPool）
- 自動偵測 `CUDA_VISIBLE_DEVICES` 中所有可用 GPU
- 以「最少負載優先」策略分配請求至不同 GPU，避免使用者間相互排隊
- 每個 GPU 快取一個 WhisperTranscriber 實例，避免重複載入模型權重
- 無 GPU 環境時自動降級至 CPU 推理

### 💻 Web 介面
- **Gradio + FastAPI**：美觀直覺的 Web UI，支援多使用者並發（`max_size=10`）
- **即時進度顯示**：逐步更新的進度條與狀態訊息
- **動態 UI 聯動**：切換語言時自動過濾可用模型與 Task 選項
- **一鍵複製 / 下載**：SRT 內容可直接複製到剪貼簿或下載檔案
- **範例音檔**：內建客語與台語示範音檔，一鍵體驗
- **PDF 文件服務**：`/terms-and-privacy` 端點直接讀取 `docs/Terms_and_Privacy.pdf`
- **密碼保護**：設定 `GRADIO_PASSWORD` 即可啟用 Basic Auth

---

## 📊 性能表現

| 音訊長度 | 單 GPU | 4 × GPU | 提升 |
|---------|--------|---------|------|
| 10 分鐘 | ~60 秒 | ~23 秒 | 2.6x |
| 30 分鐘 | ~180 秒 | ~67 秒 | 2.7x |
| 60 分鐘 | ~360 秒 | ~136 秒 | 2.6x |

> 以上數據基於 RTX 2080 Ti × 4、whisper-large-v3-turbo、float16。

---

## 🚀 快速開始

### 1. 安裝必要工具

```bash
# Docker & Docker Compose
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-plugin

# NVIDIA Container Toolkit
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list \
  | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 2. 設定環境變數

```bash
cp .env.example .env
# 編輯 .env，填入 HuggingFace token 與各語言模型 ID
```

### 3. 啟動服務

```bash
# 基本模式（不含 LLM 客語翻譯）
docker compose build
docker compose up -d

# 完整模式（含 Ollama LLM，用於客語 → 國語翻譯）
docker compose --profile llm up -d

# 查看日誌
docker compose logs -f
```

### 4. 存取服務

開啟瀏覽器訪問：`http://your-server-ip:7860`

---

## ⚙️ 環境變數

所有設定均透過 `.env` 檔案管理，複製 `.env.example` 後填入實際值。

### 基本設定

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `HUGGING_FACE_HUB_TOKEN` | — | HuggingFace token（存取私有或受保護模型） |
| `WHISPER_MODEL` | `large-v3-turbo` | 容器啟動時的預設模型 |
| `WHISPER_COMPUTE_TYPE` | `float16` | 推理精度（`float16` / `int8` / `float32`） |
| `WHISPER_DEVICE` | `cuda` | 運算裝置（`cuda` / `cpu`） |
| `CUDA_VISIBLE_DEVICES` | `0,1` | 分配給 Whisper 服務的 GPU 編號 |
| `SINGLE_GPU_DEVICES` | _(同上)_ | 覆寫 TranscriberPool 的 GPU 清單 |
| `GRADIO_SERVER_NAME` | `0.0.0.0` | 伺服器監聽位址 |
| `GRADIO_SERVER_PORT` | `7860` | 伺服器埠號 |
| `GRADIO_PASSWORD` | _(空)_ | 設定後啟用 Basic Auth（帳號為 `admin`） |
| `PRELOAD_MODEL` | `false` | 啟動時預載預設模型 |
| `WHISPER_NO_SPEECH_THRESHOLD` | `0.8` | 幻覺過濾的無語音機率門檻 |

### LLM 客語翻譯

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `ENABLE_LLM` | `false` | 啟用客語 → 國語 LLM 翻譯 |
| `OLLAMA_HOST` | `http://ollama:11434` | Ollama 服務位址 |
| `OLLAMA_MODEL` | `qwen2.5:7b` | 使用的 Ollama 模型 |
| `OLLAMA_BATCH_SIZE` | `5` | 每次 LLM 呼叫的字幕行數 |
| `OLLAMA_TIMEOUT` | `300` | LLM API 逾時秒數 |
| `HAKKA_LEXICON_PATH` | `lexicon/hakka_to_mandarin.csv` | 客語詞彙表路徑 |

### 私有模型 ID

語言專屬模型的 HuggingFace repo ID 以環境變數管理，不寫入程式碼，不提交至版本控制。留空時，對應語言選項不會出現在 UI 中。

| 變數 | 說明 |
|------|------|
| `HAKKA_V2_MODEL` | 客語 v2 模型的 HuggingFace repo ID |
| `HAKKA_V3_MODEL` | 客語 v3 模型的 HuggingFace repo ID |
| `TAIGI_MODEL` | 台語模型的 HuggingFace repo ID |

---

## 🗂️ 可用模型

| 模型 | 語言 | Task | VRAM | 速度 |
|------|------|------|------|------|
| whisper-large-v3-turbo | Auto / Mandarin / English | Transcribe | ~6 GB | ⚡ 快 |
| whisper-large-v3 | Auto / Mandarin / English | Transcribe + Translate | ~10 GB | 普通 |
| 客語模型 v2 | 客語輸入 → Mandarin 輸出 | Transcribe | ~10 GB | 普通 |
| 客語模型 v3 | 客語輸入 → Mandarin 輸出 | Transcribe | ~10 GB | 普通 |
| 台語模型 | 台語輸入 → Mandarin 輸出 | Transcribe | ~10 GB | 普通 |

> whisper-large-v3-turbo 與所有語言專屬模型僅支援 Transcribe；whisper-large-v3 額外支援 Translate to English。

---

## 🖥️ 系統需求

### 最低需求
- **作業系統**：Ubuntu 22.04 / 24.04
- **Docker**：Docker Engine 20.10+ & Docker Compose v2
- **GPU**：NVIDIA GPU（CUDA 12.x），建議 11 GB VRAM 以上
- **磁碟空間**：至少 30 GB（含模型快取）

### 推薦配置（多 GPU + LLM 翻譯）
- **GPU**：RTX 2080 Ti × 4
  - GPU 0–1：Whisper ASR（`CUDA_VISIBLE_DEVICES=0,1`）
  - GPU 2–3：Ollama LLM（`--profile llm`，`device_ids: ['2','3']`）
- **RAM**：32 GB 以上
- **CPU**：8 核心以上

---

## 🔌 API 使用

服務透過 Gradio 提供 REST API，可用 `gradio_client` 呼叫：

```python
from gradio_client import Client

client = Client("http://your-server-ip:7860")

status, asr_srt, asr_file, _, translated_srt, translated_file = client.predict(
    audio_file="/path/to/audio.wav",  # 支援 wav / mp3 / m4a / aac 等格式
    youtube_url="",
    model_size="large-v3-turbo",
    language="auto",            # auto | zh | en | hakka | taigi
    task="transcribe",          # transcribe | translate
    use_vad=True,
    min_silence_duration_s=0.2,
    merge_subtitles=True,
    convert_to_traditional=True,
    max_chars=80,
    translate_hakka=False,
    llm_system_prompt="",
    use_lexicon=False,
    use_enhancement=False,
    enhancement_mix=1.0,
    api_name="/process_audio",
)

print(asr_srt)
```

**輸出說明（6 個回傳值）：**

| 索引 | 說明 |
|------|------|
| `status` | 處理狀態 HTML（進度或完成資訊） |
| `asr_srt` | ASR 原文 SRT 字串 |
| `asr_file` | ASR 原文 SRT 檔案路徑 |
| `_` | 翻譯欄位的 UI 更新（內部用） |
| `translated_srt` | LLM 翻譯 SRT 字串（未啟用翻譯時為空） |
| `translated_file` | LLM 翻譯 SRT 檔案路徑（未啟用翻譯時為 None） |

---

## 📁 專案結構

```
whisper-for-subs/
├── app.py                     # Gradio Web 介面、FastAPI 路由、TranscriberPool
├── transcriber.py             # WhisperTranscriber：單 GPU 推理、模型轉換、幻覺過濾
├── vad.py                     # Silero VAD 語音活動偵測
├── speech_enhancer.py         # DeepFilterNet3 語音降噪增強
├── hakka_translator.py        # 客語 → 繁體中文 LLM 翻譯（Ollama）、詞彙表輔助
├── srt_utils.py               # SRT 格式生成、解析、字幕合併
├── chinese_converter.py       # 簡繁轉換（OpenCC s2tw）
├── youtube_downloader.py      # YouTube 音訊下載（yt-dlp）
├── preload_deepfilter.py      # Docker 建置期預下載 DeepFilterNet3 模型
├── requirements.txt           # Python 依賴套件
├── Dockerfile                 # Docker 映像檔（CUDA 12.4 + Python 3.11）
├── docker-compose.yml         # Docker Compose（含可選 llm profile）
├── .env.example               # 環境變數範本
├── docs/
│   └── Terms_and_Privacy.pdf # 使用者條款與隱私權政策
├── lexicon/
│   └── hakka_to_mandarin.csv # 客語 → 華語詞彙對照表
└── samples/                   # UI 示範音檔（客語 × 2、台語 × 1）
```

---

## 🏗️ 架構說明

### 音訊處理流程

```
輸入（上傳檔案 / YouTube URL）
  │
  ├─ 格式轉換（MP3 / AAC / M4A 等 → WAV，使用 ffmpeg）
  │
  ├─ [可選] DeepFilterNet3 語音增強（48 kHz 處理後降回 16 kHz）
  │
  ├─ Silero VAD 分割語音段落
  │
  ├─ WhisperTranscriber 逐段轉錄
  │
  ├─ 三層幻覺過濾
  │     ├─ filter_repetition_loops：移除重複短 token（如「好好好」）
  │     ├─ filter_short_token_bursts：移除連續極短片段（如「一。二。三。」）
  │     └─ filter_hallucinations：移除尾端無語音片段
  │
  ├─ [可選] OpenCC 簡繁轉換（s2tw 台灣標準）
  │
  ├─ [可選] Ollama LLM 客語 → 繁體中文批次翻譯
  │
  └─ SRT 合併輸出（ASR 原文 + 翻譯文各一份）
```

### TranscriberPool 排程邏輯

`app.py` 中的 `TranscriberPool` 實現跨 GPU 的多使用者並發調度：

1. 啟動時從 `CUDA_VISIBLE_DEVICES`（或 `SINGLE_GPU_DEVICES`）自動建立 GPU 清單
2. 每個請求進入時，優先選取**已快取該模型且負載最低**的 GPU
3. 若無快取，選取**負載最低**的 GPU 並載入模型（替換舊快取）
4. 請求完成後釋放計數，讓其他請求可立即使用該 GPU

### 客語翻譯管線

啟用 `ENABLE_LLM=true` 並以 `--profile llm` 啟動後：

1. 客語 ASR 結果批次送至 Ollama（預設每批 5 行，可調整 `OLLAMA_BATCH_SIZE`）
2. 詞彙表（`lexicon/hakka_to_mandarin.csv`）以 longest-match 方式注入 system prompt 作為翻譯提示
3. 若回傳行數不符，自動降回逐行翻譯（slow-path fallback）
4. 同時輸出 ASR 原文 SRT 與翻譯文 SRT 兩份檔案

---

## 🤝 貢獻

歡迎貢獻！請遵循以下步驟：

1. Fork 此專案
2. 建立功能分支（`git checkout -b feature/AmazingFeature`）
3. 提交變更（`git commit -m 'Add some AmazingFeature'`）
4. 推送到分支（`git push origin feature/AmazingFeature`）
5. 開啟 Pull Request

---

## 📄 授權

MIT License — 詳見 [LICENSE](LICENSE) 文件

---

## 👥 開發團隊

### Developers
- **[李鴻欣 Hung-Shin Lee](https://www.linkedin.com/in/hungshinlee)** — 聯和科創股份有限公司
- **[陳力瑋 Li-Wei Chen](mailto:wayne900619@gmail.com)** — 國立清華大學資訊工程學研究所

### Machine Providers (RTX 2080 Ti × 4)
- **[王新民 Hsin-Min Wang](https://homepage.iis.sinica.edu.tw/pages/whm/index_zh.html)** — 中央研究院資訊科學研究所
- **[廖沛俊 Pei-Jun Liao](mailto:newsboy3423@gmail.com)** — 中央研究院資訊科學研究所

---

## 🙏 致謝

- [OpenAI Whisper](https://github.com/openai/whisper) — 語音辨識基礎模型
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) — CTranslate2 高效推理引擎
- [Silero VAD](https://github.com/snakers4/silero-vad) — 語音活動偵測
- [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) — 語音降噪增強
- [Gradio](https://gradio.app/) — Web 介面框架
- [FastAPI](https://fastapi.tiangolo.com/) — 現代 Web 框架
- [Ollama](https://ollama.com/) — 本地 LLM 推理服務
- [OpenCC](https://github.com/BYVoid/OpenCC) — 中文簡繁轉換

---

## 📞 支援

- **Issues**：[GitHub Issues](https://github.com/hungshinlee/whisper-for-subs/issues)
- **Email**：hungshinlee@gmail.com

---

**© 2024–2026 FormoSTT Team. All rights reserved.**
