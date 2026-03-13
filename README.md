# FormoSST: Speech-to-Text System for Taiwanese Languages 🎙️

**臺灣語音辨識暨翻譯系統**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![GPU](https://img.shields.io/badge/Multi--GPU-Supported-green.svg)](https://developer.nvidia.com/cuda-toolkit)

使用 OpenAI Whisper 模型的專業級自動語音辨識 (ASR) 服務，專為台灣語言優化，可將音檔、影片或 YouTube 影片轉換為高品質 SRT 字幕檔。

---

## ✨ 功能特色

### 🚀 核心功能
- **多種輸入方式**：上傳音檔／影片，或直接輸入 YouTube 網址
- **台灣多語言支援**：國語（Mandarin）、客語（Hakka）、台語（Taigi）、英文（English）及自動偵測
- **雙重模式**：轉錄（Transcribe）或翻譯成英文（Translate to English）
- **標準輸出格式**：產生標準 SRT 字幕檔，支援一鍵複製與下載

### ⚡ 多 GPU 智能調度
- **TranscriberPool**：自動偵測可用 GPU，以最少負載優先分配請求
- **模型快取**：同一 GPU 上的相同模型不重新載入，節省 VRAM 搬運時間
- **並發隔離**：不同使用者的請求分散至不同 GPU，避免相互干擾
- **CPU 備援**：無 GPU 環境時自動降級至 CPU 推理

### 🎯 智能音訊處理
- **Silero VAD**：精確偵測語音段落，剔除靜音區間，提升辨識準確率
- **DeepFilterNet3 降噪**：可選的語音增強，混合比例（mix factor）可自訂
- **可調整 VAD 靈敏度**：自訂最小靜音時長（0.01 – 2.0 秒）
- **幻覺過濾**：自動移除 Whisper 在靜音尾端產生的重複／無意義片段
- **自動合併字幕**：將過短字幕合併成適當長度，支援最大字元數限制

### 🌏 語言專屬功能
- **繁體中文**：選擇中文時自動以 OpenCC（s2tw）將簡體轉換為繁體（台灣標準）
- **客語翻譯**：透過本地 Ollama LLM（如 `qwen2.5:7b`）將客語辨識結果翻譯為繁體中文；支援詞彙表（lexicon）輔助提示
- **台語辨識**：使用 FormoAI Brecioso 台語模型，直接輸出繁體中文

### 💻 Web 介面
- **Gradio + FastAPI**：美觀直覺的 Web UI，支援多使用者並發
- **即時進度顯示**：逐步更新的進度條與狀態訊息
- **動態 UI 聯動**：切換語言／模型時自動調整可用的 Task 選項
- **範例音檔**：內建客語與台語示範音檔，一鍵體驗
- **PDF 文件服務**：內建使用者條款與隱私權政策文件（`/terms-and-privacy`）
- **密碼保護**：可透過 `GRADIO_PASSWORD` 環境變數啟用存取控管

---

## 📊 性能表現

| 音訊長度 | 單 GPU | 多 GPU (4x) | 提升 |
|---------|--------|-------------|------|
| 10 分鐘 | ~60 秒 | ~23 秒 | 2.6x |
| 30 分鐘 | ~180 秒 | ~67 秒 | 2.7x |
| 60 分鐘 | ~360 秒 | ~136 秒 | 2.6x |

> 以上數據基於 RTX 2080 Ti × 4、`large-v3-turbo`、`float16`。

---

## 🚀 快速開始

### 1. 安裝必要工具

```bash
# 安裝 Docker & Docker Compose
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-plugin

# 安裝 NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list \
  | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 2. 設定環境變數

```bash
cp .env.example .env
# 編輯 .env，填入 HuggingFace token 等設定
```

### 3. 部署服務

```bash
# 複製專案
git clone https://github.com/hungshinlee/whisper-for-subs.git
cd whisper-for-subs

# 建置並啟動（基本模式，不含 LLM）
docker compose build
docker compose up -d

# 啟動含 Ollama LLM 的完整模式（客語翻譯）
docker compose --profile llm up -d

# 查看日誌
docker compose logs -f
```

### 4. 存取服務

開啟瀏覽器訪問：`http://your-server-ip:7860`

---

## ⚙️ 配置選項

### 環境變數

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `WHISPER_MODEL` | `large-v3-turbo` | 預設載入的 Whisper 模型 |
| `WHISPER_COMPUTE_TYPE` | `float16` | 推理精度（`float16`、`int8`、`float32`） |
| `WHISPER_DEVICE` | `cuda` | 運算裝置（`cuda` 或 `cpu`） |
| `CUDA_VISIBLE_DEVICES` | `0,1` | 分配給 Whisper 服務的 GPU |
| `SINGLE_GPU_DEVICES` | _(同上)_ | 覆寫單 GPU 排程的 GPU 清單 |
| `GRADIO_SERVER_NAME` | `0.0.0.0` | 伺服器監聽位址 |
| `GRADIO_SERVER_PORT` | `7860` | 伺服器埠號 |
| `GRADIO_PASSWORD` | _(空)_ | 設定後啟用 Basic Auth 保護 |
| `ENABLE_LLM` | `false` | 啟用客語 → 國語 LLM 翻譯 |
| `OLLAMA_HOST` | `http://ollama:11434` | Ollama 服務位址 |
| `OLLAMA_MODEL` | `qwen2.5:7b` | 使用的 Ollama 模型 |
| `OLLAMA_BATCH_SIZE` | `5` | 每次 LLM 呼叫的行數 |
| `OLLAMA_TIMEOUT` | `300` | LLM API 逾時秒數 |
| `PRELOAD_MODEL` | `false` | 啟動時預載模型 |
| `HUGGING_FACE_HUB_TOKEN` | _(空)_ | HuggingFace token（存取受保護模型） |
| `HAKKA_LEXICON_PATH` | `lexicon/hakka_to_mandarin.csv` | 客語詞彙表路徑 |
| `WHISPER_NO_SPEECH_THRESHOLD` | `0.8` | 幻覺過濾的無語音機率門檻 |

### 可用模型

| 模型 | 語言支援 | Task | VRAM | 速度 | 用途 |
|------|---------|------|------|------|------|
| whisper-large-v3-turbo | Auto / Mandarin / English | Transcribe | ~6 GB | ⚡ 快 | **推薦，日常使用** |
| whisper-large-v3 | Auto / Mandarin / English | Transcribe + Translate | ~10 GB | 普通 | 高品質需求 |
| FormosaSpeech 客語 v2 | Mandarin（客語輸入） | Transcribe | ~10 GB | 普通 | 台灣客語 |
| FormosaSpeech 客語 v3 | Mandarin（客語輸入） | Transcribe | ~10 GB | 普通 | 台灣客語（v3） |
| FormoAI 台語模型 | Mandarin 輸出（台語輸入） | Transcribe | ~10 GB | 普通 | 台灣台語 |

> **注意**：whisper-large-v3-turbo 與客語／台語模型只支援 Transcribe；whisper-large-v3 額外支援 Translate to English。

---

## 🖥️ 系統需求

### 最低需求
- **作業系統**：Ubuntu 22.04 / 24.04
- **Docker**：Docker Engine 20.10+ & Docker Compose v2
- **GPU**：NVIDIA GPU（CUDA 12.x），建議 11 GB VRAM 以上（GTX 1080 Ti 起）
- **磁碟空間**：至少 30 GB（含模型快取）

### 推薦配置（多 GPU / LLM 翻譯）
- **GPU**：RTX 2080 Ti × 4（或更高）
  - GPU 0–1：Whisper ASR
  - GPU 2–3：Ollama LLM（啟用 `--profile llm` 時）
- **RAM**：32 GB 或以上
- **CPU**：8 核心或以上

---

## 🔌 API 使用

Gradio 提供標準 REST API，可透過 `gradio_client` 呼叫：

```python
from gradio_client import Client

client = Client("http://your-server-ip:7860")

status, asr_srt, asr_file, _, _, _ = client.predict(
    audio_file="/path/to/audio.wav",
    youtube_url="",
    model_size="large-v3-turbo",
    language="auto",           # auto | zh | en | hakka | taigi
    task="transcribe",         # transcribe | translate
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

---

## 📁 專案結構

```
whisper-for-subs/
├── app.py                      # Gradio Web 介面（FastAPI + Gradio）、TranscriberPool
├── transcriber.py              # WhisperTranscriber：單 GPU 轉錄、模型轉換、幻覺過濾
├── vad.py                      # Silero VAD 語音活動偵測
├── speech_enhancer.py          # DeepFilterNet3 語音降噪增強
├── hakka_translator.py         # 客語 → 繁體中文 LLM 翻譯（Ollama）、詞彙表輔助
├── srt_utils.py                # SRT 格式生成、解析、合併
├── chinese_converter.py        # 簡繁轉換（OpenCC s2tw）
├── youtube_downloader.py       # YouTube 音訊下載（yt-dlp）
├── preload_deepfilter.py       # Docker 建置期預下載 DeepFilterNet3 模型
├── requirements.txt            # Python 依賴套件
├── Dockerfile                  # Docker 映像檔（CUDA 12.4 + Python 3.11）
├── docker-compose.yml          # Docker Compose 配置（含可選 Ollama profile）
├── .env.example                # 環境變數範本
├── docs/
│   └── Terms_and_Privacy.pdf  # 使用者條款與隱私權政策
├── lexicon/
│   └── hakka_to_mandarin.csv  # 客語 → 華語詞彙對照表
├── samples/                    # UI 示範音檔（客語 × 2、台語 × 1）
└── README.md                   # 本文件
```

---

## 🏗️ 架構說明

### TranscriberPool（多使用者 GPU 排程）
`app.py` 內建的 `TranscriberPool` 負責跨 GPU 負載平衡：
- 從 `CUDA_VISIBLE_DEVICES`（或 `SINGLE_GPU_DEVICES`）自動偵測可用 GPU
- 以「最少負載優先」策略分配請求至 GPU
- 每個 GPU 快取一個 `WhisperTranscriber` 實例，避免重複載入模型
- 當請求的模型與快取不符時，才替換並重新載入

### 音訊處理流程
```
輸入（檔案 / YouTube）
    → 格式轉換（AAC/M4A → WAV，ffmpeg）
    → [可選] DeepFilterNet3 語音增強
    → Silero VAD 分割語音段落
    → WhisperTranscriber 轉錄各段落
    → 幻覺過濾（重複、無語音）
    → [可選] OpenCC 簡繁轉換
    → [可選] Ollama LLM 客語翻譯
    → SRT 合併與輸出
```

### 客語翻譯管線
啟用 `ENABLE_LLM=true` 並以 `--profile llm` 啟動時：
- 客語 ASR 結果批次送至 Ollama（預設每批 5 行）
- 詞彙表（`lexicon/hakka_to_mandarin.csv`）作為 system prompt 提示
- 行數不符時自動降回逐行翻譯
- 同時輸出 ASR 原文（SRT）與翻譯文（SRT）兩份檔案

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
- [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) — 語音降噪
- [Gradio](https://gradio.app/) — Web 介面框架
- [FastAPI](https://fastapi.tiangolo.com/) — 現代 Web 框架
- [Ollama](https://ollama.com/) — 本地 LLM 推理
- [FormosaSpeech](https://huggingface.co/formospeech) — 台灣客語模型
- [FormoAI](https://huggingface.co/formoai) — 台灣台語模型

---

## 📞 支援

- **Issues**：[GitHub Issues](https://github.com/hungshinlee/whisper-for-subs/issues)
- **Email**：hungshinlee@gmail.com

---

**© 2024–2026 FormoSST Team. All rights reserved.**
