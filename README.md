# FormoSST: Speech-to-Text System for Taiwanese Languages 🎙️

**臺灣語音辨識暨翻譯系統**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![GPU](https://img.shields.io/badge/Multi--GPU-Supported-green.svg)](https://developer.nvidia.com/cuda-toolkit)

使用 OpenAI Whisper 模型的專業級自動語音辨識 (ASR) 服務，專為台灣語言優化，可將音檔、影片或 YouTube 影片轉換為高品質 SRT 字幕檔。

---

## ✨ 功能特色

### 🚀 核心功能
- **多種輸入方式**：上傳音檔、影片，或輸入 YouTube 網址
- **台灣語言優化**：支援國語（Mandarin）、英文（English）及自動偵測
- **雙重模式**：轉錄 (Transcribe) 或翻譯成英文 (Translate)
- **多模型支援**：
  - `large-v3-turbo` - 快速高效（僅支援 Transcribe）
  - `large-v3` - 高品質通用模型
  - `formospeech/whisper-large-v2-taiwanese-hakka-v1` - 台灣客語專用模型
- **標準輸出格式**：生成標準 SRT 字幕檔

### ⚡ 性能優化
- **多 GPU 並行處理**：4 張 GPU 同時運算，長音訊速度提升 **3.5 倍** 🔥
- **智能負載平衡**：短音訊（< 5 分鐘）使用單 GPU，長音訊自動啟用多 GPU
- **高速處理**：
  - 單 GPU 模式：~10x realtime
  - 多 GPU 模式：~26x realtime ⚡

### 🎯 智能功能
- **VAD 語音偵測**：使用 Silero VAD 精確偵測語音段落
- **可調整靈敏度**：自訂 VAD 最小靜音時長（0.01 - 2.0 秒）
- **自動合併字幕**：將過短的字幕合併成適當長度
- **繁體中文支持**：選擇中文時，自動將簡體轉換為繁體（台灣標準） 🇹🇼
- **模型智能限制**：
  - `large-v3-turbo` 自動限制為 Transcribe 模式
  - `formospeech` 模型自動限制為 Mandarin 語言

### 💻 介面功能
- **美觀的 Web UI**：使用 Gradio 框架，操作簡單直觀
- **即時進度顯示**：詳細的處理進度條和狀態訊息
- **一鍵複製**：直接複製 SRT 內容到剪貼簿 📋
- **PDF 文件查看**：內建使用者條款與隱私權政策文件

---

## 📊 性能表現

| 音訊長度 | 單 GPU | 多 GPU (4x) | 提升 |
|---------|--------|-------------|------|
| 10 分鐘 | 60 秒 | 23 秒 | 2.6x |
| 30 分鐘 | 180 秒 | 67 秒 | 2.7x |
| 60 分鐘 | 360 秒 | 136 秒 | 2.6x |

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
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 2. 部署服務

```bash
# 複製專案
git clone https://github.com/hungshinlee/whisper-for-subs.git
cd whisper-for-subs

# 建置並啟動
docker compose build
docker compose up -d

# 查看日誌
docker compose logs -f
```

### 3. 存取服務

開啟瀏覽器訪問：`http://your-server-ip:7860`

---

## ⚙️ 配置選項

### 環境變數 (docker-compose.yml)

```yaml
environment:
  - WHISPER_MODEL=large-v3-turbo        # 模型選擇
  - WHISPER_COMPUTE_TYPE=float16         # 精度：float16, int8, float32
  - CUDA_VISIBLE_DEVICES=0,1,2,3        # 使用的 GPU
  - GRADIO_SERVER_NAME=0.0.0.0          # 伺服器位址
  - GRADIO_SERVER_PORT=7860             # 伺服器埠號
```

### 可用模型

| 模型 | 語言支援 | Task 支援 | VRAM | 速度 | 推薦 |
|------|---------|----------|------|------|------|
| `large-v3-turbo` | Auto, Mandarin, English | Transcribe only | ~6 GB | 快 ⚡ | ✅ **推薦** |
| `large-v3` | Auto, Mandarin, English | Transcribe, Translate | ~10 GB | 較慢 | 高品質需求 |
| `formospeech/whisper-large-v2-taiwanese-hakka-v1` | Mandarin only | Transcribe, Translate | ~10 GB | 較慢 | 台灣客語 |

---

## ️ 系統需求

### 必需
- **作業系統**: Ubuntu 22.04 / 24.04
- **Docker**: Docker Engine 20.10+ & Docker Compose v2
- **GPU**: NVIDIA GPU（支援 CUDA 12.x）
  - 最低：GTX 1080 Ti (11GB VRAM)
  - 推薦：RTX 2080 Ti 或更高
- **磁碟空間**: 至少 30GB

### 推薦配置（多 GPU）
- **GPU**: 4x RTX 2080 Ti 或更高
- **RAM**: 32GB 或更多
- **CPU**: 8 核心或更多

---

## 🔌 API 使用

### Python 範例

```python
from gradio_client import Client

client = Client("http://your-server-ip:7860")

result = client.predict(
    audio_file="/path/to/audio.wav",
    youtube_url="",
    model_size="large-v3-turbo",
    language="auto",  # auto, zh, en
    task="transcribe",  # transcribe, translate
    use_vad=True,
    min_silence_duration_s=0.1,
    merge_subtitles=True,
    zh_conv=True,  # Convert to Traditional Chinese
    max_chars=80,
    use_multi_gpu=True,
    api_name="/process_audio"
)

status, srt_content, srt_file = result
print(srt_content)
```

---

## 📁 專案結構

```
whisper-for-subs/
├── app.py                      # Gradio Web 介面（FastAPI + Gradio）
├── transcriber.py              # 單 GPU 轉錄邏輯
├── parallel_transcriber.py     # 多 GPU 並行處理
├── vad.py                      # Silero VAD 語音檢測
├── youtube_downloader.py       # YouTube 下載
├── srt_utils.py                # SRT 格式處理
├── chinese_converter.py        # 簡繁轉換
├── requirements.txt            # Python 依賴
├── Dockerfile                  # Docker 映像檔
├── docker-compose.yml          # Docker Compose 配置
├── docs/                       # 政策文件
│   └── Terms_and_Privacy.pdf   # 使用者條款與隱私權政策
└── README.md                   # 本文件
```

---

## 🎨 主要改進

### v2.0 更新
- ✅ **FastAPI 整合**：使用 FastAPI 作為主應用，提供更好的擴展性
- ✅ **PDF 文件服務**：內建 Terms and Privacy Policy 文件查看
- ✅ **UI 優化**：
  - Language 改為 Radio 按鈕（Auto, Mandarin, English）
  - 模型特定限制（large-v3-turbo 只能 Transcribe，formospeech 只能 Mandarin）
  - 移除冗餘的系統信息顯示
- ✅ **代碼清理**：移除未使用的導入和變數
- ✅ **多用戶隔離**：Session-based 文件管理
- ✅ **Transcriber Pool**：防止多用戶間的干擾

---

## 🤝 貢獻

歡迎貢獻！請遵循以下步驟：

1. Fork 此專案
2. 創建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交變更 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 開啟 Pull Request

---

## 📄 授權

MIT License - 詳見 [LICENSE](LICENSE) 文件

---

## 👥 開發團隊

### Developers
- **[李鴻欣 Hung-Shin Lee](https://www.linkedin.com/in/hungshinlee)** - 聯和科創
- **[陳力瑋 Li-Wei Chen](mailto:wayne900619@gmail.com)** - 國立清華大學

### Contributors
- **[王新民 Hsin-Min Wang](https://homepage.iis.sinica.edu.tw/pages/whm/index_zh.html)** - 中央研究院資訊科學研究所

---

## 🙏 致謝

- [OpenAI Whisper](https://github.com/openai/whisper) - 語音辨識模型
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - 高效推理引擎
- [Silero VAD](https://github.com/snakers4/silero-vad) - 語音活動檢測
- [Gradio](https://gradio.app/) - Web 介面框架
- [FastAPI](https://fastapi.tiangolo.com/) - 現代 Web 框架
- [FormosaSpeech](https://huggingface.co/formospeech) - 台灣語言模型

---

## 📞 支援

- **Issues**: [GitHub Issues](https://github.com/hungshinlee/whisper-for-subs/issues)
- **Email**: hungshinlee@gmail.com

---

**© 2024-2026 FormoSST Team. All rights reserved.**
