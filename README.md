# Whisper ASR 字幕生成服務 🎙️

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![GPU](https://img.shields.io/badge/Multi--GPU-Supported-green.svg)](https://developer.nvidia.com/cuda-toolkit)

使用 OpenAI Whisper 模型的專業級自動語音辨識 (ASR) 服務，可將音檔、影片或 YouTube 影片轉換為高品質 SRT 字幕檔。

[English](./docs/README.en.md) | [更新日誌](./docs/CHANGELOG.md)

---

## ✨ 功能特色

### 🚀 核心功能
- **多種輸入方式**：上傳音檔、影片，或輸入 YouTube 網址
- **多語言支援**：支援中文、英文、日文、韓文等 18 種語言
- **雙重模式**：轉錄 (Transcribe) 或翻譯成英文 (Translate)
- **高品質模型**：使用 Whisper large-v3 和 large-v3-turbo 模型
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

### 💻 介面功能
- **美觀的 Web UI**：使用 Gradio 框架，操作簡單直觀
- **即時進度顯示**：詳細的處理進度條和狀態訊息
- **一鍵複製**：直接複製 SRT 內容到剪貼簿 📋

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
  - WHISPER_MODEL=large-v3-turbo        # 模型：large-v3-turbo, large-v3, large-v2
  - WHISPER_COMPUTE_TYPE=float16         # 精度：float16, int8, float32
  - CUDA_VISIBLE_DEVICES=0,1,2,3        # 使用的 GPU
```

### 可用模型

| 模型 | VRAM | 速度 | 推薦 |
|------|------|------|------|
| `large-v3-turbo` | ~6 GB | 快 ⚡ | ✅ **推薦** |
| `large-v3` | ~10 GB | 較慢 | 高品質需求 |
| `large-v2` | ~10 GB | 較慢 | 向下相容 |

---

## 📖 詳細文檔

完整的使用指南和技術文檔請參考 [docs](./docs) 目錄：

- **[部署指南](./docs/DEPLOYMENT_GUIDE.md)** - 詳細的安裝和配置說明
- **[多 GPU 指南](./docs/MULTI_GPU_GUIDE.md)** - 多 GPU 並行處理詳解
- **[快速開始 (多 GPU)](./docs/QUICKSTART_MULTI_GPU.md)** - 快速設置多 GPU 環境
- **[故障排除](./docs/TROUBLESHOOTING_MULTI_GPU.md)** - 常見問題解決方案
- **[更新日誌](./docs/CHANGELOG.md)** - 版本更新記錄
- **[English Version](./docs/README.en.md)** - English documentation

---

## 🛠️ 系統需求

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
    language="zh",
    task="transcribe",
    use_vad=True,
    min_silence_duration_s=0.1,
    merge_subtitles=True,
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
├── app.py                      # Gradio Web 介面
├── transcriber.py              # 單 GPU 轉錄邏輯
├── parallel_transcriber.py     # 多 GPU 並行處理
├── vad.py                      # Silero VAD 語音檢測
├── youtube_downloader.py       # YouTube 下載
├── srt_utils.py                # SRT 格式處理
├── chinese_converter.py        # 簡繁轉換
├── requirements.txt            # Python 依賴
├── Dockerfile                  # Docker 映像檔
├── docker-compose.yml          # Docker Compose 配置
├── docs/                       # 詳細文檔
│   ├── DEPLOYMENT_GUIDE.md
│   ├── MULTI_GPU_GUIDE.md
│   ├── QUICKSTART_MULTI_GPU.md
│   ├── TROUBLESHOOTING_MULTI_GPU.md
│   ├── CHANGELOG.md
│   └── README.en.md
└── README.md                   # 本文件
```

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

## 🙏 致謝

- [OpenAI Whisper](https://github.com/openai/whisper) - 語音辨識模型
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - 高效推理引擎
- [Silero VAD](https://github.com/snakers4/silero-vad) - 語音活動檢測
- [Gradio](https://gradio.app/) - Web 介面框架
- [王新民](https://homepage.iis.sinica.edu.tw/pages/whm/index_zh.html) 教授 - 硬體支援

---

## 📞 支援

- **Issues**: [GitHub Issues](https://github.com/hungshinlee/whisper-for-subs/issues)
- **文檔**: [docs 目錄](./docs)

---

**作者**: 李鴻欣 (Hung-Shin Lee)  
**公司**: 聯和科創（United Link Co., Ltd.）  
**Email**: hungshinlee@gmail.com
