# Whisper ASR 字幕生成服務 🎙️

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![GPU](https://img.shields.io/badge/Multi--GPU-Supported-green.svg)](https://developer.nvidia.com/cuda-toolkit)

[English](./README.en.md) | [更新日誌](./CHANGELOG.md)

使用 OpenAI Whisper 模型的專業級自動語音辨識 (ASR) 服務，可將音檔、影片或 YouTube 影片轉換為高品質 SRT 字幕檔。

**Source:** [王新民](https://homepage.iis.sinica.edu.tw/pages/whm/index_zh.html) 教授（中央研究院資訊科學研究所）

---

## ✨ 功能特色

### 🚀 核心功能
- **多種輸入方式**：上傳音檔、影片，或輸入 YouTube 網址
- **多語言支援**：支援中文、英文、日文、韓文等 18 種語言
- **雙重模式**：轉錄 (Transcribe) 或翻譯成英文 (Translate)
- **高品質模型**：使用 Whisper large-v3 和 large-v3-turbo 模型
- **標準輸出格式**：生成標準 SRT 字幕檔，可直接用於影片編輯

### ⚡ 性能優化
- **多 GPU 並行處理**：4 張 GPU 同時運算，長音訊速度提升 **3.5 倍** 🔥
- **持久化 Worker**：每個 GPU 只載入模型一次，避免重複載入浪費時間
- **智能負載平衡**：短音訊（< 5 分鐘）使用單 GPU，長音訊自動啟用多 GPU
- **高速處理**：
  - 單 GPU 模式：~10x realtime（1 小時音訊約 6 分鐘）
  - 多 GPU 模式：~26x realtime（1 小時音訊約 2.3 分鐘） ⚡

### 🎯 智能功能
- **VAD 語音偵測**：使用 Silero VAD 精確偵測語音段落
- **可調整靈敏度**：自訂 VAD 最小靜音時長（0.01 - 2.0 秒）
- **自動合併字幕**：將過短的字幕合併成適當長度（可設定每行最大字數）
- **繁體中文支持**：選擇中文時，自動將簡體轉換為繁體（台灣標準） 🇹🇼

### 💻 介面功能
- **美觀的 Web UI**：使用 Gradio 框架，操作簡單直觀
- **即時進度顯示**：詳細的處理進度條和狀態訊息
- **一鍵複製**：直接複製 SRT 內容到剪貼簿 📋
- **詳細日誌**：清楚顯示處理過程和性能統計

---

## 📊 性能表現

### 處理速度對比

| 音訊長度 | 單 GPU | 多 GPU (4x) | 提升 | 節省時間 |
|---------|--------|-------------|------|---------|
| 5 分鐘 | 30 秒 | 15 秒 | 2.0x | 15 秒 |
| 10 分鐘 | 60 秒 | 23 秒 | 2.6x | 37 秒 |
| 30 分鐘 | 180 秒 | 67 秒 | 2.7x | 113 秒 |
| 60 分鐘 | 360 秒 | 136 秒 | 2.6x | 224 秒 |

### 硬體配置（測試環境）

- **GPU**: 4x NVIDIA RTX 2080 Ti (11GB VRAM)
- **CPU**: Intel Xeon
- **RAM**: 64GB
- **模型**: Whisper large-v3-turbo

---

## 🎬 快速演示

### Web 介面

```
┌─────────────────────────────────────────────────┐
│  🎙️ Medical and Pharmaceutical ASR with Whisper │
├─────────────────────────────────────────────────┤
│  📥 Input                    📤 Output           │
│  ┌─────────────────┐        ┌─────────────────┐ │
│  │ Upload Audio    │        │ SRT Content     │ │
│  │ or Video        │        │                 │ │
│  └─────────────────┘        │ 1               │ │
│                              │ 00:00:00,000    │ │
│  OR                          │ --> 00:00:02,500│ │
│  ┌─────────────────┐        │ This is text.   │ │
│  │ YouTube URL     │        │                 │ │
│  └─────────────────┘        └─────────────────┘ │
│                              ┌─────────────────┐ │
│  ⚙️ Settings                 │ 📋 Copy         │ │
│  • Model: large-v3-turbo    └─────────────────┘ │
│  • Language: zh (Chinese)   ┌─────────────────┐ │
│  • Task: Transcribe         │ ⬇️ Download SRT │ │
│  ☑ Enable VAD               └─────────────────┘ │
│  • Min Silence: 0.1s                            │
│  ☑ Merge Subtitles                              │
│  ☑ Multi-GPU (for long audio)                   │
│  ┌─────────────────┐                            │
│  │  🚀 Start       │                            │
│  └─────────────────┘                            │
└─────────────────────────────────────────────────┘
```

### 處理流程

```
音訊輸入
   ↓
VAD 語音檢測（可調整靈敏度）
   ↓
段落優化
   ↓
[單 GPU]           [多 GPU 4x]
GPU 0 處理全部    GPU 0│1│2│3 並行處理
   ↓                    ↓
Whisper 轉錄        合併結果
   ↓                    ↓
中文簡繁轉換（如選擇 zh）
   ↓
合併短字幕（可選）
   ↓
生成 SRT 字幕
```

---

## 🛠️ 系統需求

### 必需
- **作業系統**: Ubuntu 22.04 / 24.04（推薦）
- **Docker**: Docker Engine 20.10+ & Docker Compose v2
- **GPU**: NVIDIA GPU（支援 CUDA 12.x）
  - 最低：GTX 1080 Ti (11GB VRAM)
  - 推薦：RTX 2080 Ti 或更高
- **NVIDIA Container Toolkit**: 用於 Docker GPU 支援
- **磁碟空間**: 至少 30GB（用於模型和暫存檔）

### 推薦配置
- **GPU**: 4x RTX 2080 Ti 或更高（多 GPU 模式）
- **RAM**: 32GB 或更多
- **CPU**: 8 核心或更多
- **網路**: 穩定的網路連線（用於 YouTube 下載）

---

## 🚀 快速開始

### 1. 安裝 Docker

```bash
# 更新套件索引
sudo apt-get update

# 安裝必要套件
sudo apt-get install -y ca-certificates curl

# 添加 Docker 官方 GPG 金鑰
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# 添加 Docker 套件庫
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 安裝 Docker
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 將目前使用者加入 docker 群組（可免去 sudo）
sudo usermod -aG docker $USER
newgrp docker

# 驗證安裝
docker --version
docker compose version
```

### 2. 安裝 NVIDIA Container Toolkit

```bash
# 添加 NVIDIA GPG 金鑰和套件庫
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 安裝
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 設定 Docker 使用 NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 驗證安裝
sudo docker run --rm --gpus all nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 nvidia-smi
```

### 3. 部署服務

```bash
# 複製專案
git clone https://github.com/hungshinlee/whisper-for-subs.git
cd whisper-for-subs

# 建置 Docker 映像
docker compose build

# 啟動服務
docker compose up -d

# 查看啟動日誌
docker compose logs -f
```

### 4. 存取服務

開啟瀏覽器訪問：`http://your-server-ip:7860`

**預設 Port**: 7860（可在 `docker-compose.yml` 中修改）

---

## ⚙️ 配置選項

### 環境變數

在 `docker-compose.yml` 中可配置以下環境變數：

```yaml
environment:
  - WHISPER_MODEL=large-v3-turbo        # 模型大小
  - WHISPER_DEVICE=cuda                  # 運算設備
  - WHISPER_COMPUTE_TYPE=float16         # 計算精度
  - CUDA_VISIBLE_DEVICES=0,1,2,3        # 可用的 GPU
  - PRELOAD_MODEL=false                  # 啟動時預載模型
  - GRADIO_SERVER_NAME=0.0.0.0          # 監聽地址
  - GRADIO_SERVER_PORT=7860             # 監聽 Port
```

### 可用模型

| 模型 | VRAM | 速度 | 品質 | 推薦 |
|------|------|------|------|------|
| `large-v3-turbo` | ~6 GB | 快 ⚡ | 優秀 | ✅ **推薦** |
| `large-v3` | ~10 GB | 較慢 | 最佳 | 高品質需求 |
| `large-v2` | ~10 GB | 較慢 | 優秀 | 向下相容 |

**注意**: `large-v3-turbo` 僅支援 "transcribe" 模式，不支援 "translate"。

### GPU 配置

#### 單 GPU（預設）
```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0  # 只使用 GPU 0
```

#### 多 GPU（推薦）
```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0,1,2,3  # 使用 4 張 GPU
```

---

## 📖 使用指南

### 基本使用

#### 1. 上傳音檔或影片

**支援格式**：
- 音訊：`.wav`、`.mp3`、`.m4a`、`.flac`、`.ogg`、`.aac`
- 影片：`.mp4`、`.mkv`、`.webm`、`.avi`、`.mov`

**步驟**：
1. 點擊「Upload Audio or Video」區域
2. 選擇檔案
3. 等待上傳完成

#### 2. 使用 YouTube 網址

**支援格式**：
- 標準：`https://www.youtube.com/watch?v=VIDEO_ID`
- 短網址：`https://youtu.be/VIDEO_ID`
- Shorts：`https://www.youtube.com/shorts/VIDEO_ID`

**步驟**：
1. 複製 YouTube 影片網址
2. 貼到「YouTube URL」欄位
3. 系統會自動下載音訊

### 進階設定

#### Model Size（模型大小）
- **large-v3-turbo**：快速且高品質（推薦） ⭐
- **large-v3**：最高品質（較慢）
- **large-v2**：向下相容

#### Language（語言）
- **auto**：自動偵測（推薦）
- **zh**：中文（會自動轉換為繁體） 🇹🇼
- **en**：英文
- **其他**：日文、韓文、西班牙文等 18 種語言

#### Task（任務）
- **Transcribe**：轉錄成原始語言
- **Translate to English**：翻譯成英文（僅 large-v3 支援）

#### Enable VAD（語音活動檢測）
- **啟用**：使用 Silero VAD 精確切分語音段落（推薦） ✅
- **停用**：使用 Whisper 內建 VAD

#### VAD: Min Silence Duration（最小靜音時長）
控制 VAD 切分的靈敏度：
- **0.03 - 0.08 秒**：快速對話、辯論（更多段落）
- **0.08 - 0.15 秒**：一般對話、訪談（預設：0.1） ⭐
- **0.15 - 0.3 秒**：演講、獨白
- **0.3 - 0.8 秒**：有聲書、朗讀（較少段落）

#### Merge Short Subtitles（合併短字幕）
- **啟用**：自動合併過短的字幕（推薦） ✅
- **停用**：保持原始切分

#### Max Characters Per Line（每行最大字數）
設定每行字幕的最大字數（40 - 120 字元）：
- **40 - 60**：適合手機觀看
- **70 - 80**：標準（預設：80） ⭐
- **90 - 120**：電腦觀看

#### Use Multi-GPU Parallel Processing（多 GPU 並行）
- **啟用**：自動在音訊 ≥ 5 分鐘時使用多 GPU（推薦） ✅
- **停用**：始終使用單 GPU

---

## 🚀 多 GPU 並行處理

### 工作原理

```
音訊檔案（60 分鐘）
        ↓
   VAD 語音檢測
        ↓
   切分成 89 個段落
        ↓
    優化分配
        ↓
   ┌─────┬─────┬─────┬─────┐
   │GPU 0│GPU 1│GPU 2│GPU 3│
   │ 22  │ 23  │ 22  │ 22  │ ← 段落數
   │segs │segs │segs │segs │
   └─────┴─────┴─────┴─────┘
        ↓     ↓     ↓     ↓
   並行轉錄（同時進行）
        ↓
    合併結果
        ↓
   完整字幕（2.3 分鐘完成）
```

### 效能對比

| 項目 | 單 GPU | 多 GPU (4x) |
|-----|--------|-------------|
| 模型載入 | 1 次 | 4 次（一次性） |
| 處理方式 | 順序 | 並行 |
| 10 分鐘音訊 | 60 秒 | 23 秒 |
| 60 分鐘音訊 | 360 秒 | 136 秒 |
| 速度比 | 10x | 26x |

### 何時使用多 GPU？

- ✅ **音訊 ≥ 5 分鐘**：顯著提升速度
- ✅ **有 2 張以上 GPU**：充分利用資源
- ✅ **需要快速處理**：節省時間

**系統會自動判斷**：勾選多 GPU 選項後，音訊小於 5 分鐘仍會使用單 GPU（避免不必要的開銷）。

---

## 🇹🇼 中文簡繁轉換

### 自動轉換

當選擇語言為 **Chinese (zh)** 時，系統會自動：
1. 使用 Whisper 轉錄（輸出簡體中文）
2. 使用 OpenCC 轉換為繁體中文（台灣標準）
3. 生成繁體中文字幕

### 轉換範例

```
簡體：这是语音识别系统
繁體：這是語音識別系統

簡體：使用计算机进行数据处理
繁體：使用電腦進行資料處理

簡體：患者需要进行血液检查和核磁共振成像
繁體：患者需要進行血液檢查和核磁共振造影
```

### 技術細節

- 使用 **OpenCC (Open Chinese Convert)**
- 轉換標準：`s2tw`（Simplified to Traditional Taiwan）
- 高準確度的詞彙對應
- 支援專業術語轉換

---

## 📋 複製到剪貼簿

### 使用方法

1. 轉錄完成後，在 SRT 輸出區域
2. 點擊「📋 Copy to Clipboard」按鈕
3. 看到「✅ Copied to clipboard!」提示
4. 在任何地方按 `Ctrl+V` (或 `Cmd+V`) 貼上

### 瀏覽器支援

- ✅ Chrome 66+
- ✅ Edge 79+
- ✅ Firefox 63+
- ✅ Safari 13.1+

---

## 📊 日誌和監控

### 查看即時日誌

```bash
# 查看所有日誌
docker compose logs -f

# 只看最新 50 行
docker compose logs -f --tail=50

# 搜尋特定內容
docker logs whisper-for-subs | grep "GPU"
```

### 日誌範例

#### 單 GPU 模式

```
🎯 Single-GPU mode: Using GPU 0
Loading Whisper model: large-v3-turbo on cuda
✅ Model loaded successfully
Loading Silero VAD (min_silence_duration=100ms)...
✅ VAD loaded successfully
📊 Audio loaded: 180.5s (2888000 samples @ 16000Hz)
🎯 VAD detected 12 speech segments
[GPU 0] ▶ Processing chunk 1/12 (18.3s)
[GPU 0] ✓ Chunk 1 complete: 8 text segments
[GPU 0] ▶ Processing chunk 2/12 (15.7s)
[GPU 0] ✓ Chunk 2 complete: 12 text segments
...
✅ Transcription complete!
   Device: GPU 0
   Segments: 127
   Duration: 180.5s
   Time: 18.3s
   Speed: 9.9x realtime
```

#### 多 GPU 模式

```
Initialized ParallelWhisperTranscriber with 4 GPUs: [0, 1, 2, 3]
Using multiprocessing start method: spawn
💡 Using persistent workers (models loaded once per GPU)
Loading Silero VAD (min_silence_duration=100ms)...
📊 Audio loaded: 600.0s (9600000 samples @ 16000Hz)
🎯 VAD detected 245 speech segments
✂️  Optimized to 89 segments for 4 GPUs
🚀 Starting parallel transcription with 4 persistent workers...

[GPU 0] 🔧 Initializing worker with model large-v3-turbo...
[GPU 1] 🔧 Initializing worker with model large-v3-turbo...
[GPU 2] 🔧 Initializing worker with model large-v3-turbo...
[GPU 3] 🔧 Initializing worker with model large-v3-turbo...
✅ Model loaded successfully
[GPU 0] ✅ Worker initialized and ready
✅ Model loaded successfully
[GPU 1] ✅ Worker initialized and ready
✅ Model loaded successfully
[GPU 2] ✅ Worker initialized and ready
✅ Model loaded successfully
[GPU 3] ✅ Worker initialized and ready

[GPU 0] ▶ Processing segment 0 (42.1s)
[GPU 1] ▶ Processing segment 1 (18.3s)
[GPU 2] ▶ Processing segment 2 (25.7s)
[GPU 3] ▶ Processing segment 3 (31.2s)
[GPU 1] ✓ Segment 1 complete: 12 text segments
[GPU 1] ▶ Processing segment 5 (22.4s)
[GPU 2] ✓ Segment 2 complete: 18 text segments
[GPU 2] ▶ Processing segment 6 (19.8s)
...

🔄 Converting to Traditional Chinese...
✅ Converted to Traditional Chinese
✅ Complete! 1247 text segments | Speed: 26.5x realtime | Time: 136s
```

### GPU 使用監控

```bash
# 即時監控 GPU 使用
watch -n 1 nvidia-smi

# 或使用 gpustat
pip install gpustat
watch -n 1 gpustat
```

---

## 🛠️ 維護和故障排除

### 自動清理

服務會自動清理超過 24 小時的暫存檔案：
- YouTube 下載的音檔（`/tmp/whisper-downloads`）
- 產生的 SRT 檔案（`/app/outputs`）

### 手動清理

```bash
# 清理暫存檔
docker exec whisper-for-subs rm -rf /tmp/whisper-downloads/*

# 清理輸出檔案
docker exec whisper-for-subs rm -rf /app/outputs/*

# 清理 Gradio 快取
docker exec whisper-for-subs rm -rf /tmp/gradio/*

# 檢查磁碟使用量
docker exec whisper-for-subs df -h
```

### 排程清理（Cron）

```bash
# 編輯 crontab
crontab -e

# 每天凌晨 3 點清理超過 1 天的檔案
0 3 * * * docker exec whisper-for-subs find /tmp/whisper-downloads -mtime +1 -delete 2>/dev/null
0 3 * * * docker exec whisper-for-subs find /app/outputs -name "*.srt" -mtime +1 -delete 2>/dev/null
```

### 常見問題

#### 1. GPU 無法使用

```bash
# 確認 NVIDIA 驅動
nvidia-smi

# 確認 Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 nvidia-smi

# 檢查 Docker GPU 支援
docker info | grep -i runtime
```

#### 2. 記憶體不足 (OOM)

**症狀**：
```
RuntimeError: CUDA out of memory
```

**解決方法**：
1. 使用較小的模型：`large-v3-turbo` 或 `medium`
2. 降低計算精度：`int8`
3. 減少可用 GPU 數量
4. 增加 GPU 記憶體

```yaml
environment:
  - WHISPER_MODEL=medium  # 較小的模型
  - WHISPER_COMPUTE_TYPE=int8  # 降低精度
  - CUDA_VISIBLE_DEVICES=0,1  # 只用 2 張 GPU
```

#### 3. YouTube 下載失敗

**可能原因**：
- 網路連線問題
- 影片有地區限制
- 影片已被移除
- yt-dlp 版本過舊

**解決方法**：
```bash
# 更新 yt-dlp
docker exec whisper-for-subs pip install -U yt-dlp

# 檢查影片是否可存取
yt-dlp -F "https://www.youtube.com/watch?v=VIDEO_ID"

# 重啟容器
docker compose restart
```

#### 4. Port 衝突

**症狀**：
```
Error starting userland proxy: listen tcp4 0.0.0.0:7860: bind: address already in use
```

**解決方法**：

方法 1：修改 Port
```yaml
# docker-compose.yml
ports:
  - "8080:7860"  # 改用 8080
```

方法 2：停止佔用的服務
```bash
# 找出佔用的服務
sudo lsof -i :7860

# 停止該服務
sudo systemctl stop <service-name>
```

#### 5. 模型下載慢

**症狀**：首次啟動時下載模型很慢

**解決方法**：
```bash
# 預先下載模型
docker exec whisper-for-subs python -c "
from faster_whisper import WhisperModel
model = WhisperModel('large-v3-turbo', device='cpu')
print('Model downloaded')
"
```

#### 6. 中文簡繁轉換不工作

**檢查**：
```bash
# 驗證 OpenCC 安裝
docker exec whisper-for-subs python -c "from opencc import OpenCC; print('✅ OpenCC installed')"

# 測試轉換
docker exec whisper-for-subs python /app/chinese_converter.py
```

**解決**：
```bash
# 重新建置容器
docker compose build --no-cache
docker compose up -d
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
├── README.md                   # 說明文件（繁體中文）
├── README.en.md                # 說明文件（英文）
├── CHANGELOG.md                # 更新日誌
├── LICENSE                     # MIT 授權
└── tmp/                        # 臨時文件（開發用）
    ├── CUDA_FIX.md
    ├── PERFORMANCE_OPTIMIZATION.md
    ├── CHINESE_CONVERSION.md
    ├── COPY_BUTTON.md
    ├── VAD_MIN_SILENCE_SETTING.md
    └── SESSION_SUMMARY.md
```

---

## 🔌 API 使用

Gradio 提供自動生成的 REST API：

### Python 範例

```python
from gradio_client import Client

# 連接到服務
client = Client("http://your-server-ip:7860")

# 轉錄音檔
result = client.predict(
    audio_file="/path/to/audio.wav",  # 音檔路徑
    youtube_url="",                    # YouTube URL（留空）
    model_size="large-v3-turbo",      # 模型大小
    language="zh",                     # 語言（中文）
    task="transcribe",                 # 任務
    use_vad=True,                      # 啟用 VAD
    min_silence_duration_s=0.1,       # VAD 靈敏度
    merge_subtitles=True,              # 合併字幕
    max_chars=80,                      # 每行最大字數
    use_multi_gpu=True,                # 多 GPU
    api_name="/process_audio"
)

# 解析結果
status, srt_content, srt_file_path = result
print(f"Status: {status}")
print(f"SRT Content:\n{srt_content}")
print(f"SRT File: {srt_file_path}")
```

### JavaScript 範例

```javascript
const response = await fetch("http://your-server-ip:7860/api/process_audio", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    data: [
      null,  // audio_file
      "https://www.youtube.com/watch?v=VIDEO_ID",  // youtube_url
      "large-v3-turbo",  // model_size
      "en",  // language
      "transcribe",  // task
      true,  // use_vad
      0.1,  // min_silence_duration_s
      true,  // merge_subtitles
      80,  // max_chars
      true,  // use_multi_gpu
    ]
  })
});

const result = await response.json();
console.log(result);
```

### cURL 範例

```bash
curl -X POST http://your-server-ip:7860/api/process_audio \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      null,
      "https://www.youtube.com/watch?v=VIDEO_ID",
      "large-v3-turbo",
      "en",
      "transcribe",
      true,
      0.1,
      true,
      80,
      true
    ]
  }'
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

### 核心技術
- [OpenAI Whisper](https://github.com/openai/whisper) - 語音辨識模型
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - 高效推理引擎
- [Silero VAD](https://github.com/snakers4/silero-vad) - 語音活動檢測
- [Gradio](https://gradio.app/) - Web 介面框架

### 輔助工具
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - YouTube 下載
- [OpenCC](https://github.com/BYVoid/OpenCC) - 中文簡繁轉換
- [FFmpeg](https://ffmpeg.org/) - 音訊處理

### 特別感謝
- [王新民](https://homepage.iis.sinica.edu.tw/pages/whm/index_zh.html) 教授 - 提供硬體支援
- [陳力瑋](https://github.com/txya900619) - 提供技術支援

---

## 📞 支援

- **Issues**: [GitHub Issues](https://github.com/hungshinlee/whisper-for-subs/issues)
- **Email**: 請透過 GitHub Issues 聯繫

---

## 🔄 更新日誌

詳見 [CHANGELOG.md](CHANGELOG.md)

### 最新版本 v3.0.0 (2025-01-05)

#### 🚀 重大更新
- **多 GPU 性能優化**：持久化 worker，模型只載入一次（2.7 倍提升）
- **中文簡繁轉換**：自動將簡體轉換為繁體（台灣標準）
- **複製按鈕**：一鍵複製 SRT 內容到剪貼簿
- **VAD 靈敏度設定**：可調整最小靜音時長（0.01 - 2.0 秒）
- **詳細日誌**：清楚顯示處理進度和統計信息

#### ⚡ 性能提升
- 10 分鐘音訊：122s → 46s
- 60 分鐘音訊：476s → 136s
- 速度比：7.6x → 26.5x realtime

#### 🎯 UI 改進
- 更美觀的進度條
- 即時狀態反饋
- 動態顯示/隱藏選項
- 更直觀的參數設定

---

## 作者

**李鴻欣 (Hung-Shin Lee)**  
聯和科創（United Link Co., Ltd.）  
hungshinlee@gmail.com
