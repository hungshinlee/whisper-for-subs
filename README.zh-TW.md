# Whisper ASR 字幕生成服務

[English](./README.md)

使用 OpenAI Whisper 模型的自動語音辨識 (ASR) 服務，可將音檔、影片或 YouTube 影片轉換為 SRT 字幕檔。

## 功能特色

- 🎙️ **多種輸入方式**：上傳音檔、影片，或輸入 YouTube 網址
- 🌍 **多語言支援**：支援中文、英文、日文等多種語言
- 🔄 **雙重模式**：轉錄 (Transcribe) 或翻譯成英文 (Translate)
- 🎯 **VAD 語音偵測**：使用 Silero VAD 精確偵測語音段落
- 📝 **SRT 輸出**：標準 SRT 格式，可直接用於影片字幕
- 🚀 **GPU 加速**：支援多 GPU 並行處理

## 系統需求

- Ubuntu Server 24.04
- Docker & Docker Compose
- NVIDIA GPU（建議 RTX 2080 Ti 或更高）
- NVIDIA Container Toolkit

## 快速開始

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

# 將目前使用者加入 docker 群組（選用，可免去 sudo）
sudo usermod -aG docker $USER
newgrp docker

# 驗證安裝
docker --version
docker compose version
```

### 2. 安裝 NVIDIA Container Toolkit

```bash
# 添加 NVIDIA GPG 金鑰
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# 添加 NVIDIA 套件庫
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 安裝
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 設定 Docker 使用 NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker

# 重啟 Docker
sudo systemctl restart docker

# 驗證安裝
sudo docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

### 3. 建置與啟動服務

```bash
# 複製專案
git clone https://github.com/hungshinlee/whisper-for-subs.git
cd whisper-for-subs

# 建置 Docker 映像
docker compose build

# 啟動服務
docker compose up -d

# 查看日誌
docker compose logs -f
```

### 4. 存取服務

開啟瀏覽器訪問：`http://your-server-ip`

## 配置選項

### 環境變數

在 `docker-compose.yml` 中可配置以下環境變數：

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `WHISPER_MODEL` | `large-v3` | Whisper 模型大小 |
| `WHISPER_DEVICE` | `cuda` | 運算設備 (`cuda` 或 `cpu`) |
| `WHISPER_COMPUTE_TYPE` | `float16` | 計算精度 (`float16`, `int8`, `float32`) |
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3` | 可用的 GPU 編號 |
| `PRELOAD_MODEL` | `false` | 啟動時預載模型 |

### 可用模型

| 模型 | VRAM 需求 | 速度 | 品質 |
|------|-----------|------|------|
| `tiny` | ~1 GB | 最快 | 一般 |
| `base` | ~1 GB | 很快 | 一般 |
| `small` | ~2 GB | 快 | 好 |
| `medium` | ~5 GB | 中等 | 很好 |
| `large-v2` | ~10 GB | 較慢 | 優秀 |
| `large-v3` | ~10 GB | 較慢 | 最佳 |
| `large-v3-turbo` | ~6 GB | 快 | 優秀 |

## 使用方式

### 上傳音檔或影片

1. 點擊「上傳音檔或影片」區域
2. 選擇音檔 (`.wav`, `.mp3`, `.m4a`, `.flac`) 或影片 (`.mp4`, `.mkv`, `.webm`)
3. 設定語言和轉錄模式
4. 點擊「開始轉錄」

### 使用 YouTube 網址

1. 在「YouTube 網址」欄位貼上影片連結
2. 支援格式：
   - `https://www.youtube.com/watch?v=VIDEO_ID`
   - `https://youtu.be/VIDEO_ID`
   - `https://www.youtube.com/shorts/VIDEO_ID`
3. 設定語言和轉錄模式
4. 點擊「開始轉錄」

### 設定選項

- **模型大小**：較大的模型品質較好但速度較慢
- **語言**：選擇「自動偵測」或指定語言
- **功能**：
  - 轉錄 (Transcribe)：輸出原始語言字幕
  - 翻譯 (Translate)：翻譯成英文字幕
- **VAD 語音偵測**：啟用可提高分段精確度
- **合併短字幕**：將過短的字幕合併成適當長度

## API 使用

Gradio 提供自動生成的 API，可透過 Python 呼叫：

```python
from gradio_client import Client

client = Client("http://your-server-ip")

# 上傳檔案轉錄
result = client.predict(
    audio_file="/path/to/audio.wav",
    youtube_url="",
    model_size="large-v3",
    language="auto",
    task="transcribe",
    use_vad=True,
    merge_subtitles=True,
    max_chars=80,
    api_name="/process_audio"
)

srt_content, srt_file_path, status = result
print(status)
print(srt_content)
```

## 目錄結構

```
whisper-for-subs/
├── app.py                 # Gradio 主程式
├── transcriber.py         # Whisper 轉錄邏輯
├── vad.py                 # Silero VAD 處理
├── youtube_downloader.py  # YouTube 下載
├── srt_utils.py           # SRT 格式處理
├── requirements.txt       # Python 依賴
├── Dockerfile             # Docker 映像檔
├── docker-compose.yml     # Docker Compose 配置
└── README.md              # 說明文件
```

## 故障排除

### GPU 無法使用

```bash
# 確認 NVIDIA 驅動
nvidia-smi

# 確認 Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

### 記憶體不足

- 使用較小的模型（如 `medium` 或 `small`）
- 設定 `WHISPER_COMPUTE_TYPE=int8` 減少 VRAM 使用

### YouTube 下載失敗

- 確認網路連線
- 更新 yt-dlp：`pip install -U yt-dlp`
- 檢查影片是否有地區限制

## 授權

MIT License

## 致謝

- [OpenAI Whisper](https://github.com/openai/whisper)
- [faster-whisper](https://github.com/guillaumekln/faster-whisper)
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp)
- [Gradio](https://gradio.app/)
