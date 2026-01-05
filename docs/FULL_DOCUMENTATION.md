# Whisper ASR 完整使用文檔

## 目錄

1. [功能詳解](#功能詳解)
2. [使用指南](#使用指南)
3. [日誌和監控](#日誌和監控)
4. [維護和清理](#維護和清理)
5. [API 使用範例](#api-使用範例)

---

## 功能詳解

### 🎬 Web 介面

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

## 使用指南

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
...

[GPU 0] ▶ Processing segment 0 (42.1s)
[GPU 1] ▶ Processing segment 1 (18.3s)
[GPU 2] ▶ Processing segment 2 (25.7s)
[GPU 3] ▶ Processing segment 3 (31.2s)
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

## 🛠️ 維護和清理

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

---

## 🔌 API 使用範例

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

## 支援的語言

完整的支援語言列表：

| 代碼 | 語言 | 代碼 | 語言 |
|-----|------|------|------|
| `auto` | 自動偵測 | `en` | English |
| `zh` | Chinese (繁體) | `ja` | Japanese |
| `ko` | Korean | `es` | Spanish |
| `fr` | French | `de` | German |
| `it` | Italian | `pt` | Portuguese |
| `ru` | Russian | `ar` | Arabic |
| `hi` | Hindi | `th` | Thai |
| `vi` | Vietnamese | `id` | Indonesian |
| `ms` | Malay | `tl` | Filipino |

---

**最後更新**：2025-01-05
