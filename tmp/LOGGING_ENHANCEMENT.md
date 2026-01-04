# 單 GPU 模式詳細日誌增強

## 🎯 問題

單 GPU 模式的日誌過於簡單，沒有顯示：
- 使用哪張 GPU
- 處理進度詳情
- 每個 chunk 的狀態
- 最終統計信息

## ✅ 已增強的日誌內容

### 1. 初始化階段

**修改前：**
```
Loading Whisper model: large-v3-turbo on cuda
Loading Silero VAD...
```

**修改後：**
```
🎯 Single-GPU mode: Using GPU 0
Loading Whisper model: large-v3-turbo on cuda
✅ Model loaded successfully
Loading Silero VAD...
✅ VAD loaded successfully
```

### 2. 音訊載入階段

**新增：**
```
📊 Audio loaded: 180.5s (2888000 samples @ 16000Hz)
```

### 3. VAD 檢測階段

**新增：**
```
🎯 VAD detected 12 speech segments
```

### 4. 處理階段（每個 chunk）

**新增：**
```
[GPU 0] ▶ Processing chunk 1/12 (18.3s)
[GPU 0] ✓ Chunk 1 complete: 8 text segments
[GPU 0] ▶ Processing chunk 2/12 (15.7s)
[GPU 0] ✓ Chunk 2 complete: 12 text segments
[GPU 0] ▶ Processing chunk 3/12 (22.1s)
[GPU 0] ✓ Chunk 3 complete: 15 text segments
...
```

### 5. 完成階段（統計信息）

**新增：**
```
✅ Transcription complete!
   Device: GPU 0
   Segments: 127
   Duration: 180.5s
   Time: 18.3s
   Speed: 9.9x realtime
```

---

## 📊 完整日誌範例

### 單 GPU 模式（取消勾選 Multi-GPU）

```bash
$ docker compose logs -f

whisper-for-subs  | 
whisper-for-subs  | ==========
whisper-for-subs  | == CUDA ==
whisper-for-subs  | ==========
whisper-for-subs  | 
whisper-for-subs  | CUDA Version 12.4.1
whisper-for-subs  | 
whisper-for-subs  | * Running on local URL:  http://0.0.0.0:7860
whisper-for-subs  | * To create a public link, set `share=True` in `launch()`.

# 用戶上傳音訊並點擊 Start

whisper-for-subs  | 🎯 Single-GPU mode: Using GPU 0
whisper-for-subs  | Loading Whisper model: large-v3-turbo on cuda
whisper-for-subs  | ✅ Model loaded successfully
whisper-for-subs  | Loading Silero VAD...
whisper-for-subs  | Using cache found in /root/.cache/torch/hub/snakers4_silero-vad_master
whisper-for-subs  | ✅ VAD loaded successfully
whisper-for-subs  | 📊 Audio loaded: 180.5s (2888000 samples @ 16000Hz)
whisper-for-subs  | 🎯 VAD detected 12 speech segments
whisper-for-subs  | [GPU 0] ▶ Processing chunk 1/12 (18.3s)
whisper-for-subs  | [GPU 0] ✓ Chunk 1 complete: 8 text segments
whisper-for-subs  | [GPU 0] ▶ Processing chunk 2/12 (15.7s)
whisper-for-subs  | [GPU 0] ✓ Chunk 2 complete: 12 text segments
whisper-for-subs  | [GPU 0] ▶ Processing chunk 3/12 (22.1s)
whisper-for-subs  | [GPU 0] ✓ Chunk 3 complete: 15 text segments
whisper-for-subs  | [GPU 0] ▶ Processing chunk 4/12 (11.2s)
whisper-for-subs  | [GPU 0] ✓ Chunk 4 complete: 9 text segments
whisper-for-subs  | [GPU 0] ▶ Processing chunk 5/12 (25.8s)
whisper-for-subs  | [GPU 0] ✓ Chunk 5 complete: 18 text segments
whisper-for-subs  | [GPU 0] ▶ Processing chunk 6/12 (14.3s)
whisper-for-subs  | [GPU 0] ✓ Chunk 6 complete: 11 text segments
whisper-for-subs  | [GPU 0] ▶ Processing chunk 7/12 (19.7s)
whisper-for-subs  | [GPU 0] ✓ Chunk 7 complete: 14 text segments
whisper-for-subs  | [GPU 0] ▶ Processing chunk 8/12 (16.9s)
whisper-for-subs  | [GPU 0] ✓ Chunk 8 complete: 10 text segments
whisper-for-subs  | [GPU 0] ▶ Processing chunk 9/12 (12.4s)
whisper-for-subs  | [GPU 0] ✓ Chunk 9 complete: 8 text segments
whisper-for-subs  | [GPU 0] ▶ Processing chunk 10/12 (21.5s)
whisper-for-subs  | [GPU 0] ✓ Chunk 10 complete: 16 text segments
whisper-for-subs  | [GPU 0] ▶ Processing chunk 11/12 (8.7s)
whisper-for-subs  | [GPU 0] ✓ Chunk 11 complete: 5 text segments
whisper-for-subs  | [GPU 0] ▶ Processing chunk 12/12 (13.0s)
whisper-for-subs  | [GPU 0] ✓ Chunk 12 complete: 11 text segments
whisper-for-subs  | ✅ Transcription complete!
whisper-for-subs  |    Device: GPU 0
whisper-for-subs  |    Segments: 127
whisper-for-subs  |    Duration: 180.5s
whisper-for-subs  |    Time: 18.3s
whisper-for-subs  |    Speed: 9.9x realtime
```

### 多 GPU 模式（勾選 Multi-GPU）

```bash
whisper-for-subs  | Initialized ParallelWhisperTranscriber with 4 GPUs: [0, 1, 2, 3]
whisper-for-subs  | Using multiprocessing start method: spawn
whisper-for-subs  | 📊 Audio loaded: 1800.0s (28800000 samples @ 16000Hz)
whisper-for-subs  | 🎯 VAD detected 245 speech segments
whisper-for-subs  | ✂️  Optimized to 89 segments for 4 GPUs
whisper-for-subs  | 🚀 Starting parallel transcription on 4 GPUs...
whisper-for-subs  | [GPU 0] ▶ Processing segment 0 (42.1s)
whisper-for-subs  | [GPU 1] ▶ Processing segment 1 (18.3s)
whisper-for-subs  | [GPU 2] ▶ Processing segment 2 (25.7s)
whisper-for-subs  | [GPU 3] ▶ Processing segment 3 (31.2s)
whisper-for-subs  | Loading Whisper model: large-v3-turbo on cuda
whisper-for-subs  | Loading Whisper model: large-v3-turbo on cuda
whisper-for-subs  | Loading Whisper model: large-v3-turbo on cuda
whisper-for-subs  | Loading Whisper model: large-v3-turbo on cuda
whisper-for-subs  | [GPU 1] ✓ Segment 1 complete: 12 text segments
whisper-for-subs  | [GPU 2] ✓ Segment 2 complete: 18 text segments
whisper-for-subs  | [GPU 3] ✓ Segment 3 complete: 22 text segments
whisper-for-subs  | [GPU 0] ✓ Segment 0 complete: 28 text segments
whisper-for-subs  | ...
whisper-for-subs  | ✅ Complete! 1247 text segments | Speed: 28.3x realtime | Time: 63.6s
```

---

## 🚀 部署

```bash
cd /Users/winston/Projects/whisper-for-subs

# 重建容器
docker compose down
docker compose build
docker compose up -d

# 查看詳細日誌
docker compose logs -f
```

---

## 📊 日誌符號說明

| 符號 | 說明 |
|-----|------|
| 🎯 | 模式/配置信息 |
| 📊 | 統計數據 |
| ✅ | 成功完成 |
| ⚠️ | 警告 |
| ▶ | 開始處理 |
| ✓ | 完成處理 |
| ✂️ | 段落切分 |
| 🚀 | 啟動 |

---

## 🔍 日誌對比

### 單 GPU vs 多 GPU

| 項目 | 單 GPU | 多 GPU |
|-----|--------|--------|
| 初始化 | 🎯 Single-GPU mode: Using GPU 0 | Initialized with 4 GPUs: [0,1,2,3] |
| 處理單位 | Chunk (VAD 段落) | Segment (優化後段落) |
| 並發性 | 順序處理 | 並行處理 |
| GPU 標籤 | [GPU 0] | [GPU 0] [GPU 1] [GPU 2] [GPU 3] |
| 速度 | ~10x realtime | ~28x realtime |

---

## ✨ 增強的功能

### 1. GPU 識別
- 顯示使用的 GPU 編號
- 便於監控和除錯

### 2. 處理進度
- 每個 chunk 的開始和完成
- 顯示 chunk 數量和時長
- 顯示產生的文字段落數

### 3. 統計信息
- 設備信息（GPU 0 / CPU）
- 總段落數
- 音訊時長
- 處理時間
- 速度比率（倍速）

### 4. 視覺化改進
- 使用表情符號增強可讀性
- 清晰的階段分隔
- 一致的格式

---

## 🎯 使用場景

### 除錯模式
查看詳細的處理過程，了解：
- 哪個 GPU 在工作
- VAD 切分了多少段落
- 每個段落的處理時間
- 是否有段落失敗

### 性能分析
比較：
- 單 GPU 和多 GPU 的速度
- 不同音訊長度的處理效率
- VAD 切分的影響

### 監控運行
實時查看：
- 當前處理進度
- GPU 使用情況
- 預估完成時間

---

## 📝 修改的檔案

只修改了一個檔案：
- ✅ `/Users/winston/Projects/whisper-for-subs/transcriber.py`

新增的日誌功能：
1. GPU 索引檢測和顯示
2. 詳細的初始化日誌
3. 音訊載入信息
4. VAD 檢測結果
5. Chunk 處理進度
6. 最終統計摘要

---

## 🎉 總結

### 問題
單 GPU 模式日誌過於簡單，缺少細節

### 解決方案
在 `transcriber.py` 中增加詳細日誌輸出

### 結果
- ✅ 清晰顯示使用 GPU 0
- ✅ 詳細的處理進度
- ✅ 每個 chunk 的狀態
- ✅ 完整的統計信息
- ✅ 與多 GPU 模式風格一致

---

**立即部署，享受詳細的日誌輸出！** 🚀

```bash
docker compose down && docker compose build && docker compose up -d && docker compose logs -f
```
