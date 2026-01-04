# 修正單 GPU 模式 - 只使用第一張 GPU

## 📋 修改說明

當取消勾選「Use Multi-GPU Parallel Processing」時，系統應該只使用第一張 GPU（GPU 0），而不是所有 GPU。

## 🔧 修改內容

需要修改 `app.py` 中的兩個地方：

### 1. 修改 get_transcriber 函數

**原代碼（約第 115-127 行）：**
```python
def get_transcriber(
    model_size: str = "large-v3",
    use_vad: bool = True,
) -> WhisperTranscriber:
    """Get or create single-GPU transcriber instance."""
    global transcriber
    
    if transcriber is None or transcriber.model_size != model_size:
        transcriber = WhisperTranscriber(
            model_size=model_size,
            device=os.environ.get("WHISPER_DEVICE", "cuda"),
            compute_type=os.environ.get("WHISPER_COMPUTE_TYPE", "float16"),
            use_vad=use_vad,
        )
    
    return transcriber
```

**修改為：**
```python
def get_transcriber(
    model_size: str = "large-v3",
    use_vad: bool = True,
) -> WhisperTranscriber:
    """Get or create single-GPU transcriber instance (uses only GPU 0)."""
    global transcriber
    
    if transcriber is None or transcriber.model_size != model_size:
        # For single GPU mode, explicitly use only the first GPU (cuda:0)
        device = os.environ.get("WHISPER_DEVICE", "cuda")
        if device == "cuda":
            device = "cuda:0"  # Explicitly use GPU 0
        
        transcriber = WhisperTranscriber(
            model_size=model_size,
            device=device,
            compute_type=os.environ.get("WHISPER_COMPUTE_TYPE", "float16"),
            use_vad=use_vad,
        )
    
    return transcriber
```

### 2. 修改 process_audio 函數中的提示信息

在 process_audio 函數中（約第 250 行），將：
```python
# Single GPU processing
yield format_progress_html(35, "Loading Whisper model..."), "", None
trans = get_transcriber(model_size, use_vad)

yield format_progress_html(40, "Model loaded. Starting transcription..."), "", None
```

修改為：
```python
# Single GPU processing (uses only GPU 0)
yield format_progress_html(35, "Loading Whisper model on GPU 0..."), "", None
trans = get_transcriber(model_size, use_vad)

yield format_progress_html(40, "Model loaded on GPU 0. Starting transcription..."), "", None
```

並修改狀態信息（約第 297 行）：
```python
# Format status message with duration and processing time
gpu_info = f"{num_gpus_used} GPUs" if use_parallel else "GPU 0 (single)"
```

---

## 🚀 快速修復

完整的修改已準備在：`tmp/app_fixed.py`

部署方法：
```bash
cd /Users/winston/Projects/whisper-for-subs

# 備份
cp app.py app.py.backup_gpu

# 部署修復版本
cp tmp/app_fixed.py app.py

# 重建容器
docker compose down
docker compose build
docker compose up -d

# 查看日誌
docker compose logs -f
```

---

## ✅ 預期效果

### 單 GPU 模式（取消勾選）
```
Loading Whisper model on GPU 0...
Model loaded on GPU 0. Starting transcription...
✅ Transcription complete! 
Mode: GPU 0 (single)
```

### 多 GPU 模式（勾選）
```
Starting parallel transcription on 4 GPUs...
[GPU 0] ▶ Processing segment 0
[GPU 1] ▶ Processing segment 1
[GPU 2] ▶ Processing segment 2
[GPU 3] ▶ Processing segment 3
✅ Transcription complete!
Mode: 4 GPUs
```

---

## 🔍 驗證方法

### 測試單 GPU 模式

1. 上傳短音訊（< 5 分鐘）
2. **取消勾選** 「Use Multi-GPU」
3. 點擊「🚀 Start」
4. 監控 GPU 使用情況：

```bash
# 在另一個終端執行
watch -n 1 nvidia-smi

# 應該只看到 GPU 0 在使用
# GPU 1, 2, 3 應該閒置
```

### 測試多 GPU 模式

1. 上傳長音訊（> 5 分鐘）
2. **勾選** 「Use Multi-GPU」
3. 點擊「🚀 Start」
4. 應該看到 4 張 GPU 都在工作

---

## 📊 修改前後對比

| 模式 | 修改前 | 修改後 |
|-----|--------|--------|
| 單 GPU（取消勾選）| 可能使用多張 GPU ❌ | 只使用 GPU 0 ✅ |
| 多 GPU（勾選）| 使用 4 張 GPU ✅ | 使用 4 張 GPU ✅ |
| 提示信息 | 不明確 | 清楚標示使用的 GPU |

---

## 💡 技術說明

### 為什麼需要 cuda:0

在 PyTorch 和 faster-whisper 中：
- `device="cuda"` - 使用預設 GPU（通常是 GPU 0，但不保證）
- `device="cuda:0"` - **明確使用 GPU 0**
- `device="cuda:1"` - 明確使用 GPU 1

當 `CUDA_VISIBLE_DEVICES=0,1,2,3` 時，所有 4 張 GPU 都可見，所以需要明確指定 `cuda:0` 來確保只使用第一張。

### 多 GPU 模式如何工作

多 GPU 模式使用 `CUDA_VISIBLE_DEVICES` 環境變數在**每個子進程中**控制可見的 GPU：
- 子進程 1: `CUDA_VISIBLE_DEVICES=0` → 只看到 GPU 0
- 子進程 2: `CUDA_VISIBLE_DEVICES=1` → 只看到 GPU 1
- 子進程 3: `CUDA_VISIBLE_DEVICES=2` → 只看到 GPU 2
- 子進程 4: `CUDA_VISIBLE_DEVICES=3` → 只看到 GPU 3

這樣每個進程都獨立使用一張 GPU，實現並行處理。
