# Stereo to Mono 音訊處理修復

## 問題描述
當上傳 stereo（雙聲道）MP3 音檔時，Silero VAD 會拋出錯誤：
```
ValueError: More than one dimension in audio. Are you trying to process audio with 2 channels?
```

## 根本原因
在 `parallel_transcriber.py` 中，音訊使用 `soundfile.read()` 直接加載，如果是 stereo 音檔，會返回 shape 為 `(samples, 2)` 的 2D 陣列。Silero VAD 需要 mono（單聲道）音訊，即 1D 陣列。

## 解決方案

### 1. parallel_transcriber.py（第 227-232 行）
**主要修復** - 在 VAD 處理前將 stereo 轉為 mono：
```python
# Load audio
audio, sample_rate = sf.read(audio_path, dtype="float32")

# Convert stereo to mono if needed
if audio.ndim == 2:
    print(f"🔄 Converting stereo audio to mono ({audio.shape[1]} channels)")
    audio = audio.mean(axis=1)  # Average channels

total_duration = len(audio) / sample_rate
```

### 2. vad.py（第 67-76 行）
**防護措施** - 改進 VAD 的音訊處理邏輯：
```python
# Ensure 1D (convert stereo to mono if needed)
if audio_tensor.dim() > 1:
    if audio_tensor.shape[1] > 1:
        # Multiple channels - average them to mono
        audio_tensor = audio_tensor.mean(dim=1)
    else:
        # Single channel - just squeeze
        audio_tensor = audio_tensor.squeeze()
```

## 為什麼這是最佳方案

### Stereo 轉 Mono 的方法比較：

1. **平均聲道（推薦）** ✅
   ```python
   mono = stereo.mean(axis=1)
   ```
   - 優點：保留兩個聲道的資訊，音質平衡
   - 缺點：無
   - **這是我們使用的方法**

2. **只取左聲道**
   ```python
   mono = stereo[:, 0]
   ```
   - 優點：簡單快速
   - 缺點：丟失右聲道資訊

3. **使用 FFmpeg（transcriber.py 已使用）**
   ```bash
   ffmpeg -i input.mp3 -ac 1 output.wav
   ```
   - 優點：專業音訊處理，品質最高
   - 缺點：需要額外的 I/O 操作

## 測試建議
1. 測試 stereo MP3 音檔
2. 測試 mono MP3 音檔
3. 測試其他格式（WAV, M4A 等）
4. 確認單 GPU 和多 GPU 模式都正常

## 注意事項
- `transcriber.py` 不需要修改，因為它使用 ffmpeg 的 `-ac 1` 參數已經處理了聲道轉換
- 修復後的代碼向後兼容，對 mono 音檔不會有任何影響
