# 多 GPU 功能問題診斷與修復

## 📊 當前狀況分析

根據你的日誌：

```
Initialized ParallelWhisperTranscriber with 4 GPUs: [0, 1, 2, 3]
Loading Whisper model: large-v3-turbo on cuda (重複 19 次)
Warning: 19 segments failed to transcribe
```

### ✅ 正常的部分
1. **多 GPU 功能已啟動** - 4 張 GPU 正確識別
2. **模型載入 19 次** - 這是**正常行為**（每個子進程處理一個段落時載入模型）
3. **CUDA 和 Gradio** - 都正常運作

### ⚠️ 問題：19 個段落轉錄失敗

可能的原因：
1. **音訊段落太短** - VAD 切分產生了很多極短的段落
2. **暫存檔案問題** - 子進程間可能有檔案衝突
3. **變數未初始化** - `temp_path` 在異常時可能未定義

---

## 🔧 修復方案

### 方案 1: 更新 parallel_transcriber.py（推薦）

我已經為你準備了改進版本，包含：

1. **更好的錯誤處理**
   - 詳細的錯誤日誌和 traceback
   - 檢查音訊段落長度
   - 安全的暫存檔案清理

2. **過濾極短段落**
   - 自動跳過 < 100ms 的段落
   - 減少無效的轉錄嘗試

3. **進度日誌**
   - 每個段落的處理狀態
   - 清楚標示 GPU 使用情況

將以下改進版本的代碼複製到容器中：

```python
# 改進的 transcribe_segment_on_gpu 函數
def transcribe_segment_on_gpu(args: tuple) -> Dict:
    """
    Transcribe a single audio segment on a specific GPU.
    """
    (
        segment_idx, audio_data, start_time, end_time,
        gpu_id, model_size, language, task, compute_type,
    ) = args
    
    temp_path = None  # 初始化變數
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    try:
        # 驗證音訊數據
        if len(audio_data) == 0:
            raise ValueError(f"Segment {segment_idx}: Empty audio data")
        
        duration = end_time - start_time
        
        # 過濾太短的段落
        if duration < 0.1:  # 小於 100ms
            print(f"Warning: Segment {segment_idx} too short ({duration:.2f}s), skipping")
            return {
                "segment_idx": segment_idx,
                "success": True,
                "segments": [],
                "gpu_id": gpu_id,
                "duration": duration,
                "skipped": True,
            }
        
        # 創建暫存檔案
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            sf.write(temp_path, audio_data, 16000)
        
        print(f"[GPU {gpu_id}] Processing segment {segment_idx} ({duration:.1f}s)")
        
        # 初始化並轉錄
        transcriber = WhisperTranscriber(
            model_size=model_size,
            device="cuda",
            compute_type=compute_type,
            use_vad=False,
        )
        
        segments = transcriber.transcribe(
            temp_path,
            language=language,
            task=task,
            progress_callback=None,
        )
        
        # 調整時間戳
        adjusted_segments = []
        for seg in segments:
            adjusted_segments.append({
                "start": start_time + seg["start"],
                "end": start_time + seg["end"],
                "text": seg["text"],
            })
        
        print(f"[GPU {gpu_id}] ✓ Segment {segment_idx}: {len(adjusted_segments)} texts")
        
        return {
            "segment_idx": segment_idx,
            "success": True,
            "segments": adjusted_segments,
            "gpu_id": gpu_id,
            "duration": duration,
        }
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"[GPU {gpu_id}] ✗ ERROR in segment {segment_idx}: {str(e)}")
        print(f"[GPU {gpu_id}] Traceback:\n{error_detail}")
        
        return {
            "segment_idx": segment_idx,
            "success": False,
            "error": str(e),
            "error_detail": error_detail,
            "gpu_id": gpu_id,
        }
    
    finally:
        # 安全清理暫存檔案
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                print(f"Warning: Could not delete {temp_path}: {e}")
```

### 方案 2: 調整 VAD 參數（臨時措施）

如果無法立即更新代碼，可以調整 VAD 參數來減少極短段落：

在 `app.py` 或 `parallel_transcriber.py` 中：

```python
# 增加最小段落長度
para_trans = ParallelWhisperTranscriber(
    model_size=model_size,
    compute_type=compute_type,
    gpu_ids=gpu_ids,
    vad_threshold=0.5,  # 可以稍微提高到 0.6
)

segments = para_trans.transcribe_parallel(
    audio_path,
    language=language,
    task=task,
    min_segment_duration=15.0,  # 從 10s 增加到 15s
    max_segment_duration=45.0,  # 從 60s 減少到 45s
)
```

---

## 🔍 診斷步驟

### 1. 檢查詳細錯誤

進入容器查看更多細節：

```bash
# 查看完整日誌
docker logs whisper-for-subs --tail=100

# 或即時監控
docker logs -f whisper-for-subs 2>&1 | grep -E "ERROR|Warning|GPU"
```

### 2. 測試單一段落處理

```bash
# 進入容器
docker exec -it whisper-for-subs bash

# 測試基本轉錄功能
python -c "
from transcriber import WhisperTranscriber
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
t = WhisperTranscriber('large-v3-turbo', 'cuda', 'float16', False)
print('✓ Transcriber loaded successfully')
"
```

### 3. 檢查 GPU 記憶體

```bash
# 監控 GPU 使用情況
watch -n 1 nvidia-smi

# 查看是否有記憶體不足
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### 4. 測試不同音訊

```bash
# 用較短的測試音訊（5-10 分鐘）
# 觀察是否仍有失敗
```

---

## 📋 快速修復清單

### 立即可做：

1. **降低失敗率**
   ```bash
   # 在 docker-compose.yml 中調整參數
   environment:
     - WHISPER_COMPUTE_TYPE=float16  # 確認使用 float16
   
   # 重啟容器
   docker compose restart
   ```

2. **增加日誌級別**
   ```bash
   # 在 app.py 開頭加入
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

3. **測試單 GPU 模式**
   ```bash
   # 暫時取消勾選多 GPU 選項
   # 看看單 GPU 是否正常
   ```

### 長期改善：

1. **更新代碼** - 使用上面的改進版本
2. **優化 VAD 參數** - 減少極短段落
3. **監控系統** - 加入 Prometheus + Grafana

---

## 🎯 預期結果

修復後應該看到：

```
✅ 好的日誌：
[GPU 0] Processing segment 0 (12.3s)
[GPU 1] Processing segment 1 (15.7s)
[GPU 2] Processing segment 2 (11.2s)
[GPU 3] Processing segment 3 (18.4s)
[GPU 0] ✓ Segment 0: 8 texts
[GPU 1] ✓ Segment 1: 12 texts
Warning: Segment 4 too short (0.05s), skipping  # 自動跳過
[GPU 2] ✓ Segment 2: 10 texts
...
Complete! 127 segments | Speed: 28.3x realtime
```

---

## 💡 暫時的工作方案

如果需要立即使用，可以：

1. **使用單 GPU 模式**
   - 取消勾選多 GPU 選項
   - 雖然較慢但更穩定

2. **使用較短音訊**
   - 先測試 5-15 分鐘的音訊
   - 確認基本功能正常

3. **手動分段處理**
   - 將長音訊分成多個檔案
   - 分別上傳處理

---

## 📞 需要更多幫助？

如果問題持續：

1. 提供完整的錯誤日誌（約 200 行）
2. 說明測試的音訊特性（長度、來源）
3. 執行診斷指令並分享結果

---

**下一步**：建議先執行診斷步驟 1-3，確認具體的錯誤原因，然後決定使用哪個修復方案。
