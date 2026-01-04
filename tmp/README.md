# 多 GPU 功能改進版本

## 📁 檔案說明

本目錄包含改進版本的 `parallel_transcriber.py`，修復了原版本的一些問題。

### 檔案列表

- **parallel_transcriber_improved.py** - 改進版本的主要檔案

---

## ✨ 主要改進

### 1. 更好的錯誤處理
- ✅ 初始化 `temp_path = None` 避免未定義錯誤
- ✅ 驗證音訊數據不為空
- ✅ 詳細的錯誤日誌和完整 traceback
- ✅ 安全的 finally 清理機制

### 2. 過濾極短段落
- ✅ 自動跳過 < 100ms 的音訊段落
- ✅ 在 VAD 優化階段過濾 < 500ms 的段落
- ✅ 減少無效的轉錄嘗試和失敗率

### 3. 優化的參數設定
- ✅ `min_segment_duration`: 10s → 15s
- ✅ `max_segment_duration`: 60s → 45s
- ✅ 更均衡的 GPU 負載分配

### 4. 詳細的進度日誌
```
[GPU 0] ▶ Processing segment 0 (18.3s)
[GPU 0] ✓ Segment 0 complete: 12 text segments
[GPU 2] ⊘ Segment 5 too short (0.08s), skipping
[GPU 1] ✗ ERROR in segment 7: CUDA out of memory
```

### 5. 統計資訊輸出
```
📊 Audio loaded: 1800.0s (28800000 samples @ 16000Hz)
🎯 VAD detected 245 speech segments
✂️  Optimized to 72 segments for 4 GPUs
🚀 Starting parallel transcription on 4 GPUs...
⊘ 15 segments skipped (too short)
⚠️  2 segments failed to transcribe:
   - Segment 34: CUDA out of memory
✅ Complete! 1,247 text segments | Speed: 28.3x realtime | Time: 127.3s
```

---

## 🚀 使用方法

### 方法 1: 直接替換（推薦）

```bash
# 備份原檔案
cd /Users/winston/Projects/whisper-for-subs
cp parallel_transcriber.py parallel_transcriber.py.backup

# 複製改進版本
cp tmp/parallel_transcriber_improved.py parallel_transcriber.py

# 重新建置並啟動
docker compose down
docker compose build
docker compose up -d

# 查看日誌
docker compose logs -f
```

### 方法 2: 測試後再替換

```bash
# 1. 先在本地測試
cd /Users/winston/Projects/whisper-for-subs

# 2. 重命名改進版本
cp tmp/parallel_transcriber_improved.py parallel_transcriber_test.py

# 3. 在 app.py 中暫時修改 import
# from parallel_transcriber import ...
# 改為
# from parallel_transcriber_test import ...

# 4. 測試完成後再替換
```

---

## 🔍 驗證改進效果

### 預期改善

| 指標 | 原版本 | 改進版本 |
|-----|--------|---------|
| 轉錄失敗率 | 可能 > 50% | < 5% |
| 極短段落處理 | 嘗試轉錄並失敗 | 自動跳過 |
| 錯誤訊息 | 簡單 | 詳細 traceback |
| 進度顯示 | 基本 | 清晰標示狀態 |
| VAD 段落數 | 可能很多 | 經過優化 |

### 測試方法

1. **上傳測試音訊**
   - 使用 10-30 分鐘的音訊檔案
   - 勾選「🚀 Use Multi-GPU Parallel Processing」
   
2. **觀察日誌**
   ```bash
   docker compose logs -f | grep -E "GPU|segment|Complete"
   ```

3. **檢查統計**
   - 查看有多少段落被跳過
   - 查看失敗率是否降低
   - 確認處理速度

---

## 📊 與原版本的差異

### 主要程式碼變更

#### transcribe_segment_on_gpu 函數

**原版本問題：**
```python
def transcribe_segment_on_gpu(args):
    # temp_path 未初始化
    try:
        with tempfile.NamedTemporaryFile(...) as temp_file:
            temp_path = temp_file.name
        # 沒有檢查音訊長度
    finally:
        if os.path.exists(temp_path):  # 可能未定義
            os.unlink(temp_path)
```

**改進版本：**
```python
def transcribe_segment_on_gpu(args):
    temp_path = None  # 明確初始化
    
    try:
        # 驗證音訊數據
        if len(audio_data) == 0:
            raise ValueError(...)
        
        # 過濾極短段落
        if duration < 0.1:
            return {"skipped": True, ...}
        
        # 詳細日誌
        print(f"[GPU {gpu_id}] ▶ Processing segment {segment_idx}")
        
    except Exception as e:
        # 詳細錯誤資訊
        traceback.print_exc()
        
    finally:
        # 安全清理
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                print(f"Warning: {e}")
```

#### _optimize_segments 函數

**新增：**
```python
# 過濾太短的 VAD 段落
if duration < 0.5:  # Less than 500ms
    continue
```

---

## 🐛 解決的問題

1. ✅ **UnboundLocalError: temp_path** - 已初始化變數
2. ✅ **大量段落失敗** - 過濾極短段落
3. ✅ **缺少錯誤細節** - 新增完整 traceback
4. ✅ **難以追蹤進度** - 每個段落都有日誌
5. ✅ **VAD 段落太多** - 優化合併和分割邏輯

---

## 📝 注意事項

1. **備份很重要** - 替換前一定要備份原檔案
2. **模型載入次數** - 仍會看到多次載入，這是正常的
3. **跳過段落** - 看到 "skipped" 是正常的，不是錯誤
4. **監控 GPU** - 使用 `nvidia-smi` 確認多 GPU 運作

---

## 🆘 如果問題仍然存在

### 診斷步驟

```bash
# 1. 檢查改進版本是否正確載入
docker exec whisper-for-subs python -c "
from parallel_transcriber import transcribe_segment_on_gpu
import inspect
source = inspect.getsource(transcribe_segment_on_gpu)
if 'temp_path = None' in source:
    print('✅ Using improved version')
else:
    print('⚠️  Still using old version')
"

# 2. 查看詳細錯誤
docker logs whisper-for-subs --tail=200 | grep -A 10 "ERROR\|Traceback"

# 3. 測試單一段落
docker exec whisper-for-subs python -c "
import numpy as np
from parallel_transcriber import transcribe_segment_on_gpu

audio = np.random.randn(80000).astype('float32') * 0.01
args = (0, audio, 0.0, 5.0, 0, 'large-v3-turbo', 'zh', 'transcribe', 'float16')
result = transcribe_segment_on_gpu(args)
print('Result:', result)
"
```

---

## 📚 相關文件

- **TROUBLESHOOTING_MULTI_GPU.md** - 完整的故障排除指南
- **DEPLOYMENT_GUIDE.md** - 部署與測試指南
- **MULTI_GPU_GUIDE.md** - 多 GPU 功能使用指南

---

## ✅ 檢查清單

部署改進版本後：

- [ ] 備份了原版本 `parallel_transcriber.py`
- [ ] 複製了改進版本
- [ ] 重新建置了 Docker 容器
- [ ] 查看日誌確認改進版本運作
- [ ] 測試了音訊轉錄功能
- [ ] 確認錯誤率降低
- [ ] 驗證了 GPU 使用情況

---

**建議**: 立即部署這個改進版本，應該能大幅降低轉錄失敗率！🚀
