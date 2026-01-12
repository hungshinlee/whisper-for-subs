# 使用改進版本

## 快速開始

改進版本完全向後相容，使用方式與原版相同：

```bash
# 啟動服務
python app.py

# 或使用 Docker
docker-compose up
```

服務會在啟動時顯示改進資訊：

```
============================================================
🚀 Starting Whisper ASR Service (Improved Version)
============================================================

🧹 Cleaning up old files...
✨ Improvements enabled:
  - Session-based isolation
  - Transcriber pool (max 2 concurrent)
  - Enhanced file cleanup
  - UUID-based naming
```

## 主要改進

### 1. 多用戶並發支援

現在可以安全地處理多個用戶同時使用：

- **自動隔離**：每個請求使用獨立的 transcriber 實例
- **資源管理**：最多支援 2 個並發請求（可調整）
- **無干擾**：不同用戶的轉錄結果完全獨立

**測試方法：**
```bash
# 開啟兩個瀏覽器分頁，同時上傳不同音檔
# 兩個請求都會正常完成，互不影響
```

### 2. 完整的檔案清理

所有臨時檔案都會被自動清理：

- ✅ YouTube 下載的音檔
- ✅ 用戶上傳檔案的臨時複製
- ✅ Session 工作目錄
- ✅ 超過 24 小時的舊檔案

**檔案位置：**
```
/tmp/whisper-sessions/{session-id}/  # 處理完立即刪除
/tmp/whisper-downloads/              # 24 小時後清理
/app/outputs/*.srt                   # 24 小時後清理
```

### 3. Session 追蹤

每個請求都有唯一的 Session ID，方便追蹤和除錯：

```
Session: abc123def456
```

可在日誌中看到完整的處理過程：

```
============================================================
🎬 Starting session: abc123def456
============================================================
📁 Uploaded file copied to session: /tmp/whisper-sessions/abc123def456/upload_xyz.mp3
⏱️  Audio duration: 180.5s
🔧 Using single-GPU transcriber: single_a1b2c3d4
📝 Generated 45 segments
...
✅ Session completed: abc123def456
============================================================
```

## 進階設定

### 調整並發數量

編輯 `app.py` 中的 TranscriberPool 初始化：

```python
# 預設：最多 2 個並發
transcriber_pool = TranscriberPool(max_workers=2)

# 增加到 4 個（需要足夠的 GPU 記憶體）
transcriber_pool = TranscriberPool(max_workers=4)
```

**建議設定：**
- 單 GPU (16GB)：`max_workers=1-2`
- 單 GPU (24GB+)：`max_workers=2-3`
- 多 GPU：`max_workers=2-4`（取決於 GPU 數量）

### 調整清理時間

```python
# 預設：24 小時
cleanup_old_files(max_age_hours=24)

# 縮短到 12 小時
cleanup_old_files(max_age_hours=12)

# 延長到 48 小時
cleanup_old_files(max_age_hours=48)
```

## 驗證改進

運行測試腳本：

```bash
python test_improvements.py
```

預期輸出：

```
============================================================
🔍 Whisper-for-Subs Improvement Verification
============================================================

📋 Test 1: Session Directory Cleanup
------------------------------------------------------------
✅ No session directory exists (good - nothing to clean)

📋 Test 2: Temporary Files Cleanup
------------------------------------------------------------
✅ /tmp/whisper-downloads: clean
✅ /tmp/whisper-sessions: clean

📋 Test 3: TranscriberPool
------------------------------------------------------------
✅ TranscriberPool initialized
   Max workers: 2
   Single GPU pool size: 0
   Parallel pool size: 0

============================================================
📊 Test Summary
============================================================
✅ PASS: Session Cleanup
✅ PASS: Temp Files
✅ PASS: TranscriberPool

🎉 All tests passed! Improvements are working correctly.
```

## 監控系統

### 查看 Session 目錄

```bash
# 檢查當前活躍的 session
ls -la /tmp/whisper-sessions/

# 應該在處理時看到目錄，處理完後自動消失
```

### 查看記憶體使用

```bash
# 查看 GPU 記憶體
nvidia-smi

# 當有 2 個並發請求時，會看到 2 個 Python 進程使用 GPU
```

### 查看日誌

服務運行時會輸出詳細日誌：

```bash
python app.py 2>&1 | tee whisper.log
```

## 故障排除

### 問題：記憶體不足 (OOM)

**現象：** CUDA out of memory 錯誤

**解決：**
```python
# 降低並發數
transcriber_pool = TranscriberPool(max_workers=1)
```

### 問題：磁碟空間不足

**檢查：**
```bash
df -h /tmp
```

**解決：**
```python
# 縮短清理週期
cleanup_old_files(max_age_hours=6)  # 6 小時清理一次
```

### 問題：Session 目錄沒有清理

**檢查：**
```bash
ls -la /tmp/whisper-sessions/
```

**手動清理：**
```bash
rm -rf /tmp/whisper-sessions/*
```

## 效能比較

### 並發處理

| 場景 | 原版 | 改進版 |
|------|------|--------|
| 單一請求 | ✅ 正常 | ✅ 正常 |
| 2 個並發 | ⚠️ 可能干擾 | ✅ 完全隔離 |
| 記憶體使用 | 1x | 2x (並發時) |

### 檔案清理

| 檔案類型 | 原版 | 改進版 |
|---------|------|--------|
| YouTube 下載 | ✅ 清理 | ✅ 清理 |
| 輸出 SRT | ✅ 清理 | ✅ 清理 |
| 用戶上傳 | ❌ 不清理 | ✅ 清理 |
| Session 目錄 | - | ✅ 清理 |

## 相容性

- ✅ 完全向後相容
- ✅ API 不變
- ✅ 使用方式不變
- ✅ 所有功能保持原樣
- ✅ 只是內部改進

## 更多資訊

詳細的技術說明請參考：
- [IMPROVEMENTS.md](IMPROVEMENTS.md) - 完整的改進說明
- [test_improvements.py](test_improvements.py) - 測試腳本

---

**享受更穩定、更可靠的 Whisper ASR 服務！** 🎉
