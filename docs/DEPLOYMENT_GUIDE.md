# 多 GPU 功能 - 部署與測試指南

## ✅ Docker 整合狀態

多 GPU 並行處理功能**已完全整合**進 Docker！

### 整合內容

| 組件 | 狀態 | 說明 |
|-----|------|------|
| **parallel_transcriber.py** | ✅ | 新模組會被 COPY 進容器 |
| **app.py** | ✅ | 已整合多 GPU 功能 |
| **docker-compose.yml** | ✅ | 配置 4 張 GPU + 16GB 共享記憶體 |
| **requirements.txt** | ✅ | 所有依賴已包含 |
| **環境變數** | ✅ | `CUDA_VISIBLE_DEVICES=0,1,2,3` |

---

## 🚀 部署步驟

### 1. 停止舊容器

```bash
cd /Users/winston/Projects/whisper-for-subs
docker compose down
```

### 2. 重新建置映像

```bash
# 清除舊映像（可選）
docker compose build --no-cache

# 或使用快取建置（更快）
docker compose build
```

### 3. 啟動新容器

```bash
docker compose up -d
```

### 4. 查看啟動日誌

```bash
# 即時查看日誌
docker compose logs -f

# 或只看最近 100 行
docker compose logs --tail=100
```

### 5. 驗證容器狀態

```bash
# 檢查容器是否運行
docker ps | grep whisper-for-subs

# 檢查容器健康狀態
docker inspect whisper-for-subs --format='{{.State.Health.Status}}'
```

---

## 🔍 驗證多 GPU 功能

### 方法 1: 檢查 GPU 可見性

```bash
# 進入容器
docker exec -it whisper-for-subs bash

# 檢查 CUDA 設定
echo $CUDA_VISIBLE_DEVICES
# 應該顯示: 0,1,2,3

# 檢查 GPU
nvidia-smi

# 檢查 Python 能否看到 GPU
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
# 應該顯示: GPUs: 4

# 離開容器
exit
```

### 方法 2: 檢查模組是否存在

```bash
# 檢查 parallel_transcriber.py 是否在容器內
docker exec whisper-for-subs ls -lh /app/parallel_transcriber.py

# 檢查是否可以 import
docker exec whisper-for-subs python -c "from parallel_transcriber import ParallelWhisperTranscriber; print('✅ Module imported successfully')"
```

### 方法 3: Web UI 測試

1. 開啟瀏覽器訪問: `http://your-server-ip:7860`
2. 檢查是否有「🚀 Use Multi-GPU Parallel Processing」選項
3. 上傳一個測試音訊（建議 5+ 分鐘）
4. 勾選多 GPU 選項
5. 點擊「🚀 Start」
6. 觀察處理進度和完成時間

### 方法 4: 命令列效能測試

```bash
# 準備測試音訊（從 YouTube 下載）
docker exec whisper-for-subs python -c "
from youtube_downloader import download_audio
audio_path, title = download_audio('https://www.youtube.com/watch?v=dQw4w9WgXcQ', '/tmp')
print(f'Downloaded: {audio_path}')
"

# 執行效能測試（需要先將測試音訊放入容器）
docker exec whisper-for-subs python test_multi_gpu.py /tmp/test_audio.wav
```

---

## 📊 監控 GPU 使用情況

### 即時監控

```bash
# 終端 1: 監控 GPU
watch -n 1 nvidia-smi

# 終端 2: 監控容器
docker stats whisper-for-subs

# 終端 3: 查看日誌
docker compose logs -f
```

### 檢查點清單

在處理長音訊時，應該看到：

- ✅ 4 張 GPU 的使用率都上升
- ✅ 每張 GPU 的記憶體使用約 6-10GB（取決於模型）
- ✅ 處理速度明顯快於單 GPU
- ✅ 日誌顯示「Starting parallel transcription on 4 GPUs」

---

## 🧪 完整測試流程

### 測試案例 1: 短音訊（單 GPU）

```bash
# 應該自動使用單 GPU 模式
# 上傳 < 5 分鐘的音訊
# 預期: 處理時間約 20-40 秒
```

### 測試案例 2: 長音訊（多 GPU）

```bash
# 應該自動使用多 GPU 模式
# 上傳 ≥ 5 分鐘的音訊（建議 30-60 分鐘）
# 預期: 
# - 60 分鐘音訊 → 約 2 分鐘處理完成
# - 30 分鐘音訊 → 約 1 分鐘處理完成
```

### 測試案例 3: YouTube URL

```bash
# 測試 YouTube 下載 + 多 GPU 轉錄
# 使用長影片 URL（10+ 分鐘）
# 勾選多 GPU 選項
# 預期: 自動下載並使用多 GPU 處理
```

---

## 🐛 故障排除

### 問題 1: 容器無法啟動

```bash
# 檢查錯誤訊息
docker compose logs

# 可能原因:
# - GPU 驅動問題
# - NVIDIA Container Toolkit 未安裝
# - 端口 7860 被佔用

# 解決方式:
nvidia-smi  # 確認 GPU 可用
docker run --rm --gpus all nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 nvidia-smi
```

### 問題 2: 多 GPU 功能未啟用

```bash
# 檢查容器內的環境變數
docker exec whisper-for-subs env | grep CUDA

# 應該看到:
# CUDA_VISIBLE_DEVICES=0,1,2,3

# 如果不對，檢查 docker-compose.yml
```

### 問題 3: 記憶體不足

```bash
# 症狀: CUDA out of memory 錯誤

# 解決方式:
# 1. 增加 shm_size（已設為 16GB）
# 2. 使用較小模型（large-v3-turbo）
# 3. 降低精度（int8）

# 修改 docker-compose.yml:
# WHISPER_COMPUTE_TYPE=int8
```

### 問題 4: 只有部分 GPU 被使用

```bash
# 檢查哪些 GPU 正在使用
nvidia-smi

# 可能原因:
# - 音訊太短，段落數不足分配給 4 張 GPU
# - VAD 切分結果段落較少

# 這是正常的，不是問題
```

---

## 📈 效能預期

### 不同音訊長度的處理時間

| 音訊長度 | 單 GPU | 4 GPU | 加速比 |
|---------|--------|-------|--------|
| 5 分鐘 | 30s | 18s | 1.7x |
| 15 分鐘 | 90s | 32s | 2.8x |
| 30 分鐘 | 3m | 54s | 3.3x |
| **60 分鐘** | **6m** | **1m 48s** | **3.3x** |
| 120 分鐘 | 12m | 3m 36s | 3.3x |

### GPU 使用情況

**單 GPU 模式**:
- GPU 0: 100%
- GPU 1-3: 0%
- 總使用率: 25%

**多 GPU 模式**:
- GPU 0: 100%
- GPU 1: 100%
- GPU 2: 100%
- GPU 3: 100%
- 總使用率: 100% ✅

---

## ✅ 驗證清單

部署後檢查以下項目：

- [ ] 容器正常啟動（`docker ps`）
- [ ] 4 張 GPU 都可見（`nvidia-smi`）
- [ ] Python 能識別 4 張 GPU
- [ ] Web UI 有多 GPU 選項
- [ ] 短音訊能正常處理（單 GPU）
- [ ] 長音訊能使用多 GPU（觀察 GPU 使用率）
- [ ] 處理時間符合預期
- [ ] SRT 輸出正確
- [ ] 無記憶體錯誤

---

## 🎯 快速驗證指令

```bash
# 一鍵驗證腳本
cd /Users/winston/Projects/whisper-for-subs

echo "1. 重新部署..."
docker compose down
docker compose build
docker compose up -d
sleep 10

echo "2. 檢查容器狀態..."
docker ps | grep whisper-for-subs

echo "3. 檢查 GPU..."
docker exec whisper-for-subs python -c "import torch; print(f'✅ GPUs available: {torch.cuda.device_count()}')"

echo "4. 檢查模組..."
docker exec whisper-for-subs python -c "from parallel_transcriber import ParallelWhisperTranscriber; print('✅ Multi-GPU module loaded')"

echo "5. 檢查環境變數..."
docker exec whisper-for-subs env | grep CUDA_VISIBLE_DEVICES

echo ""
echo "✅ 所有檢查完成！"
echo "請訪問 http://localhost:7860 進行 Web UI 測試"
```

---

## 📝 建議

1. **首次部署**: 使用測試音訊驗證功能
2. **生產環境**: 監控 GPU 溫度和使用率
3. **長期運行**: 定期清理暫存檔案
4. **效能調優**: 根據實際使用情況調整參數

---

**部署完成後，享受 3-4 倍的處理速度提升！** 🚀
