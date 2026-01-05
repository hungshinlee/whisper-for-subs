# 更新日誌 (Changelog)

所有專案的重要變更都會記錄在此文件中。

格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-TW/1.0.0/)，
版本號遵循 [Semantic Versioning](https://semver.org/lang/zh-TW/)。

---

## [3.0.0] - 2025-01-05

### 🚀 重大更新

#### 多 GPU 性能優化（持久化 Worker）
- **新增**：持久化 worker 機制，每個 GPU 只載入模型一次
- **改進**：避免重複載入模型，大幅提升處理速度
- **效能**：
  - 10 分鐘音訊：122s → 46s（2.7 倍提升）
  - 60 分鐘音訊：476s → 136s（3.5 倍提升）
  - 速度比：7.6x → 26.5x realtime

#### 中文簡繁轉換
- **新增**：自動簡繁轉換功能（選擇 zh 語言時）
- **技術**：使用 OpenCC (Open Chinese Convert)
- **標準**：台灣標準（s2tw）
- **支援**：專業術語、常用詞彙精準轉換
- **依賴**：添加 `opencc-python-reimplemented>=0.1.7`

#### SRT 複製按鈕
- **新增**：Web UI 中添加「📋 Copy to Clipboard」按鈕
- **功能**：一鍵複製 SRT 字幕內容到剪貼簿
- **反饋**：即時顯示複製狀態（成功/失敗）
- **相容**：支援所有現代瀏覽器（Chrome、Firefox、Safari、Edge）

#### VAD 靈敏度設定
- **新增**：可調整 VAD 最小靜音時長（Min Silence Duration）
- **範圍**：0.01 - 2.0 秒（預設 0.1 秒）
- **用途**：根據音訊類型調整語音切分靈敏度
- **UI**：滑桿控制，動態顯示/隱藏

### ✨ 功能改進

#### 詳細日誌輸出
- **單 GPU 模式**：
  - 顯示使用的 GPU 編號
  - 每個 chunk 的處理進度
  - 完整的統計信息（段落數、時長、速度）
- **多 GPU 模式**：
  - Worker 初始化狀態
  - 每個 GPU 的處理進度
  - 段落分配和合併信息

#### UI/UX 優化
- **進度條**：更美觀的漸層進度條
- **狀態提示**：清晰的成功/警告/錯誤訊息
- **動態控制**：根據選項自動顯示/隱藏相關設定
- **參數說明**：每個設定都有清楚的說明文字

### 🐛 錯誤修復

#### CUDA 初始化錯誤
- **問題**：多 GPU 模式出現 `CUDA failed with error initialization error`
- **原因**：fork 模式與 CUDA 不相容
- **解決**：改用 spawn 模式，設定 `CUDA_VISIBLE_DEVICES`
- **影響**：多 GPU 模式穩定性大幅提升

#### 單 GPU 模式 GPU 選擇
- **問題**：取消多 GPU 時沒有明確使用 GPU 0
- **解決**：使用 `torch.cuda.set_device(0)` 明確設置
- **效果**：確保單 GPU 模式只使用第一張 GPU

### 📝 文件更新

#### README 重寫
- **內容**：完整重寫，包含所有最新功能
- **結構**：更清晰的章節組織
- **範例**：豐富的使用範例和配置說明
- **視覺**：添加 ASCII 圖表和流程圖

#### 技術文件
新增多個詳細的技術文件：
- `CUDA_FIX.md` - CUDA 錯誤修復說明
- `PERFORMANCE_OPTIMIZATION.md` - 性能優化詳解
- `CHINESE_CONVERSION.md` - 簡繁轉換功能說明
- `COPY_BUTTON.md` - 複製按鈕功能說明
- `VAD_MIN_SILENCE_SETTING.md` - VAD 設定說明
- `SESSION_SUMMARY.md` - 開發會話總結

#### 部署腳本
提供多個一鍵部署腳本：
- `deploy_optimized.sh` - 性能優化版本部署
- `deploy_chinese_conversion.sh` - 簡繁轉換部署
- `deploy_copy_button.sh` - 複製按鈕部署
- `deploy_vad_setting.sh` - VAD 設定部署

### 🔧 技術變更

#### 依賴更新
```diff
+ opencc-python-reimplemented>=0.1.7  # 簡繁轉換
```

#### 核心檔案修改
- `transcriber.py` - 添加 min_silence_duration_ms 參數，詳細日誌
- `parallel_transcriber.py` - 持久化 worker，中文轉換整合
- `app.py` - UI 元件更新，簡繁轉換整合
- `vad.py` - 維持不變
- `chinese_converter.py` - 新建，簡繁轉換模組

---

## [2.0.0] - 2025-01-03

### 🚀 新增功能

#### 多 GPU 並行處理
- **功能**：支援 4 張 GPU 同時處理不同音訊段落
- **效能**：長音訊處理速度提升 3-4 倍
- **自動化**：音訊 ≥ 5 分鐘時自動啟用
- **工作原理**：
  1. VAD 切分音訊
  2. 優化段落分配
  3. 多 GPU 並行轉錄
  4. 合併結果

#### Web UI 多 GPU 開關
- **新增**：「🚀 Use Multi-GPU Parallel Processing」選項
- **說明**：自動提示適用於長音訊
- **智能**：短音訊自動使用單 GPU

#### 效能測試工具
- **新增**：`test_multi_gpu.py` 測試腳本
- **功能**：比較單 GPU 和多 GPU 的處理時間
- **輸出**：詳細的性能統計和對比

### 📚 文件

#### 多 GPU 使用指南
- `docs/MULTI_GPU_GUIDE.md` - 完整的使用指南
- `docs/QUICKSTART_MULTI_GPU.md` - 快速開始
- `docs/IMPLEMENTATION_SUMMARY.md` - 技術實作總結

### 🔧 技術改進

#### 核心模組
- **新增**：`parallel_transcriber.py` - 多 GPU 並行處理
- **優化**：VAD 段落合併和切分邏輯
- **改進**：錯誤處理和日誌輸出

#### Docker 配置
- **環境變數**：`CUDA_VISIBLE_DEVICES` 控制可用 GPU
- **優化**：GPU 資源分配

---

## [1.0.0] - 2024-12-25

### 🎉 初始版本

#### 核心功能
- **Whisper ASR**：使用 faster-whisper 實現高效推理
- **多語言支援**：支援 18 種語言
- **雙重模式**：轉錄和翻譯
- **VAD 支援**：Silero VAD 精確語音檢測
- **SRT 輸出**：標準字幕格式

#### 輸入方式
- **檔案上傳**：支援音訊和影片檔案
- **YouTube**：直接輸入 URL 下載並轉錄
- **麥克風**：即時錄音轉錄

#### Web 介面
- **Gradio UI**：美觀、易用的 Web 介面
- **即時進度**：顯示處理進度
- **結果預覽**：即時查看 SRT 內容
- **檔案下載**：下載生成的 SRT 檔案

#### 模型支援
- large-v3
- large-v2
- medium
- small
- base
- tiny

#### 進階功能
- **VAD 語音檢測**：精確的語音段落切分
- **字幕合併**：自動合併過短的字幕
- **可調參數**：自訂每行最大字數
- **自動清理**：定期清理暫存檔案

#### 部署
- **Docker 容器化**：一鍵部署
- **GPU 加速**：NVIDIA CUDA 支援
- **環境變數配置**：靈活的參數設定

#### 文件
- **README.md**：完整的使用說明（繁體中文）
- **README.en.md**：英文說明文件
- **安裝指南**：詳細的系統要求和安裝步驟

---

## 版本號說明

格式：`MAJOR.MINOR.PATCH`

- **MAJOR**：不相容的 API 變更
- **MINOR**：向下相容的功能新增
- **PATCH**：向下相容的錯誤修復

---

## 類型標籤

- **新增 (Added)**：新功能
- **改進 (Changed)**：現有功能的變更
- **棄用 (Deprecated)**：即將移除的功能
- **移除 (Removed)**：已移除的功能
- **修復 (Fixed)**：錯誤修復
- **安全 (Security)**：安全性更新

---

## 未來計劃

### v3.1.0（計劃中）
- [ ] 支援更多語言的簡繁轉換（香港、澳門標準）
- [ ] 批次處理功能（一次處理多個檔案）
- [ ] 自訂字幕樣式（字體、顏色、位置）
- [ ] 匯出更多格式（VTT、ASS、TXT）
- [ ] 使用者帳號系統
- [ ] 處理歷史記錄

### v3.2.0（計劃中）
- [ ] 即時語音轉錄（WebSocket）
- [ ] 字幕編輯器（在線編輯）
- [ ] 時間軸微調工具
- [ ] 多人協作編輯
- [ ] 雲端儲存整合

### v4.0.0（遠期規劃）
- [ ] 說話人識別（Speaker Diarization）
- [ ] 情緒分析（Sentiment Analysis）
- [ ] 自動標點符號優化
- [ ] AI 字幕潤飾
- [ ] 多模態分析（影片內容理解）

---

## 貢獻

歡迎提交 Issue 或 Pull Request！

查看最新動態：[GitHub Releases](https://github.com/hungshinlee/whisper-for-subs/releases)

---

**最後更新**: 2025-01-05
