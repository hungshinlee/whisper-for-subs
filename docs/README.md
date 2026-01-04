# 📚 文件目錄

本目錄包含 whisper-for-subs 專案的詳細技術文件。

## 📖 文件列表

### 🚀 [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)
**部署與測試指南** - Docker 部署、驗證和故障排除

**適合對象**: 系統管理員、部署人員

**內容包含**:
- ✅ Docker 整合狀態確認
- 🚀 完整部署步驟
- 🔍 多 GPU 功能驗證方法
- 📊 GPU 使用監控
- 🧪 完整測試流程
- 🐛 故障排除指南
- ✅ 驗證清單

**何時閱讀**: 部署前必讀，確保多 GPU 功能正常運作

---

### 🚀 [QUICKSTART_MULTI_GPU.md](./QUICKSTART_MULTI_GPU.md)
**快速開始指南** - 5 分鐘快速部署多 GPU 並行處理

**適合對象**: 想要快速上手的使用者

**內容包含**:
- ⚡ 5 分鐘快速部署步驟
- 📊 視覺化架構圖
- 🎯 效能基準測試數據
- 💡 使用技巧和最佳實踐
- ❓ 常見問題解答
- 🎉 真實使用案例

**何時閱讀**: 部署前必讀，快速了解多 GPU 功能

---

### 📗 [MULTI_GPU_GUIDE.md](./MULTI_GPU_GUIDE.md)
**完整使用指南** - 多 GPU 並行處理的詳細文件

**適合對象**: 需要深入了解和自訂配置的使用者

**內容包含**:
- 🔧 詳細的工作原理說明
- ⚙️ 完整的配置選項
- 📈 效能優化建議
- 🔍 故障排除指南
- 🎯 最佳實踐
- 📊 效能測試方法

**何時閱讀**: 
- 想要深入了解技術細節
- 需要調整參數優化效能
- 遇到問題需要排查
- 進行效能調優

---

### 📙 [IMPLEMENTATION_SUMMARY.md](./IMPLEMENTATION_SUMMARY.md)
**技術實作總結** - 開發者和維護者的技術文件

**適合對象**: 開發者、系統管理員、技術研究者

**內容包含**:
- 🏗️ 完整的技術架構說明
- 💻 實作細節和設計決策
- 📊 效能分析（含 Amdahl's Law）
- 🔧 部署和維護指南
- 🧪 測試方法和驗證清單
- 🔮 未來優化方向

**何時閱讀**: 
- 需要理解內部實作
- 計劃修改或擴展功能
- 進行系統維護
- 撰寫技術文件或論文

---

## 🗺️ 閱讀路徑建議

### 初次使用者
```
1. 主 README.md (專案概覽)
   ↓
2. QUICKSTART_MULTI_GPU.md (快速上手)
   ↓
3. 實際測試和使用
```

### 進階使用者
```
1. QUICKSTART_MULTI_GPU.md (快速回顧)
   ↓
2. MULTI_GPU_GUIDE.md (深入理解)
   ↓
3. 根據需求調整配置
```

### 開發者/研究者
```
1. QUICKSTART_MULTI_GPU.md (功能概覽)
   ↓
2. MULTI_GPU_GUIDE.md (使用細節)
   ↓
3. IMPLEMENTATION_SUMMARY.md (技術深度)
   ↓
4. 閱讀原始碼
```

---

## 📊 文件統計

| 文件 | 行數 | 字數 | 閱讀時間 |
|-----|------|------|---------|
| QUICKSTART_MULTI_GPU.md | 400+ | ~3000 | 10 分鐘 |
| MULTI_GPU_GUIDE.md | 700+ | ~5000 | 20 分鐘 |
| IMPLEMENTATION_SUMMARY.md | 600+ | ~4500 | 15 分鐘 |
| **總計** | **1700+** | **~12500** | **45 分鐘** |

---

## 🔗 外部資源

### 相關技術文件
- [OpenAI Whisper 文件](https://github.com/openai/whisper)
- [faster-whisper 文件](https://github.com/guillaumekln/faster-whisper)
- [Silero VAD 文件](https://github.com/snakers4/silero-vad)
- [CUDA 最佳實踐指南](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

### 學習資源
- [並行計算基礎](https://en.wikipedia.org/wiki/Parallel_computing)
- [Amdahl's Law](https://en.wikipedia.org/wiki/Amdahl%27s_law)
- [Python multiprocessing](https://docs.python.org/3/library/multiprocessing.html)

---

## 💬 獲取協助

如果文件中沒有找到需要的資訊：

1. 📖 查看主 [README.md](../README.zh-TW.md)
2. 🔍 使用 GitHub Issues 搜尋
3. 💡 提出新的 Issue
4. 📧 聯繫維護者

---

## 📝 文件維護

文件版本: **v2.0.0**  
最後更新: **2025-01-03**  

如發現文件錯誤或有改進建議，歡迎提交 Pull Request！
