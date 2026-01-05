# Whisper ASR 文檔中心 📚

歡迎來到 Whisper ASR 字幕生成服務的文檔中心。這裡提供完整的使用指南、技術文檔和故障排除資源。

---

## 📖 快速導航

### 新手入門
- **[快速開始 (多 GPU)](./QUICKSTART_MULTI_GPU.md)** - 5 分鐘快速部署指南
- **[部署指南](./DEPLOYMENT_GUIDE.md)** - 詳細的安裝和配置步驟
- **[完整使用文檔](./FULL_DOCUMENTATION.md)** - 完整的功能說明和使用指南

### 核心文檔
- **[多 GPU 並行處理指南](./MULTI_GPU_GUIDE.md)** - 多 GPU 工作原理和性能優化
- **[故障排除](./TROUBLESHOOTING_MULTI_GPU.md)** - 常見問題和解決方案
- **[實作總結](./IMPLEMENTATION_SUMMARY.md)** - 技術實作細節和架構說明

### 其他資源
- **[更新日誌](./CHANGELOG.md)** - 版本歷史和更新記錄
- **[English Version](./README.en.md)** - 英文版文檔
- **[清理總結](./CLEANUP_SUMMARY.md)** - 代碼清理和文檔重組說明

---

## 📋 文檔分類

### 🚀 入門指南

#### [快速開始 (多 GPU)](./QUICKSTART_MULTI_GPU.md)
最快速的部署方式，適合：
- 第一次使用的用戶
- 需要快速部署的場景
- 基本配置和測試

**內容包含**：
- 系統要求檢查
- Docker 快速安裝
- 3 步驟啟動服務
- 基本測試

#### [部署指南](./DEPLOYMENT_GUIDE.md)
詳細的部署文檔，適合：
- 需要深入了解配置的用戶
- 生產環境部署
- 自定義配置需求

**內容包含**：
- 完整的系統需求
- 詳細的安裝步驟
- 環境變數配置
- 安全性設置
- 性能調優

---

### 📚 使用指南

#### [完整使用文檔](./FULL_DOCUMENTATION.md)
全面的使用說明，涵蓋：
- 詳細的功能介紹
- 各種輸入方式（音檔、影片、YouTube）
- 所有設定選項說明
- 中文簡繁轉換
- 日誌和監控
- API 使用範例

**適合**：
- 需要了解所有功能的用戶
- API 集成開發者
- 進階用戶

---

### ⚡ 性能優化

#### [多 GPU 並行處理指南](./MULTI_GPU_GUIDE.md)
深入了解多 GPU 並行處理，包含：
- 工作原理詳解
- 性能對比數據
- 最佳實踐
- 調優技巧
- 負載平衡策略

**適合**：
- 擁有多張 GPU 的用戶
- 需要處理大量音訊的場景
- 性能優化需求

---

### 🛠️ 故障排除

#### [故障排除指南](./TROUBLESHOOTING_MULTI_GPU.md)
常見問題和解決方案：
- GPU 相關問題
- 記憶體不足 (OOM)
- YouTube 下載失敗
- Port 衝突
- 網路問題
- 中文轉換問題

**適合**：
- 遇到錯誤的用戶
- 系統管理員
- 調試需求

---

### 🔧 技術文檔

#### [實作總結](./IMPLEMENTATION_SUMMARY.md)
技術實作細節：
- 系統架構
- 多 GPU 並行實現
- VAD 集成
- 簡繁轉換實現
- 性能優化技術

**適合**：
- 開發者
- 貢獻者
- 技術研究

---

### 📅 版本資訊

#### [更新日誌](./CHANGELOG.md)
版本更新記錄：
- 新功能
- Bug 修復
- 性能改進
- 已知問題

**適合**：
- 追蹤版本更新
- 了解新功能
- 升級參考

---

## 🌍 多語言文檔

### [English Version](./README.en.md)
Complete English documentation

---

## 📊 快速參考

### 性能表現

| 音訊長度 | 單 GPU | 多 GPU (4x) | 提升 |
|---------|--------|-------------|------|
| 10 分鐘 | 60 秒 | 23 秒 | 2.6x |
| 30 分鐘 | 180 秒 | 67 秒 | 2.7x |
| 60 分鐘 | 360 秒 | 136 秒 | 2.6x |

### 支援的模型

| 模型 | VRAM | 速度 | 推薦 |
|------|------|------|------|
| `large-v3-turbo` | ~6 GB | 快 ⚡ | ✅ **推薦** |
| `large-v3` | ~10 GB | 較慢 | 高品質需求 |
| `large-v2` | ~10 GB | 較慢 | 向下相容 |

### 常用命令

```bash
# 啟動服務
docker compose up -d

# 查看日誌
docker compose logs -f

# 重啟服務
docker compose restart

# 停止服務
docker compose down

# 更新映像
docker compose build
docker compose up -d
```

---

## 🔗 相關鏈接

- **主專案**: [GitHub Repository](https://github.com/hungshinlee/whisper-for-subs)
- **Issues**: [報告問題](https://github.com/hungshinlee/whisper-for-subs/issues)
- **OpenAI Whisper**: [官方文檔](https://github.com/openai/whisper)
- **faster-whisper**: [GitHub](https://github.com/guillaumekln/faster-whisper)
- **Silero VAD**: [GitHub](https://github.com/snakers4/silero-vad)

---

## 📞 獲取幫助

### 遇到問題？

1. **查看文檔**
   - 先檢查 [故障排除指南](./TROUBLESHOOTING_MULTI_GPU.md)
   - 查看 [完整使用文檔](./FULL_DOCUMENTATION.md)

2. **查看日誌**
   ```bash
   docker compose logs -f
   ```

3. **搜尋已知問題**
   - 查看 [GitHub Issues](https://github.com/hungshinlee/whisper-for-subs/issues)

4. **報告新問題**
   - 提供詳細的錯誤信息
   - 包含系統配置
   - 附上相關日誌

---

## 🤝 貢獻文檔

發現文檔中的錯誤或想要改進？

1. Fork 專案
2. 修改文檔
3. 提交 Pull Request

文檔貢獻指南：
- 保持語言清晰簡潔
- 提供實際範例
- 確保技術準確性
- 遵循現有格式

---

## 📝 文檔維護

**最後更新**: 2025-01-05  
**維護者**: 李鴻欣 (Hung-Shin Lee)  
**Email**: hungshinlee@gmail.com

---

**返回**: [主 README](../README.md)
