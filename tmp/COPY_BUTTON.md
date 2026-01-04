# SRT 複製按鈕功能

## 🎯 功能說明

在 **SRT Subtitle Content** 文字框下方添加了一個 **📋 Copy to Clipboard** 按鈕，可以一鍵複製 SRT 字幕內容到剪貼簿。

---

## ✨ 功能特點

### 1. 一鍵複製
- 點擊按鈕即可複製所有 SRT 內容
- 無需手動選取文字
- 適用於長字幕內容

### 2. 即時反饋
- **✅ Copied to clipboard!** - 複製成功
- **⚠️ No content to copy** - 沒有內容可複製
- **❌ Failed to copy** - 複製失敗（罕見）

### 3. 瀏覽器支援
- 使用現代瀏覽器的 Clipboard API
- 支援所有主流瀏覽器：
  - ✅ Chrome/Edge
  - ✅ Firefox
  - ✅ Safari
  - ✅ Opera

---

## 🖼️ UI 設計

### 按鈕位置
```
┌─────────────────────────────────┐
│ SRT Subtitle Content            │
│                                 │
│ 1                               │
│ 00:00:00,000 --> 00:00:02,500   │
│ This is the first subtitle.     │
│                                 │
│ 2                               │
│ 00:00:02,500 --> 00:00:05,000   │
│ This is the second subtitle.    │
│                                 │
└─────────────────────────────────┘
┌──────────────────┬──────────────┐
│ 📋 Copy to       │ ✅ Copied to │
│    Clipboard     │   clipboard! │
└──────────────────┴──────────────┘
┌─────────────────────────────────┐
│ Download SRT File               │
└─────────────────────────────────┘
```

### 樣式特點
- 📋 圖示清楚表示複製功能
- 綠色提示表示成功
- 與其他按鈕風格一致

---

## 🚀 使用方法

### 步驟 1：完成轉錄
上傳音訊並等待轉錄完成，SRT 內容會顯示在文字框中

### 步驟 2：點擊複製按鈕
點擊 **📋 Copy to Clipboard** 按鈕

### 步驟 3：確認成功
看到 **✅ Copied to clipboard!** 提示

### 步驟 4：貼上使用
在任何地方按 `Ctrl+V` (Windows/Linux) 或 `Cmd+V` (Mac) 貼上

---

## 💡 使用場景

### 場景 1：快速分享
轉錄完成後，直接複製貼到聊天軟體或郵件中

### 場景 2：編輯字幕
複製到字幕編輯軟體（如 Subtitle Edit）進行調整

### 場景 3：文字處理
貼到 Word、Google Docs 進行進一步編輯

### 場景 4：翻譯校對
貼到翻譯工具進行校對或翻譯

---

## 🔧 技術實現

### 使用的技術

**Clipboard API**
```javascript
navigator.clipboard.writeText(content)
```

**Gradio JavaScript 回調**
```python
copy_btn.click(
    fn=None,  # 不需要 Python 函數
    inputs=[srt_output],  # 取得 SRT 內容
    outputs=[copy_status],  # 更新狀態提示
    js="..."  # JavaScript 代碼
)
```

### 錯誤處理

```javascript
navigator.clipboard.writeText(srt_content).then(
    () => {
        // 成功
        return "✅ Copied to clipboard!";
    },
    (err) => {
        // 失敗
        return "❌ Failed to copy: " + err;
    }
);
```

### 內容驗證

```javascript
if (!srt_content) {
    return "⚠️ No content to copy";
}
```

---

## 📋 狀態訊息

### 成功狀態
```
✅ Copied to clipboard!
```
- 顯示為綠色
- 表示內容已成功複製

### 警告狀態
```
⚠️ No content to copy
```
- 顯示為橙色（預設）
- 表示文字框為空

### 錯誤狀態
```
❌ Failed to copy: [錯誤訊息]
```
- 顯示為紅色（預設）
- 表示瀏覽器不支援或權限問題

---

## 🐛 故障排除

### 問題 1：複製按鈕沒有出現

**檢查**：
1. 確認 `app.py` 已更新
2. 重啟容器

**解決**：
```bash
docker compose down
docker compose build
docker compose up -d
```

### 問題 2：點擊後沒有反應

**可能原因**：
- 瀏覽器不支援 Clipboard API
- HTTPS 問題（本地 localhost 通常沒問題）
- 瀏覽器權限被拒絕

**解決**：
1. 使用現代瀏覽器（Chrome/Firefox）
2. 檢查瀏覽器控制台錯誤訊息
3. 確認瀏覽器剪貼簿權限

### 問題 3：顯示「Failed to copy」

**可能原因**：
- 瀏覽器安全限制
- 需要 HTTPS（生產環境）

**解決**：
- 本地開發通常沒問題（localhost 被視為安全來源）
- 生產環境需要使用 HTTPS

---

## 🌐 瀏覽器相容性

### 完全支援 ✅
- Chrome 66+
- Edge 79+
- Firefox 63+
- Safari 13.1+
- Opera 53+

### 部分支援 ⚠️
- 舊版瀏覽器可能需要用戶手動選取複製

### 不支援 ❌
- IE 11 及更早版本

---

## 🎨 自訂樣式

如果要調整按鈕外觀，可以修改 CSS：

```python
# app.py 中的 CUSTOM_CSS
.copy-button {
    margin-top: 10px;
    /* 可以添加更多樣式 */
}

.copy-success {
    color: #4CAF50;  /* 成功提示顏色 */
    font-weight: 500;
    margin-top: 5px;
}
```

---

## 📊 與下載功能對比

| 功能 | 複製按鈕 | 下載按鈕 |
|-----|---------|---------|
| 速度 | 即時 | 需要下載 |
| 用途 | 快速分享/編輯 | 保存文件 |
| 格式 | 純文字 | .srt 文件 |
| 適用 | 短時間使用 | 長期保存 |

**建議**：
- 快速分享或編輯 → 使用複製按鈕
- 保存檔案或批量處理 → 使用下載按鈕

---

## 🚀 部署

### 方法 1：自動部署

```bash
cd /Users/winston/Projects/whisper-for-subs

# 重建容器
docker compose down
docker compose build
docker compose up -d
```

### 方法 2：使用部署腳本

```bash
cd /Users/winston/Projects/whisper-for-subs
bash tmp/deploy_copy_button.sh
```

---

## ✅ 驗證功能

### 測試步驟

1. **訪問 Web UI**
   ```
   http://localhost:7860
   ```

2. **上傳音訊並轉錄**
   - 等待 SRT 內容顯示

3. **點擊複製按鈕**
   - 應該看到 "📋 Copy to Clipboard" 按鈕
   - 點擊後應該看到 "✅ Copied to clipboard!"

4. **測試貼上**
   - 在記事本或任何文字編輯器中
   - 按 `Ctrl+V` (或 `Cmd+V`)
   - 應該貼上完整的 SRT 內容

---

## 📝 修改的檔案

只修改了一個檔案：
- ✅ `app.py` - 添加複製按鈕和相關功能

修改內容：
1. CSS 樣式（.copy-button 和 .copy-success）
2. UI 元件（複製按鈕和狀態提示）
3. JavaScript 回調函數（複製功能）

---

## 💡 進階功能（未來可能添加）

### 可能的改進
- [ ] 複製特定時間範圍的字幕
- [ ] 複製為不同格式（VTT、ASS 等）
- [ ] 複製時自動格式化（去除時間碼等）
- [ ] 複製統計信息（字數、段落數等）

---

## 🎉 總結

### 功能
在 SRT 輸出區域添加一鍵複製按鈕

### 優勢
- ✅ 快速複製
- ✅ 即時反饋
- ✅ 無需手動選取
- ✅ 適用所有瀏覽器
- ✅ 簡單易用

### 使用方法
轉錄完成 → 點擊 📋 Copy → 看到 ✅ 提示 → 貼上使用

---

**立即部署，享受一鍵複製的便利！** 📋

```bash
docker compose down && docker compose build && docker compose up -d
```
