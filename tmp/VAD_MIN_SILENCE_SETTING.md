# VAD Min Silence Duration 設定功能

## 🎯 功能說明

在 Web 介面中添加了 **VAD: Min Silence Duration (seconds)** 滑桿，讓使用者可以自訂 VAD（語音活動檢測）的最小靜音時長參數。

### 什麼是 Min Silence Duration？

**Min Silence Duration**（最小靜音時長）是 VAD 用來決定何時分割語音段落的關鍵參數：

- **作用**：當檢測到靜音超過此時長時，VAD 會將語音分成兩個獨立的段落
- **單位**：秒（在 UI 中）/ 毫秒（在內部）
- **預設值**：0.1 秒（100 毫秒）

---

## 📊 參數影響

### 值太小（0.01 - 0.05 秒）

**效果**：
- ✅ 更精確的語音切分
- ✅ 捕捉更多短暫停頓
- ❌ 產生非常多的小段落
- ❌ 可能在句子中間切斷
- ❌ 處理效率較低

**適用場景**：
- 快速對話
- 辯論或訪談
- 需要非常精細的時間軸

### 預設值（0.1 秒）

**效果**：
- ✅ 平衡的切分
- ✅ 適合大多數情況
- ✅ 合理的段落數量
- ✅ 良好的處理效率

**適用場景**：
- 一般演講
- 訪談節目
- 教學影片
- 會議記錄

### 值太大（0.5 - 2.0 秒）

**效果**：
- ✅ 較少的段落數量
- ✅ 更長的連續段落
- ❌ 可能錯過自然停頓
- ❌ 段落過長不易閱讀
- ✅ 處理效率較高

**適用場景**：
- 正式演講（停頓較明顯）
- 有聲書
- 單人獨白
- 需要長段落的場景

---

## 🎛️ UI 元件

### 滑桿設定

```python
gr.Slider(
    minimum=0.01,      # 最小值：10 毫秒
    maximum=2.0,       # 最大值：2 秒
    value=0.1,         # 預設值：0.1 秒
    step=0.01,         # 步進：0.01 秒
    label="VAD: Min Silence Duration (seconds)",
    info="Minimum silence duration to split segments (default: 0.1s)",
)
```

### 動態顯示/隱藏

- 當 **Enable VAD** 勾選時：顯示滑桿
- 當 **Enable VAD** 取消勾選時：隱藏滑桿（因為不使用 VAD 就不需要這個參數）

---

## 📝 使用範例

### 範例 1：快速對話（短停頓）

**設定**：
- Enable VAD: ✅
- Min Silence Duration: **0.05 秒**

**效果**：
```
1
00:00:00,000 --> 00:00:01,500
嗨！

2
00:00:01,600 --> 00:00:02,800
你好嗎？

3
00:00:02,900 --> 00:00:04,200
很好！
```

**特點**：捕捉到所有短暫停頓，段落較多

---

### 範例 2：正常對話（預設）

**設定**：
- Enable VAD: ✅
- Min Silence Duration: **0.1 秒**（預設）

**效果**：
```
1
00:00:00,000 --> 00:00:04,200
嗨！你好嗎？很好！

2
00:00:04,500 --> 00:00:08,300
今天天氣真不錯。
```

**特點**：平衡的切分，適合大多數情況

---

### 範例 3：正式演講（長停頓）

**設定**：
- Enable VAD: ✅
- Min Silence Duration: **0.5 秒**

**效果**：
```
1
00:00:00,000 --> 00:00:15,800
各位女士先生，大家好。今天我要跟大家分享的主題是人工智慧的未來發展。

2
00:00:16,500 --> 00:00:32,100
首先，讓我們回顧一下人工智慧的歷史...
```

**特點**：只在明顯停頓處切分，段落較長

---

## 🔧 技術實現

### 1. 修改 transcriber.py

```python
class WhisperTranscriber:
    def __init__(
        self,
        ...
        min_silence_duration_ms: int = 100,  # 新增參數
    ):
        if use_vad:
            self.vad = SileroVAD(
                threshold=vad_threshold,
                min_silence_duration_ms=min_silence_duration_ms,  # 傳遞參數
            )
```

### 2. 修改 parallel_transcriber.py

```python
class ParallelWhisperTranscriber:
    def __init__(
        self,
        ...
        min_silence_duration_ms: int = 100,  # 新增參數
    ):
        self.vad = SileroVAD(
            threshold=vad_threshold,
            min_silence_duration_ms=min_silence_duration_ms,  # 傳遞參數
        )
```

### 3. 修改 app.py

#### 添加 UI 元件
```python
min_silence_slider = gr.Slider(
    minimum=0.01,
    maximum=2.0,
    value=0.1,
    step=0.01,
    label="VAD: Min Silence Duration (seconds)",
    info="Minimum silence duration to split segments (default: 0.1s)",
    visible=True,
)
```

#### 秒數轉毫秒
```python
def get_transcriber(
    ...
    min_silence_duration_s: float = 0.1,
):
    # Convert seconds to milliseconds
    min_silence_duration_ms = int(min_silence_duration_s * 1000)
    
    transcriber = WhisperTranscriber(
        ...
        min_silence_duration_ms=min_silence_duration_ms,
    )
```

#### 動態顯示/隱藏
```python
use_vad_checkbox.change(
    fn=lambda x: gr.update(visible=x),
    inputs=[use_vad_checkbox],
    outputs=[min_silence_slider],
)
```

---

## 📊 日誌輸出

### 單 GPU 模式

```bash
🎯 Single-GPU mode: Using GPU 0
Loading Whisper model: large-v3-turbo on cuda
✅ Model loaded successfully
Loading Silero VAD (min_silence_duration=50ms)...    ← 顯示設定值
✅ VAD loaded successfully
📊 Audio loaded: 180.5s
🎯 VAD detected 25 speech segments                   ← 更多段落（因為 50ms）
...
```

### 多 GPU 模式

```bash
Initialized ParallelWhisperTranscriber with 4 GPUs: [0, 1, 2, 3]
Using multiprocessing start method: spawn
💡 Using persistent workers (models loaded once per GPU)
Loading Silero VAD (min_silence_duration=200ms)...   ← 顯示設定值
📊 Audio loaded: 600.0s
🎯 VAD detected 89 speech segments                    ← 較少段落（因為 200ms）
✂️  Optimized to 35 segments for 4 GPUs
...
```

---

## 🚀 部署

```bash
cd /Users/winston/Projects/whisper-for-subs

# 重建容器
docker compose down
docker compose build
docker compose up -d

# 查看日誌
docker compose logs -f
```

---

## ✅ 測試步驟

### 1. 檢查 UI 元件

訪問 http://localhost:7860

**應該看到**：
- ✅ 在 VAD checkbox 下方有滑桿
- ✅ 標籤：「VAD: Min Silence Duration (seconds)」
- ✅ 預設值：0.1
- ✅ 範圍：0.01 - 2.0

### 2. 測試動態顯示

- **勾選** Enable VAD → 滑桿顯示
- **取消勾選** Enable VAD → 滑桿隱藏

### 3. 測試不同值

#### 測試 1：預設值（0.1 秒）
```
1. 設定 Min Silence Duration = 0.1
2. 上傳音訊
3. 觀察段落數量
```

#### 測試 2：小值（0.05 秒）
```
1. 設定 Min Silence Duration = 0.05
2. 上傳同一個音訊
3. 觀察段落數量應該增加
```

#### 測試 3：大值（0.5 秒）
```
1. 設定 Min Silence Duration = 0.5
2. 上傳同一個音訊
3. 觀察段落數量應該減少
```

### 4. 查看日誌

```bash
docker logs whisper-for-subs | grep "min_silence_duration"
```

**應該看到**：
```
Loading Silero VAD (min_silence_duration=XXXms)...
```

---

## 💡 使用建議

### 快速對話、辯論

```
建議值：0.03 - 0.08 秒
原因：捕捉頻繁的短暫停頓
```

### 一般對話、訪談

```
建議值：0.08 - 0.15 秒（預設）
原因：平衡的切分
```

### 演講、獨白

```
建議值：0.15 - 0.3 秒
原因：較長的自然停頓
```

### 有聲書、朗讀

```
建議值：0.3 - 0.8 秒
原因：明確的句子停頓
```

### 音樂背景的語音

```
建議值：0.1 - 0.2 秒
原因：避免被背景音樂干擾
```

---

## 🔍 故障排除

### 問題 1：滑桿沒有出現

**檢查**：
1. 確認 Enable VAD 已勾選
2. 確認 app.py 已更新

**解決**：
```bash
docker compose down
docker compose build
docker compose up -d
```

### 問題 2：改變值沒有效果

**檢查**：
1. 確認 VAD 已啟用
2. 查看日誌中的 min_silence_duration 值

**解決**：
```bash
# 查看日誌確認參數
docker logs whisper-for-subs | grep "min_silence"
```

### 問題 3：段落數量異常

**太多段落**：
- 值可能太小
- 嘗試增加到 0.15 - 0.2

**太少段落**：
- 值可能太大
- 嘗試減少到 0.05 - 0.1

---

## 📝 修改的檔案

### 1. transcriber.py
- 添加 `min_silence_duration_ms` 參數
- 傳遞給 SileroVAD

### 2. parallel_transcriber.py
- 添加 `min_silence_duration_ms` 參數
- 傳遞給 SileroVAD

### 3. app.py
- 添加 UI 滑桿元件
- 秒數轉毫秒邏輯
- 動態顯示/隱藏
- 傳遞參數給 transcriber

---

## 🎉 總結

### 新增功能
在 Web UI 中添加 VAD Min Silence Duration 設定

### 優勢
- ✅ 使用者可自訂切分靈敏度
- ✅ 適應不同類型的音訊
- ✅ 靈活控制段落數量
- ✅ 即時預覽效果

### 使用方法
Enable VAD → 調整滑桿（0.01 - 2.0 秒）→ 轉錄

### 建議值
- 快速對話：0.03 - 0.08 秒
- 一般對話：0.08 - 0.15 秒
- 演講獨白：0.15 - 0.3 秒
- 有聲書：0.3 - 0.8 秒

---

**立即部署，自訂 VAD 切分靈敏度！** 🎛️

```bash
docker compose down && docker compose build && docker compose up -d
```
