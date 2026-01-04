#!/bin/bash

# VAD Min Silence Duration 設定功能部署腳本

echo "🎛️ VAD Min Silence Duration 設定功能部署"
echo "=========================================="
echo ""
echo "功能：在 Web UI 中添加 VAD 最小靜音時長設定"
echo ""
echo "改進："
echo "  ✅ 添加滑桿讓使用者輸入秒數（0.01 - 2.0）"
echo "  ✅ 自動轉換秒數為毫秒"
echo "  ✅ 預設值 0.1 秒（100 毫秒）"
echo "  ✅ 動態顯示/隱藏（根據 VAD 啟用狀態）"
echo ""

cd /Users/winston/Projects/whisper-for-subs

# 檢查修改
echo "📋 檢查修改..."

if ! grep -q "min_silence_duration_ms" transcriber.py; then
    echo "❌ 錯誤: transcriber.py 未更新"
    exit 1
fi

if ! grep -q "min_silence_duration_ms" parallel_transcriber.py; then
    echo "❌ 錯誤: parallel_transcriber.py 未更新"
    exit 1
fi

if ! grep -q "min_silence_slider" app.py; then
    echo "❌ 錯誤: app.py 未添加 UI 元件"
    exit 1
fi

echo "✅ 所有修改已確認"
echo ""

# 顯示改動摘要
echo "📝 改動摘要："
echo "  1. transcriber.py - 接受 min_silence_duration_ms 參數"
echo "  2. parallel_transcriber.py - 接受 min_silence_duration_ms 參數"
echo "  3. app.py - 添加滑桿 UI 元件"
echo "  4. app.py - 秒數轉毫秒邏輯"
echo "  5. app.py - 動態顯示/隱藏滑桿"
echo ""

# 詢問是否重建
read -p "是否立即重新建置並啟動容器？(y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "🐳 停止容器..."
    docker compose down
    
    echo ""
    echo "🔨 重新建置容器..."
    docker compose build
    
    echo ""
    echo "🚀 啟動容器..."
    docker compose up -d
    
    echo ""
    echo "⏳ 等待容器啟動 (10 秒)..."
    sleep 10
    
    echo ""
    echo "📊 容器狀態："
    docker ps | grep whisper-for-subs
    
    echo ""
    echo "✅ 部署完成！"
    echo ""
    echo "📋 測試步驟："
    echo "  1. 訪問 http://localhost:7860"
    echo "  2. 在 Settings 區域勾選 'Enable VAD'"
    echo "  3. 應該看到 'VAD: Min Silence Duration (seconds)' 滑桿"
    echo "  4. 預設值應該是 0.1"
    echo "  5. 可以調整範圍：0.01 - 2.0"
    echo ""
    echo "🎯 測試不同值："
    echo "  • 0.05 秒 - 更多段落（捕捉短暫停頓）"
    echo "  • 0.1 秒  - 預設（平衡）"
    echo "  • 0.5 秒  - 較少段落（只在明顯停頓處切分）"
    echo ""
    echo "📝 查看日誌："
    echo "  docker logs whisper-for-subs | grep 'min_silence_duration'"
    echo ""
    echo "預期看到："
    echo "  Loading Silero VAD (min_silence_duration=XXXms)..."
    echo ""
    
    # 詢問是否打開瀏覽器
    read -p "是否立即在瀏覽器中打開？(y/n): " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🌐 打開瀏覽器..."
        
        # 偵測作業系統並打開瀏覽器
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            open http://localhost:7860
        elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
            # Linux
            xdg-open http://localhost:7860 2>/dev/null || echo "請手動訪問 http://localhost:7860"
        else
            # Windows (Git Bash)
            start http://localhost:7860 2>/dev/null || echo "請手動訪問 http://localhost:7860"
        fi
    fi
    
else
    echo ""
    echo "⏭️  跳過容器重建"
    echo ""
    echo "稍後請手動執行："
    echo "  docker compose down"
    echo "  docker compose build"
    echo "  docker compose up -d"
fi

echo ""
echo "📚 詳細說明："
echo "  請參閱 tmp/VAD_MIN_SILENCE_SETTING.md"
echo ""
echo "💡 使用建議："
echo "  • 快速對話：0.03 - 0.08 秒"
echo "  • 一般對話：0.08 - 0.15 秒（預設）"
echo "  • 演講獨白：0.15 - 0.3 秒"
echo "  • 有聲書：0.3 - 0.8 秒"
echo ""
