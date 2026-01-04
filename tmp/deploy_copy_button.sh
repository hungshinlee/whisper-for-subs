#!/bin/bash

# SRT 複製按鈕功能部署腳本

echo "📋 SRT 複製按鈕功能部署"
echo "================================"
echo ""
echo "功能：在 SRT 輸出區域添加一鍵複製按鈕"
echo ""
echo "改進："
echo "  ✅ 添加 📋 Copy to Clipboard 按鈕"
echo "  ✅ 即時複製反饋"
echo "  ✅ 使用 Clipboard API"
echo "  ✅ 支援所有現代瀏覽器"
echo ""

cd /Users/winston/Projects/whisper-for-subs

# 檢查 app.py 是否已更新
echo "📋 檢查修改..."
if ! grep -q "Copy to Clipboard" app.py; then
    echo "❌ 錯誤: app.py 未包含複製按鈕"
    exit 1
fi

if ! grep -q "copy-button" app.py; then
    echo "❌ 錯誤: CSS 樣式未添加"
    exit 1
fi

echo "✅ 修改已確認"
echo ""

# 顯示改動摘要
echo "📝 改動摘要："
echo "  1. CSS - 添加 .copy-button 和 .copy-success 樣式"
echo "  2. UI - 在 SRT 輸出下方添加複製按鈕和狀態提示"
echo "  3. JavaScript - 實現剪貼簿複製功能"
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
    echo "  2. 上傳音訊並轉錄"
    echo "  3. 在 SRT 輸出下方應該看到 '📋 Copy to Clipboard' 按鈕"
    echo "  4. 點擊按鈕"
    echo "  5. 應該看到 '✅ Copied to clipboard!' 提示"
    echo "  6. 在記事本中按 Ctrl+V 測試貼上"
    echo ""
    echo "🎯 預期結果："
    echo "  • 按鈕出現在 SRT 文字框下方"
    echo "  • 點擊後立即顯示成功提示"
    echo "  • 可以在任何地方貼上 SRT 內容"
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
echo "  請參閱 tmp/COPY_BUTTON.md"
echo ""
