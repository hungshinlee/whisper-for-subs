#!/bin/bash

# 單 GPU 模式修復 - 快速部署腳本
# 確保單 GPU 模式只使用 GPU 0

echo "🔧 單 GPU 模式修復"
echo "================================"
echo ""
echo "修改內容："
echo "  1. ✅ get_transcriber() 明確使用 cuda:0"
echo "  2. ✅ 更新進度提示為 'GPU 0'"
echo "  3. ✅ 狀態顯示為 'GPU 0 (single)'"
echo ""

cd /Users/winston/Projects/whisper-for-subs

echo "📋 檢查修改..."
if grep -q "cuda:0" app.py; then
    echo "✅ 修改已應用"
else
    echo "❌ 修改未應用"
    exit 1
fi
echo ""

# 詢問是否重建容器
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
    echo "  2. 上傳音訊並 **取消勾選** Multi-GPU"
    echo "  3. 點擊 Start 並查看日誌"
    echo "  4. 應該看到 'Loading Whisper model on GPU 0...'"
    echo "  5. 使用 nvidia-smi 確認只有 GPU 0 在使用"
    echo ""
    echo "💡 監控 GPU："
    echo "  watch -n 1 nvidia-smi"
    echo ""
    
else
    echo ""
    echo "⏭️  跳過容器重建"
    echo ""
    echo "稍後請手動執行："
    echo "  docker compose down"
    echo "  docker compose build"
    echo "  docker compose up -d"
fi
