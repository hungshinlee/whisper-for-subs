#!/bin/bash

# 多 GPU 功能改進版本 - 快速部署腳本
# 使用方法: bash tmp/deploy_improvement.sh

set -e  # 遇到錯誤立即停止

echo "🚀 開始部署多 GPU 功能改進版本..."
echo ""

# 1. 檢查改進版本檔案是否存在
if [ ! -f "tmp/parallel_transcriber_improved.py" ]; then
    echo "❌ 錯誤: 找不到 tmp/parallel_transcriber_improved.py"
    exit 1
fi

echo "✅ 找到改進版本檔案"
echo ""

# 2. 備份原檔案
if [ -f "parallel_transcriber.py" ]; then
    BACKUP_FILE="parallel_transcriber.py.backup.$(date +%Y%m%d_%H%M%S)"
    echo "📦 備份原檔案到 $BACKUP_FILE..."
    cp parallel_transcriber.py "$BACKUP_FILE"
    echo "✅ 備份完成"
else
    echo "⚠️  警告: 原檔案不存在，跳過備份"
fi
echo ""

# 3. 複製改進版本
echo "📝 複製改進版本..."
cp tmp/parallel_transcriber_improved.py parallel_transcriber.py
echo "✅ 檔案已更新"
echo ""

# 4. 確認是否要重新建置容器
read -p "是否要立即重新建置並啟動 Docker 容器？(y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "🐳 停止舊容器..."
    docker compose down
    
    echo ""
    echo "🔨 重新建置容器..."
    docker compose build
    
    echo ""
    echo "🚀 啟動新容器..."
    docker compose up -d
    
    echo ""
    echo "⏳ 等待容器啟動 (10 秒)..."
    sleep 10
    
    echo ""
    echo "📊 檢查容器狀態..."
    docker ps | grep whisper-for-subs
    
    echo ""
    echo "📝 顯示最近的日誌 (按 Ctrl+C 停止)..."
    echo ""
    docker compose logs -f --tail=50
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
echo "✅ 部署完成！"
echo ""
echo "📋 後續步驟："
echo "  1. 訪問 http://your-server:7860"
echo "  2. 上傳測試音訊 (建議 10-30 分鐘)"
echo "  3. 勾選 '🚀 Use Multi-GPU Parallel Processing'"
echo "  4. 觀察日誌: docker compose logs -f"
echo ""
