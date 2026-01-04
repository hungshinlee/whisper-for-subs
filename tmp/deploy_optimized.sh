#!/bin/bash

# 多 GPU 性能優化部署腳本

echo "🚀 多 GPU 性能優化 - 持久化 Worker 模式"
echo "========================================"
echo ""
echo "問題：每個 segment 都重新載入模型"
echo "解決：Worker 初始化時載入模型一次，重複使用"
echo ""
echo "預期提升："
echo "  • 10 分鐘音訊：122s → 46s（2.7倍）"
echo "  • 60 分鐘音訊：476s → 136s（3.5倍）"
echo ""

cd /Users/winston/Projects/whisper-for-subs

# 檢查優化版本是否存在
if [ ! -f "tmp/parallel_transcriber_optimized.py" ]; then
    echo "❌ 錯誤: 找不到 tmp/parallel_transcriber_optimized.py"
    exit 1
fi

echo "✅ 找到優化版本"
echo ""

# 備份當前版本
if [ -f "parallel_transcriber.py" ]; then
    BACKUP_FILE="parallel_transcriber.py.backup_slow_$(date +%Y%m%d_%H%M%S)"
    echo "📦 備份當前版本到: $BACKUP_FILE"
    cp parallel_transcriber.py "$BACKUP_FILE"
    echo "✅ 備份完成"
else
    echo "⚠️  警告: 原檔案不存在"
fi
echo ""

# 部署優化版本
echo "📝 部署優化版本..."
cp tmp/parallel_transcriber_optimized.py parallel_transcriber.py
echo "✅ 檔案已更新"
echo ""

# 顯示關鍵改進
echo "📋 關鍵改進："
echo "  ✅ Worker initializer - 每個 GPU worker 啟動時載入模型一次"
echo "  ✅ 全局變數 - 存儲模型實例供重複使用"
echo "  ✅ 獨立 executors - 每個 GPU 一個 executor 確保 worker 持久化"
echo "  ✅ 無重複載入 - 後續 segments 直接使用已載入的模型"
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
    echo "⏳ 等待容器啟動 (15 秒)..."
    sleep 15
    
    echo ""
    echo "📊 容器狀態："
    docker ps | grep whisper-for-subs
    
    echo ""
    echo "✅ 部署完成！"
    echo ""
    echo "📋 測試步驟："
    echo "  1. 訪問 http://localhost:7860"
    echo "  2. 上傳 10 分鐘以上的音訊"
    echo "  3. **勾選** 'Use Multi-GPU Parallel Processing'"
    echo "  4. 點擊 Start"
    echo ""
    echo "📝 觀察日誌（按 Ctrl+C 停止）："
    echo ""
    sleep 2
    
    echo "應該看到："
    echo "  ✅ [GPU 0] 🔧 Initializing worker... (開始時)"
    echo "  ✅ [GPU 0] ✅ Worker initialized and ready"
    echo "  ✅ [GPU 0] ▶ Processing segment 0"
    echo "  ✅ [GPU 0] ✓ Segment 0 complete"
    echo "  ✅ [GPU 0] ▶ Processing segment 4  ← 重複使用模型，無再載入！"
    echo ""
    echo "不應該看到："
    echo "  ❌ 每個 segment 都有 'Model loaded successfully'"
    echo ""
    echo "查看即時日誌..."
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
    echo ""
    echo "查看日誌："
    echo "  docker compose logs -f"
fi
