#!/bin/bash

# CUDA 初始化錯誤 - 快速修復腳本
# 使用方法: bash tmp/fix_cuda_error.sh

set -e  # 遇到錯誤立即停止

echo "🔧 CUDA 初始化錯誤修復腳本"
echo "================================"
echo ""
echo "問題: RuntimeError: CUDA failed with error initialization error"
echo "原因: fork 模式與 CUDA 不兼容"
echo "解決: 使用 spawn 模式創建子進程"
echo ""

# 檢查修復檔案是否存在
if [ ! -f "tmp/parallel_transcriber_fixed.py" ]; then
    echo "❌ 錯誤: 找不到修復檔案 tmp/parallel_transcriber_fixed.py"
    exit 1
fi

echo "✅ 找到修復檔案"
echo ""

# 備份當前檔案
if [ -f "parallel_transcriber.py" ]; then
    BACKUP_FILE="parallel_transcriber.py.backup_cuda_$(date +%Y%m%d_%H%M%S)"
    echo "📦 備份當前檔案到: $BACKUP_FILE"
    cp parallel_transcriber.py "$BACKUP_FILE"
    echo "✅ 備份完成"
else
    echo "⚠️  警告: 原檔案不存在"
fi
echo ""

# 複製修復版本
echo "📝 部署修復版本..."
cp tmp/parallel_transcriber_fixed.py parallel_transcriber.py
echo "✅ 檔案已更新"
echo ""

# 顯示關鍵變更
echo "📋 關鍵修復內容："
echo "  1. ✅ 設置 multiprocessing.set_start_method('spawn')"
echo "  2. ✅ 使用 spawn 上下文創建進程池"
echo "  3. ✅ 確保 CUDA 在每個進程中獨立初始化"
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
    echo "🔍 驗證修復..."
    docker exec whisper-for-subs python -c "
import multiprocessing
method = multiprocessing.get_start_method()
print(f'✅ Multiprocessing start method: {method}')
if method == 'spawn':
    print('✅ CUDA 兼容模式已啟用！')
else:
    print('⚠️  警告: 仍在使用', method, '模式')
" 2>/dev/null || echo "⚠️  容器尚未完全啟動"
    
    echo ""
    echo "📊 容器狀態："
    docker ps | grep whisper-for-subs
    
    echo ""
    echo "📝 查看日誌 (按 Ctrl+C 停止)..."
    echo ""
    sleep 2
    docker compose logs -f --tail=30
else
    echo ""
    echo "⏭️  跳過容器重建"
    echo ""
    echo "稍後請手動執行："
    echo "  cd /Users/winston/Projects/whisper-for-subs"
    echo "  docker compose down"
    echo "  docker compose build"
    echo "  docker compose up -d"
fi

echo ""
echo "✅ 修復部署完成！"
echo ""
echo "📋 驗證步驟："
echo "  1. 上傳測試音訊 (10-30 分鐘)"
echo "  2. 勾選 '🚀 Use Multi-GPU Parallel Processing'"
echo "  3. 觀察日誌確認無 CUDA 錯誤"
echo "  4. 應該看到成功的轉錄結果"
echo ""
echo "📖 更多資訊請查看: tmp/CUDA_FIX.md"
echo ""
