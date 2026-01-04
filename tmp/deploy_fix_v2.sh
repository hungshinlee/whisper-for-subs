#!/bin/bash

# 單 GPU 模式修復 v2 - 正確版本部署腳本

echo "🔧 單 GPU 模式修復 v2 - 正確版本"
echo "================================"
echo ""
echo "修復內容："
echo "  ✅ 使用 torch.cuda.set_device(0)"
echo "  ✅ device='cuda' (不是 'cuda:0')"
echo "  ✅ 符合 faster-whisper API"
echo ""

cd /Users/winston/Projects/whisper-for-subs

echo "📋 檢查修改..."
if grep -q "torch.cuda.set_device(0)" app.py; then
    echo "✅ 修改已應用（v2 正確版本）"
else
    echo "❌ 修改未應用"
    exit 1
fi

if grep -q "cuda:0" app.py; then
    echo "⚠️  警告: 仍包含 'cuda:0' 字串"
    echo "請確認是否在註解中"
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
    echo "🔨 重新建置容器（無快取）..."
    docker compose build --no-cache
    
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
    echo "🔍 檢查啟動日誌..."
    docker logs whisper-for-subs 2>&1 | tail -20
    
    echo ""
    echo "✅ 部署完成！"
    echo ""
    echo "📋 驗證步驟："
    echo "  1. 訪問 http://localhost:7860"
    echo "  2. 上傳短音訊並取消勾選 Multi-GPU"
    echo "  3. 點擊 Start"
    echo "  4. 應該看到成功載入（無錯誤）"
    echo ""
    echo "  預期日誌："
    echo "    ✅ Loading Whisper model: large-v3-turbo on cuda"
    echo ""
    echo "  不應該看到："
    echo "    ❌ ValueError: unsupported device cuda:0"
    echo ""
    echo "💡 監控 GPU："
    echo "  watch -n 1 nvidia-smi"
    echo ""
    echo "📝 查看完整日誌："
    echo "  docker logs -f whisper-for-subs"
    echo ""
    
else
    echo ""
    echo "⏭️  跳過容器重建"
    echo ""
    echo "稍後請手動執行："
    echo "  docker compose down"
    echo "  docker compose build --no-cache"
    echo "  docker compose up -d"
fi
