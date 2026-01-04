#!/bin/bash

# 中文簡繁轉換功能部署腳本

echo "🇹🇼 中文簡繁轉換功能部署"
echo "================================"
echo ""
echo "功能：選擇 zh 語言時，自動將簡體中文轉換成繁體中文"
echo ""
echo "改進："
echo "  ✅ 添加 OpenCC 依賴"
echo "  ✅ 創建 chinese_converter.py 模組"
echo "  ✅ 整合到 app.py（單 GPU）"
echo "  ✅ 整合到 parallel_transcriber.py（多 GPU）"
echo ""

cd /Users/winston/Projects/whisper-for-subs

# 檢查文件是否存在
echo "📋 檢查文件..."
if [ ! -f "chinese_converter.py" ]; then
    echo "❌ 錯誤: chinese_converter.py 不存在"
    exit 1
fi

if ! grep -q "opencc-python-reimplemented" requirements.txt; then
    echo "❌ 錯誤: requirements.txt 未更新"
    exit 1
fi

if ! grep -q "chinese_converter" app.py; then
    echo "❌ 錯誤: app.py 未整合轉換功能"
    exit 1
fi

if ! grep -q "chinese_converter" parallel_transcriber.py; then
    echo "❌ 錯誤: parallel_transcriber.py 未整合轉換功能"
    exit 1
fi

echo "✅ 所有文件就緒"
echo ""

# 顯示改動摘要
echo "📝 改動摘要："
echo "  1. requirements.txt - 添加 opencc-python-reimplemented"
echo "  2. chinese_converter.py - 新建轉換模組"
echo "  3. app.py - 轉錄後自動轉換（語言=zh）"
echo "  4. parallel_transcriber.py - 多 GPU 模式支持"
echo ""

# 詢問是否重建
read -p "是否立即重新建置並啟動容器？(y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "🐳 停止容器..."
    docker compose down
    
    echo ""
    echo "🔨 重新建置容器（會安裝 OpenCC）..."
    docker compose build --no-cache
    
    echo ""
    echo "🚀 啟動容器..."
    docker compose up -d
    
    echo ""
    echo "⏳ 等待容器啟動 (15 秒)..."
    sleep 15
    
    echo ""
    echo "🔍 驗證 OpenCC 安裝..."
    if docker exec whisper-for-subs python -c "from opencc import OpenCC; print('✅ OpenCC installed')" 2>/dev/null; then
        echo "✅ OpenCC 安裝成功"
    else
        echo "❌ OpenCC 安裝失敗"
        echo "請檢查建置日誌"
        exit 1
    fi
    
    echo ""
    echo "🔍 驗證轉換器..."
    if docker exec whisper-for-subs python -c "from chinese_converter import ChineseConverter; c = ChineseConverter(); print('✅' if c.is_available() else '❌')" 2>/dev/null; then
        echo "✅ 轉換器初始化成功"
    else
        echo "❌ 轉換器初始化失敗"
    fi
    
    echo ""
    echo "📊 容器狀態："
    docker ps | grep whisper-for-subs
    
    echo ""
    echo "✅ 部署完成！"
    echo ""
    echo "📋 測試步驟："
    echo "  1. 訪問 http://localhost:7860"
    echo "  2. Language 選擇 'zh' (Chinese)"
    echo "  3. 上傳中文音訊或 YouTube 連結"
    echo "  4. 點擊 Start"
    echo "  5. 觀察輸出是否為繁體中文"
    echo ""
    echo "🔍 測試轉換器："
    echo "  docker exec whisper-for-subs python /app/chinese_converter.py"
    echo ""
    echo "📝 查看日誌："
    echo "  docker compose logs -f"
    echo ""
    echo "預期看到："
    echo "  🔄 Converting to Traditional Chinese..."
    echo "  ✅ Converted to Traditional Chinese"
    echo ""
    
    # 詢問是否要測試轉換器
    read -p "是否立即測試轉換器？(y/n): " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "🧪 測試轉換器..."
        echo ""
        docker exec whisper-for-subs python /app/chinese_converter.py
    fi
    
else
    echo ""
    echo "⏭️  跳過容器重建"
    echo ""
    echo "稍後請手動執行："
    echo "  docker compose down"
    echo "  docker compose build --no-cache"
    echo "  docker compose up -d"
    echo ""
    echo "驗證安裝："
    echo "  docker exec whisper-for-subs python -c \"from opencc import OpenCC; print('OK')\""
fi

echo ""
echo "📚 詳細說明："
echo "  請參閱 tmp/CHINESE_CONVERSION.md"
echo ""
