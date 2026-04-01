#!/bin/bash

# ClassEye & FaceHub 资产同步脚本
# 请在项目根目录下运行 (ultralytics-ClassEye/)

PROJECT1_MODELS="../insightFaceEmpowerment/app/models"
HUB_MODELS="./face_hub/models"
ROOT_MODELS="./models"

echo "--- 开始同步核心资产 ---"

# 1. 优先同步到根目录 models/ (YOLO 需要)
echo "[1/3] 同步 YOLO 模型到根目录 models/..."
mkdir -p "$ROOT_MODELS"
cp -r "$PROJECT1_MODELS/"* "$ROOT_MODELS/"

# 2. 同步到 face_hub/models/ (FaceService 会在此处嗅探)
echo "[2/3] 同步人脸模型到包目录 face_hub/models/..."
mkdir -p "$HUB_MODELS"
cp -r "$PROJECT1_MODELS/"* "$HUB_MODELS/"

# 3. 复制 .env 配置
if [ -f "../insightFaceEmpowerment/app/.env" ]; then
    echo "[3/3] 正在同步数据库配置 (.env)..."
    cp "../insightFaceEmpowerment/app/.env" "./.env"
fi

echo "--- 同步成功 ---"
echo "你可以运行 python web/app.py 启动集成服务了。"
