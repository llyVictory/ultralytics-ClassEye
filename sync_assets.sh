#!/bin/bash

# ClassEye & InsightFace 集成同步脚本
# 请在 /home/lly/project/VISION/ultralytics-ClassEye 目录下运行

PROJECT1_ROOT="../insightFaceEmpowerment/app"
PROJECT2_ROOT="."

echo "--- 开始同步核心资产 ---"

# 1. 复制模型文件夹
echo "[1/2] 正在复制模型文件到 models/ 目录..."
mkdir -p "$PROJECT2_ROOT/models/buffalo_sc"
cp -r "$PROJECT1_ROOT/models/"* "$PROJECT2_ROOT/models/"

# 2. 复制配置文件 (如 .env)
if [ -f "$PROJECT1_ROOT/.env" ]; then
    echo "[2/2] 正在复制数据库配置 .env 文件..."
    cp "$PROJECT1_ROOT/.env" "$PROJECT2_ROOT/.env"
else
    echo "[!] 警告: 未在项目 1 中找到 .env 文件，请手动配置数据库连接。"
fi

echo "--- 同步完成 ---"
echo "请确保已安装 requirements.txt 中的依赖。"
