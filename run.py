import os
import cv2
import logging
from argparse import ArgumentParser
from classeye.detector import ClassEyeDetector
from classeye.row_splitter import RowSplitter
from classeye.visualizer import ClassEyeVisualizer
from classeye.config import config

# 配置日志记录
logging.basicConfig(level=logging.INFO, format="%(message)s")

def process_image(img_path, detector, splitter, visualizer, output_dir):
    """
    处理单张图片并保存结果。
    """
    if not os.path.exists(img_path):
        logging.error(f"[ClassEye] 为找到文件: {img_path}")
        return
    
    # 1. 载入原始图片
    img = cv2.imread(img_path)
    if img is None:
        logging.error(f"[ClassEye] 读取图片失败: {img_path}")
        return
    
    img_h, img_w = img.shape[:2]

    # 2. 人头检测
    logging.info(f"--- 正在处理 {os.path.basename(img_path)} ---")
    results = detector.detect_heads(img) # 给模型输入原图
    
    if len(results) == 0:
        logging.warning("无检测结果。")
        return

    # 3. 自动分排
    boxes = results[0].boxes
    rows_info = splitter.split_rows(boxes, img_w, img_h)

    # 4. 可视化绘制
    canvas = visualizer.draw_results(img, results, rows_info)

    # 5. 保存结果与打印汇总
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    file_name = os.path.basename(img_path)
    save_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_result.jpg")
    cv2.imwrite(save_path, canvas)
    
    logging.info(f"[ClassEye] 处理完成，结果已保存至: {save_path}")
    print("-" * 30)
    total_count = 0
    for row in rows_info:
        print(f"[ClassEye] {row['name']}: {row['counts']} 人")
        total_count += row['counts']
    print(f"[ClassEye] 全场合计: {total_count} 人")
    print("-" * 30)

def main():
    parser = ArgumentParser(description="ClassEye 教室人头识别与统计系统")
    parser.add_argument("--source", type=str, default="samples/classroom_01.jpg", 
                        help="图片路径或文件夹目录")
    parser.add_argument("--conf", type=float, default=config.CONF_THRESHOLD, 
                        help="检测置信度")
    parser.add_argument("--eps", type=int, default=config.DB_EPS, 
                        help="DBSCAN 聚类 eps 参数")
    
    args = parser.parse_args()
    
    # 初始化核心组件
    detector = ClassEyeDetector(model_path=config.MODEL_PATH)
    splitter = RowSplitter(eps=args.eps)
    visualizer = ClassEyeVisualizer()

    # 处理输入 (支持单文件或文件夹批量处理)
    if os.path.isfile(args.source):
        process_image(args.source, detector, splitter, visualizer, config.OUTPUT_DIR)
    elif os.path.isdir(args.source):
        for f in os.listdir(args.source):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_p = os.path.join(args.source, f)
                process_image(img_p, detector, splitter, visualizer, config.OUTPUT_DIR)
    else:
        logging.error(f"[ClassEye] 无法识别输入源: {args.source}")

if __name__ == "__main__":
    main()
