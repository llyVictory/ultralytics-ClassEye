import os
import sys
import cv2
import logging
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# 确保能找到上级目录的 classeye 包
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from classeye.detector import ClassEyeDetector
from classeye.row_splitter import RowSplitter
from classeye.visualizer import ClassEyeVisualizer
from classeye.config import config

app = Flask(__name__)

# 配置上传和输出路径
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(os.path.join(app.root_path, UPLOAD_FOLDER), exist_ok=True)
os.makedirs(os.path.join(app.root_path, RESULT_FOLDER), exist_ok=True)

# 初始化核心对象
detector = ClassEyeDetector(model_path=config.MODEL_PATH)
splitter = RowSplitter()
visualizer = ClassEyeVisualizer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify(success=False, message="文件未找到")
    
    file = request.files['file']
    if file.filename == '':
        return jsonify(success=False, message="未选择文件")

    if file:
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.root_path, UPLOAD_FOLDER, filename)
        file.save(save_path)

        # 获取手动划线参数 (从 JSON 里面拿，如果前端传了的话)
        manual_y_list = []
        try:
            # 兼容 JSON 请求或普通的 Form 请求
            req_data = request.get_json() if request.is_json else request.form
            if req_data and 'manual_y_list' in req_data:
                # 传入格式可能是字符串 "[0, 100...]" 或 list
                val = req_data['manual_y_list']
                manual_y_list = val if isinstance(val, list) else eval(val)
        except:
            manual_y_list = []

        # 进行 AI 处理
        try:
            img = cv2.imread(save_path)
            if img is None:
                return jsonify(success=False, message="无效的图片格式")
            
            img_h, img_w = img.shape[:2]

            # 1. 检测
            results = detector.detect_heads(img)
            
            # 2. 分排 (支持手动模式)
            boxes = results[0].boxes
            rows_info = splitter.split_rows(boxes, img_w, img_h, manual_y_list=manual_y_list)

            # 3. 渲染
            canvas = visualizer.draw_results(img, results, rows_info)

            # 4. 保存结果图
            res_filename = f"res_{filename}"
            res_path = os.path.join(app.root_path, RESULT_FOLDER, res_filename)
            cv2.imwrite(res_path, canvas)

            # 计算总人数
            total_count = sum([r['counts'] for r in rows_info])

            return jsonify(
                success=True,
                img_url=f"/static/results/{res_filename}",
                rows=rows_info,
                total=total_count
            )
        except Exception as e:
            logging.error(f"处理错误: {str(e)}")
            return jsonify(success=False, message=f"处理异常: {str(e)}")

if __name__ == '__main__':
    # 启用 Debug 模式方便开发
    app.run(host='0.0.0.0', port=5000, debug=True)
