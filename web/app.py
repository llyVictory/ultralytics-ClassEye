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

# [NEW] 人脸识别相关导入
from face_service import FaceService
from database import get_all_faces, save_face, get_user_list, update_user, delete_user, get_history_logs

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

# [NEW] 初始化人脸识别引擎
face_service = FaceService()
face_service.init_model()
known_face_db = get_all_faces()
logger = logging.getLogger("FaceBackend.App")
logger.info(f"Loaded {len(known_face_db)} face identities from DB")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/admin')
def admin():
    """管理后台页面"""
    return send_from_directory('../backend', 'index.html')

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

            # [NEW] 针对检测到的每一张脸进行识别
            box_info_list = []
            for box in boxes:
                det_conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                # 裁剪图像 (处理边界)
                px1, py1 = max(0, x1), max(0, y1)
                px2, py2 = min(img_w, x2), min(img_h, y2)
                crop = img[py1:py2, px1:px2]
                
                # 开始识别
                if crop.size > 0:
                    identify_res = face_service.identify_single_crop(crop, known_face_db)
                    name = identify_res["name"]
                    face_score = identify_res["score"]
                    number = identify_res["number"]
                else:
                    name = "Unknown"
                    face_score = 0.0
                    number = None
                
                box_info_list.append({
                    "name": name, 
                    "number": number,
                    "det_conf": det_conf, 
                    "face_conf": face_score
                })

            # 3. 渲染 (传入 box_info_list 供标注)
            canvas = visualizer.draw_results(img, results, rows_info, face_info=box_info_list)

            # 4. 保存结果图
            res_filename = f"res_{filename}"
            res_path = os.path.join(app.root_path, RESULT_FOLDER, res_filename)
            cv2.imwrite(res_path, canvas)

            # [NEW] 记录识别日志到数据库 (可选，看用户是否需要审计)
            from database import add_identify_log
            for info in box_info_list:
                add_identify_log(
                    number=info["number"], 
                    name=info["name"], 
                    score=info["face_conf"], 
                    threshold=0.45, # 假设
                    status="pass" if info["face_conf"] >= 0.45 else "not_pass"
                )

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

# --- 管理后台 API 路由 ---

@app.route('/api/users', methods=['GET'])
def list_users():
    return jsonify(success=True, data=get_user_list())

@app.route('/api/register', methods=['POST'])
def register_user():
    try:
        number = request.form.get('number')
        name = request.form.get('name')
        file = request.files.get('file')
        if not number or not file:
            return jsonify(success=False, message="参数不完整")
        
        # 1. 保存临时文件 (使用 Flask root_path 以确保路径正确)
        temp_path = os.path.join(app.root_path, "static/temp_reg.jpg")
        file.save(temp_path)
        img = cv2.imread(temp_path)
        
        # 2. 提取特征
        feat = face_service.get_feature_from_crop(img)
        if feat is None:
            return jsonify(success=False, message="未能在照片中识别出人脸")
        
        # 3. 存入数据库
        if save_face(number, name, feat):
            # 成功后记得刷新内存中的 known_face_db
            global known_face_db
            known_face_db = get_all_faces()
            return jsonify(success=True, message=f"注册成功: {name}")
        return jsonify(success=False, message="数据库保存失败")
    except Exception as e:
        return jsonify(success=False, message=str(e))

@app.route('/api/users/<number>', methods=['PUT'])
def edit_user(number):
    name = request.form.get('name')
    if update_user(number, name):
        return jsonify(success=True)
    return jsonify(success=False), 400

@app.route('/api/users/<number>', methods=['DELETE'])
def remove_user(number):
    if delete_user(number):
        global known_face_db
        known_face_db = get_all_faces() # 刷新内存
        return jsonify(success=True)
    return jsonify(success=False), 400

@app.route('/api/logs', methods=['GET'])
def list_logs():
    return jsonify(success=True, data=get_history_logs())

if __name__ == '__main__':
    # 启用 Debug 模式方便开发
    app.run(host='0.0.0.0', port=5000, debug=True)
