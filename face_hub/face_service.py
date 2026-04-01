import os
import insightface
from insightface.app import FaceAnalysis
import numpy as np
import cv2
import logging
from typing import Optional, List, Dict
# 注意：包内引用
try:
    from .adaface_loader import AdaFaceLoader
except ImportError:
    from adaface_loader import AdaFaceLoader

# 获取 logger
logger = logging.getLogger("FaceBackend.Service")

class FaceService:
    def __init__(self):
        self.app = None
        self.adaface_app = None
        self.use_adaface = os.getenv('USE_ADAFACE', 'true').lower() == 'true'
        
    def init_model(self, det_thresh: float = 0.5, det_size: int = 640):
        """初始化 InsightFace & AdaFace 模型"""
        try:
            # 路径适配：现在包在 face_hub/ 下，模型在 ../models/ 下
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # 优先检查当前包内的 models，如果没有，检查上级目录的 models
            model_root = os.path.join(current_dir, 'models')
            if not os.path.exists(os.path.join(model_root, 'buffalo_sc')):
                model_root = os.path.join(os.path.dirname(current_dir), 'models')
            
            logger.info(f"[Step 1] Initializing Face Models from root: {model_root}")
            
            # 1. InsightFace (如果启用了 AdaFace，只保留对齐功能)
            allowed_modules = ['detection', 'landmark'] if self.use_adaface else None
            self.app = FaceAnalysis(
                name='buffalo_sc', 
                root=model_root, 
                allowed_modules=allowed_modules,
                providers=['CPUExecutionProvider'] # 强制 CPU
            )
            self.app.prepare(ctx_id=-1, det_size=(det_size, det_size), det_thresh=det_thresh)
            
            # 2. AdaFace (If enabled)
            if self.use_adaface:
                adaface_path = os.path.join(model_root, 'adaface_ir101.onnx')
                if os.path.exists(adaface_path):
                    logger.info(f"[Step 1] Initializing AdaFace from: {adaface_path}")
                    self.adaface_app = AdaFaceLoader(adaface_path)
                else:
                    logger.warning(f"[Step 1] AdaFace model not found at {adaface_path}, fallback to InsightFace")
                    self.use_adaface = False
            
            logger.info("[Step 1] All face models initialization successful")
        except Exception as e:
            logger.error(f"[Step 1] Model initialization FAILED: {str(e)}")
            raise e

    def get_feature_from_crop(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        """针对已经裁剪好的单个人头/人脸图提取特征"""
        if face_img.size == 0:
            return None
            
        faces = self.app.get(face_img)
        if not faces:
            if self.use_adaface and self.adaface_app:
                return self.adaface_app.extract_feature(face_img)
            return None
            
        face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)[0]
        
        if self.use_adaface and self.adaface_app:
            from insightface.utils import face_align
            if face.kps is not None:
                aligned_face = face_align.norm_crop(face_img, landmark=face.kps)
                return self.adaface_app.extract_feature(aligned_face)
            else:
                return self.adaface_app.extract_feature(face_img)
                
        return face.embedding

    def compare_faces(self, feature1: np.ndarray, feature2: np.ndarray) -> float:
        """计算余弦相似度"""
        f1 = feature1 / (np.linalg.norm(feature1) + 1e-6)
        f2 = feature2 / (np.linalg.norm(feature2) + 1e-6)
        score = float(np.dot(f1, f2))
        return score

    def identify_single_crop(self, face_img: np.ndarray, known_faces: List[tuple], threshold: float = 0.45) -> Dict:
        """[核心识别] 对单张裁剪图进行比对"""
        feat = self.get_feature_from_crop(face_img)
        
        best_match = {"number": None, "name": "Unknown", "score": 0.0, "status": "not_pass"}
        
        if feat is None:
            return best_match

        best_score = -1
        for number, name, known_feat in known_faces:
            score = self.compare_faces(feat, known_feat)
            if score > best_score:
                best_score = score
                best_match = {
                    "number": number, 
                    "name": name, 
                    "score": round(float(score), 4),
                    "status": "pass" if score >= threshold else "not_pass"
                }
        
        if best_match["status"] == "not_pass" and best_match["name"] != "Unknown":
            best_match["name"] = f"[?] {best_match['name']}"
            
        return best_match
