import onnxruntime as ort
import numpy as np
import cv2
import logging

logger = logging.getLogger("FaceBackend.AdaFace")

class AdaFaceLoader:
    def __init__(self, model_path: str):
        """初始化 AdaFace ONNX 推理引擎"""
        self.model_path = model_path
        # 优先使用 CUDA，如果不可用则回退到 CPU
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        try:
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            logger.info(f"AdaFace model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load AdaFace model: {e}")
            raise e

    def preprocess(self, img_cv2: np.ndarray) -> np.ndarray:
        """
        AdaFace 预处理逻辑:
        1. Resize 到 112x112
        2. BGR 格式 (InsightFace 默认)
        3. 归一化: (x - 127.5) / 128.0
        4. 维度变换: [H, W, C] -> [1, C, H, W]
        """
        # 1. Resize
        img = cv2.resize(img_cv2, (112, 112))
        
        # 2. 转换为 float32 并归一化
        img = img.astype(np.float32)
        img = (img - 127.5) / 128.0
        
        # 3. HWC -> CHW
        img = np.transpose(img, (2, 0, 1))
        
        # 4. 增加 batch 维度
        img = np.expand_dims(img, axis=0)
        
        return img

    def extract_feature(self, face_img: np.ndarray) -> np.ndarray:
        """输入人脸裁剪图，提取 512 维特征向量"""
        try:
            # 1. 预处理
            input_tensor = self.preprocess(face_img)
            
            # 2. 推理
            outputs = self.session.run(None, {self.input_name: input_tensor})
            embeddings = outputs[0] # [1, 512]
            
            # 3. 后处理: L2 归一化 (确保余弦相似度计算正确)
            norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norm + 1e-6)
            
            return embeddings.flatten()
        except Exception as e:
            logger.error(f"AdaFace feature extraction error: {e}")
            # 返回全零向量作为 fallback，防止崩溃
            return np.zeros(512, dtype=np.float32)
