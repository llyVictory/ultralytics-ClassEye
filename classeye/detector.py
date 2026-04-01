from ultralytics import YOLO
import logging
from .config import config

class ClassEyeDetector:
    def __init__(self, model_path=config.MODEL_PATH):
        """
        初始化人头检测器 (基于 yolov8)
        :param model_path: 模型权重路径 (默认 yolov8m.pt)
        """
        self.model = YOLO(model_path)
        logging.info(f"[ClassEye] 已加载模型: {model_path}")

    def detect_heads(self, source, conf=config.CONF_THRESHOLD):
        """
        进行人头检测 (只对 person 类别 0 进行过滤)
        :param source: 输入源 (图片路径, 数组, 或视频)
        :param conf: 置信度
        :returns: ultralytics Results 列表
        """
        # classes=[0] 即只检测 person (在 COCO 中 = 0)
        # stream=False 默认全内存处理，适用于单次处理场景
        results = self.model.predict(
            source,
            conf=conf,
            classes=[config.TARGET_CLASS],
            verbose=False
        )
        return results

if __name__ == "__main__":
    # 本地跑个测试
    import cv2
    import os
    
    # 放置一个测试样本
    img_path = os.path.join(config.SAMPLE_DIR, "classroom_01.jpg")
    detector = ClassEyeDetector()
    results = detector.detect_heads(img_path)
    
    if results:
        results[0].show()
