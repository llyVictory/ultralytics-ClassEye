import cv2
import numpy as np
from ultralytics.utils.plotting import Annotator, colors
from .config import config

class ClassEyeVisualizer:
    def __init__(self, thickness=2, font_scale=0.8):
        self.thickness = thickness
        self.font_scale = font_scale

    def draw_results(self, img, results, rows_info, face_info=None):
        """
        绘制最终效果图：YOLO 检测框 + 分排统计线 + 汇总文字
        :param img: 原始 BGR 图片
        :param results: ultralytics Results 对象
        :param rows_info: 由 RowSplitter 提供的排信息 [{name, region, counts}, ...]
        :param face_info: (可选) 与 results[0].boxes 一一对应的识别信息列表 [{name, det_conf, face_conf}, ...]
        :return: 渲染后的 BGR 图片
        """
        if img is None:
            return None
            
        annotator = Annotator(img.copy(), line_width=self.thickness)

        # 1. 绘制检测框 (来自 YOLO)
        if results and len(results[0].boxes) > 0:
            for i, box in enumerate(results[0].boxes):
                # 只绘制 person 类
                if int(box.cls[0]) == config.TARGET_CLASS:
                    # 如果有对应的识别信息，格式化显示双得分
                    if face_info and i < len(face_info):
                        info = face_info[i]
                        name = info["name"]
                        det_score = info["det_conf"]
                        face_score = info["face_conf"]
                        label = f"{name} D:{det_score:.2f} F:{face_score:.2f}"
                    else:
                        label = f"{results[0].names[int(box.cls[0])]} {box.conf[0]:.2f}"
                    
                    annotator.box_label(box.xyxy[0], label, color=colors(0, True))

        canvas = annotator.result()

        # 2. 绘制排汇总信息 (Row separator lines & Text labels)
        total_count = 0
        img_h, img_w = canvas.shape[:2]

        for row in rows_info:
            name = row["name"]
            counts = row["counts"]
            total_count += counts
            
            # 如果存在斜线边界信息 (Manual Mode)
            if "boundary" in row:
                line_up = row["boundary"][0]
                # 绘制排的分割线 (橙色准水平线)
                # 注：OpenCV 绘制虚线比较复杂，这里先用实线，但在 Web 可视化中使用虚线
                color = (0, 165, 255) # Orange BGR
                p1 = (0, int(line_up[0]))
                p2 = (img_w, int(line_up[1]))
                cv2.line(canvas, p1, p2, color, self.thickness, cv2.LINE_AA)
                
                # 在排的中左侧标注
                mid_y = int((line_up[0] + line_up[1]) / 2)
                cv2.putText(canvas, f"{name}: {counts}p", (10, mid_y + 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, color, 2)
            else:
                # 传统自动模式 (水平线)
                y1, y2 = int(row["region"][1]), int(row["region"][3])
                color = config.DRAW_LINE_COLOR
                cv2.line(canvas, (0, y1), (img_w, y1), color, self.thickness, cv2.LINE_AA)
                label_text = f"{name}: {counts}ppl"
                cv2.putText(canvas, label_text, (10, (y1 + y2) // 2 + 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, color, 2)

        # 3. 绘制全场汇总 (右上角)
        summary_text = f"Total: {total_count} persons detected"
        # 动态计算文字宽度以防止边缘遮挡
        (w, h), _ = cv2.getTextSize(summary_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        margin = 50 
        cv2.putText(canvas, summary_text, (img_w - w - margin, 40 + h), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        return canvas
