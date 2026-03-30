import numpy as np
from sklearn.cluster import DBSCAN
import logging
from .config import config

class RowSplitter:
    def __init__(self, eps=config.DB_EPS, min_samples=config.DB_MIN_SAMPLES):
        """
        初始化自动分排统计逻辑 (基于 Y 坐标聚类)
        :param eps: DBSCAN 邻域参数，单位像素。距离在该范围内的点被视为同一排。
        :param min_samples: 聚类点合集的最小个数。
        """
        self.eps = eps
        self.min_samples = min_samples

    def split_rows(self, boxes, img_w, img_h, manual_y_list=None):
        """
        分排逻辑入口：由用户手动作画控制。
        如果没有收到手动线，则整个图片视为一排（Row-1）。
        """
        if not manual_y_list:
            # 没有线，返回全图统计作为第一排
            xyxy = boxes.xyxy.cpu().numpy()
            count = len(xyxy)
            return [{
                "id": 1,
                "name": "Row-1",
                "region": [0, 0, img_w, 0, 0, img_h, img_w, img_h], # 四角顶边
                "counts": count,
                "boundary": [[0,0], [img_h, img_h]] 
            }]
        
        # 进入手动斜线识别模式
        return self.split_rows_manual(boxes, img_w, img_h, manual_y_list)

    def split_rows_manual(self, boxes, img_w, img_h, lines):
        """
        手动斜线分排模式：逻辑为判断点是否处于两条斜线方程构成的区间内。
        :param lines: 用户手动定义的线列表 [ [[x1, y1], [x2, y2]], ... ]
        """
        if len(boxes) == 0: return []
        
        # 完善判定线组
        boundary_top = [0, 0]
        boundary_bottom = [img_h, img_h]
        
        extracted_lines = []
        for line in lines:
            p1, p2 = line[0], line[1]
            dx = p2[0] - p1[0]
            if dx == 0: continue
            k = (p2[1] - p1[1]) / dx
            y_left = p1[1] - k * p1[0]
            y_right = y_left + k * img_w
            extracted_lines.append((y_left, y_right))
        
        # 排序
        extracted_lines.sort(key=lambda x: (x[0] + x[1]) / 2)
        final_lines = [boundary_top] + extracted_lines + [boundary_bottom]
        
        xyxy = boxes.xyxy.cpu().numpy()
        centers_x = (xyxy[:, 0] + xyxy[:, 2]) / 2
        centers_y = (xyxy[:, 1] + xyxy[:, 3]) / 2
        
        final_results = []
        for j in range(len(final_lines) - 1):
            line_up = final_lines[j]
            line_down = final_lines[j+1]
            
            def get_y_on_line(line, x):
                k = (line[1] - line[0]) / img_w
                return line[0] + k * x

            in_zone_count = 0
            for idx in range(len(centers_x)):
                px, py = centers_x[idx], centers_y[idx]
                y_up = get_y_on_line(line_up, px)
                y_down = get_y_on_line(line_down, px)
                if py >= y_up and py < y_down:
                    in_zone_count += 1
            
            final_results.append({
                "id": j + 1,
                "name": f"Row-{j + 1}",
                "region": [0, line_up[0], img_w, line_up[1], 0, line_down[0], img_w, line_down[1]],
                "counts": in_zone_count,
                "boundary": [line_up, line_down]
            })
            
        return final_results
