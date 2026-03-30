class Config:
    # 模型配置
    MODEL_PATH = "yolov8m.pt"  # 默认使用 yolov8m.pt 兼顾精度与速度
    CONF_THRESHOLD = 0.5      # 检测置信度阈值
    TARGET_CLASS = 0           # 人(person) 在 COCO 中的类别索引

    # 自动分排配置 (DBSCAN 参数)
    DB_EPS = 20                # 聚类邻域半径 (像素)，根据图片高度调节
    DB_MIN_SAMPLES = 1         # 最小样本数 (单人也可成排)

    # 绘制配置
    DRAW_LINE_COLOR = (0, 255, 0) # 排分界线颜色 (BGR)
    TEXT_COLOR = (255, 255, 255)  # 文字颜色
    FONT_SCALE = 1.0              # 字体大小

    # 输出配置
    OUTPUT_DIR = "output"
    SAMPLE_DIR = "samples"

config = Config()
