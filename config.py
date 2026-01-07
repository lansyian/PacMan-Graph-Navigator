# --- 图像处理参数 ---
# 墙壁颜色阈值 (HSV或RGB范围，根据你的截图调整)
# 这里假设简单的二值化阈值，因为我们主要靠形状过滤
BINARY_THRESHOLD = 50 

# --- 形状过滤参数 (核心 Trick) ---
# 1. 大墙壁：面积只要大于这个值，直接保留
MIN_LARGE_WALL_AREA = 200 

# 2. 细墙壁：面积可以小，但必须长得像线条
MIN_THIN_WALL_AREA = 15      # 最小面积 (比豆子稍大或相当)
THIN_WALL_RATIO_HIGH = 3.0   # 长宽比 > 3 (横向细线)
THIN_WALL_RATIO_LOW = 0.33   # 长宽比 < 0.33 (纵向细线)

# --- 节点对齐参数 ---
SNAP_THRESHOLD = 6           # 像素吸附距离 (px)

# config.py

# ... (之前的参数保持不变)

# --- 隧道机制参数 ---
# 如果节点的 x 坐标小于这个值，视为“左出口”
LEFT_EDGE_THRESHOLD = 8
# 如果节点的 x 坐标大于 (图像宽度 - 这个值)，视为“右出口”
RIGHT_EDGE_THRESHOLD = 10
# 允许左右隧道在 Y 轴上的微小误差
TUNNEL_Y_TOLERANCE = 2


# --- 调试 ---
SHOW_PLOTS = True