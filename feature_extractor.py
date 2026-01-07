import cv2
import numpy as np

class FeatureExtractor:
    def __init__(self):
        # 1. 你之前微调好的 Pacman 颜色 (请确认这是你刚才调好的值)
        self.PACMAN_COLOR = np.array([210, 164, 74]) 
        
        # 2. === [必须新增] === 受惊幽灵的颜色 (蓝青色)
        # 这是一个默认估计值，建议稍后用 tune_vision.py 精确校准
        self.SCARED_GHOST_COLOR = np.array([66, 114, 194]) 
        
        self.GHOST_COLORS = [
            np.array([200, 72, 72]),   # Red
            np.array([198, 89, 179]),  # Pink
            np.array([84, 184, 153]),  # Cyan
            np.array([180, 122, 48])   # Orange
        ]
        
        self.COLOR_TOLERANCE = 15 # 你之前微调好的容差

    def _get_color_mask(self, image, color):
        lower = np.clip(color - self.COLOR_TOLERANCE, 0, 255)
        upper = np.clip(color + self.COLOR_TOLERANCE, 0, 255)
        mask = cv2.inRange(image, lower, upper)
        return mask

    def _get_centroid(self, mask):
        M = cv2.moments(mask)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
        return None

    def get_pacman_position(self, observation):
        mask = self._get_color_mask(observation, self.PACMAN_COLOR)
        
        # === 底部屏蔽 (保留这个!) ===
        h, w = mask.shape
        mask[175:, :] = 0 
        
        return self._get_centroid(mask)

    # ==========================================================
    # === [必须新增] 下面这个就是报错缺少的函数 ===
    # ==========================================================
    def is_power_mode(self, observation):
        """
        判断当前是否处于能量模式 (幽灵是否变色)
        """
        # 1. 获取受惊颜色的掩码
        mask = self._get_color_mask(observation, self.SCARED_GHOST_COLOR)
        
        # 2. 同样的，我们要屏蔽底部干扰
        h, w = mask.shape
        mask[175:, :] = 0 
        
        # 3. 统计蓝色像素数量
        # 如果画面中出现超过 30 个像素的特定蓝色，说明有受惊的幽灵
        pixel_count = np.count_nonzero(mask)
        return pixel_count > 30