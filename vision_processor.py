import cv2
import numpy as np
import config

class VisionProcessor:
    def __init__(self):
        pass

    def get_robust_wall_mask(self, image_path):
        """
        读取图片并生成纯净的墙壁掩码。
        使用“面积+长宽比”双重过滤，解决细墙壁与豆子混淆的问题。
        """
        # 1. 读取与预处理
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image not found!")
        
        # 转灰度
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. 基础二值化 (此时墙壁和豆子都是白色)
        # 注意：假设背景是黑色(0)，其他是亮的。如果是反的需调整。
        _, binary = cv2.threshold(gray, config.BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
        
        # 3. 连通域分析 (获取所有独立块的属性)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        # 创建干净的画布
        clean_mask = np.zeros_like(binary)
        
        # 4. 智能过滤循环
        for i in range(1, num_labels):  # 跳过 0 (背景)
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            
            # 计算长宽比 (防止除以0)
            ratio = width / (height + 1e-5)
            
            # --- 核心逻辑 Start ---
            
            # 逻辑 A: 足够大的块，肯定是墙 (主要结构)
            is_large_wall = area > config.MIN_LARGE_WALL_AREA
            
            # 逻辑 B: 即使面积小，如果是细长的，也是墙 (细微连接处)
            # 豆子的 ratio 通常接近 1.0 (正方形/圆形)
            is_thin_wall = (area > config.MIN_THIN_WALL_AREA) and \
                           (ratio > config.THIN_WALL_RATIO_HIGH or ratio < config.THIN_WALL_RATIO_LOW)
            
            # --- 核心逻辑 End ---
            
            if is_large_wall or is_thin_wall:
                # 只保留满足条件的区域
                clean_mask[labels == i] = 255
            # else: 丢弃 (即豆子)
            
        return clean_mask