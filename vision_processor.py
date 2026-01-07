import cv2
import numpy as np
import config

class VisionProcessor:
    def __init__(self):
        pass

    def get_robust_wall_mask(self, image_path):

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image not found!")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        _, binary = cv2.threshold(gray, config.BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        clean_mask = np.zeros_like(binary)
        
        for i in range(1, num_labels): 
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            
            ratio = width / (height + 1e-5)
            
            is_large_wall = area > config.MIN_LARGE_WALL_AREA

            is_thin_wall = (area > config.MIN_THIN_WALL_AREA) and \
                           (ratio > config.THIN_WALL_RATIO_HIGH or ratio < config.THIN_WALL_RATIO_LOW)

            
            if is_large_wall or is_thin_wall:

                clean_mask[labels == i] = 255
            
        return clean_mask