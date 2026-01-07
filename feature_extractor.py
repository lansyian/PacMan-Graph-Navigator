import cv2
import numpy as np

class FeatureExtractor:
    def __init__(self):
        self.PACMAN_COLOR = np.array([210, 164, 74]) 
        self.SCARED_GHOST_COLOR = np.array([66, 114, 194]) 
        
        self.GHOST_COLORS = [
            np.array([200, 72, 72]),   # Red
            np.array([198, 89, 179]),  # Pink
            np.array([84, 184, 153]),  # Cyan
            np.array([180, 122, 48])   # Orange
        ]
        
        self.COLOR_TOLERANCE = 15

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
        
        h, w = mask.shape
        mask[175:, :] = 0 
        
        return self._get_centroid(mask)
    
    def is_power_mode(self, observation):

        mask = self._get_color_mask(observation, self.SCARED_GHOST_COLOR)
        
        h, w = mask.shape
        mask[175:, :] = 0 
        
        pixel_count = np.count_nonzero(mask)
        return pixel_count > 30