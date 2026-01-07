import gymnasium as gym
import numpy as np
from feature_extractor import FeatureExtractor
from navigator import Navigator

class PacmanGraphEnv(gym.Wrapper):
    def __init__(self, env, graph):
        super().__init__(env)
        self.graph = graph
        self.vision = FeatureExtractor()
        self.navigator = Navigator(graph)
        
        self.current_node = 0
        self.pacman_pos = (0, 0)
        
    def step(self, action_node_id):

        total_reward = 0
        done = False
        info = {}
        
        for _ in range(50):
            
            # 1. 获取当前像素位置
            obs = self.env.unwrapped._get_image()
            pos = self.vision.get_pacman_position(obs)
            
            if pos:
                self.pacman_pos = pos
                self.current_node = self.navigator._find_nearest_node(pos)
                
            # 2. 判断是否到达了 Manager 指定的目标节点
            if self.current_node == action_node_id:
                break 
                
            # 3. Worker 计算具体的摇杆动作
            low_level_action = self.navigator.get_action(self.pacman_pos, action_node_id)
            
            # 4. 环境执行一步
            obs, reward, done, info = self.env.step(low_level_action)
            total_reward += reward
            
            if done:
                break

        next_state = self._get_graph_state() 
        
        return next_state, total_reward, done, info

    def _get_graph_state(self):
        return np.array([self.current_node])