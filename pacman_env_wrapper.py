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
        
        # 状态记录
        self.current_node = 0
        self.pacman_pos = (0, 0)
        
    def step(self, action_node_id):
        """
        这里的 action 不是 '按左键'，而是 '去节点 5' (High-level Action)
        """
        total_reward = 0
        done = False
        info = {}
        
        # --- Manager 的一步，等于 Worker 的很多步 ---
        # 我们设定一个最大步数，防止死循环
        for _ in range(50): # 比如每 50 帧做一次决策
            
            # 1. 获取当前像素位置
            obs = self.env.unwrapped._get_image() # 获取当前帧
            pos = self.vision.get_pacman_position(obs)
            
            if pos:
                self.pacman_pos = pos
                # 更新当前所在的图节点
                self.current_node = self.navigator._find_nearest_node(pos)
                
            # 2. 判断是否到达了 Manager 指定的目标节点
            if self.current_node == action_node_id:
                break # 到了！把控制权还给 Manager
                
            # 3. Worker 计算具体的摇杆动作
            low_level_action = self.navigator.get_action(self.pacman_pos, action_node_id)
            
            # 4. 环境执行一步
            obs, reward, done, info = self.env.step(low_level_action)
            total_reward += reward
            
            if done:
                break
                
            # 这里可以加渲染 self.env.render()
            
        # 返回给 Manager 的是一个抽象状态 (当前节点, 幽灵节点...)
        # 这里暂时简写，后面要详细设计 State Representation
        next_state = self._get_graph_state() 
        
        return next_state, total_reward, done, info

    def _get_graph_state(self):
        # 返回 Manager 能看懂的向量，比如 [我的节点ID, 幽灵1节点ID...]
        return np.array([self.current_node])