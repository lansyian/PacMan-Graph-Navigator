import gymnasium as gym
import numpy as np
import networkx as nx
from gymnasium import spaces
import cv2 
import collections
import ale_py
import random
from navigator import Navigator 

class PacmanGraphEnv(gym.Env):
    def __init__(self, graph, render_mode=None):
        super().__init__()
        
        self.G = graph
        self.render_mode = render_mode
        self.ale_render_mode = 'rgb_array' 
        
        # 初始化 Atari 环境
        self.env = gym.make("ALE/MsPacman-v5", render_mode=self.ale_render_mode)
        
        # 动作空间: 4个邻居的选择
        self.action_space = spaces.Discrete(4) 
        
        # 观测空间: 14维向量
        self.observation_space = spaces.Box(low=-1, high=1, shape=(14,), dtype=np.float32)
        
        self.node_list = list(self.G.nodes)
        self.current_node = self.node_list[0]
        
        # 记录最近几步防止立刻回头
        self.recent_history = collections.deque(maxlen=4)
        
        # 全局访问计数器 (防止绕圈)
        self.node_visit_counts = collections.defaultdict(int)

        self.navigator = Navigator(self.G)
        self.initial_dots = set(self.G.nodes())
        self.remaining_dots = set()
        
        self.last_obs = None

    def reset(self, seed=None, options=None):
        observation, info = self.env.reset(seed=seed)
        self.last_obs = observation
        
        pacman_pos = self.get_pacman_position()
        if pacman_pos:
            # 找到离当前物理位置最近的节点作为初始节点
            min_dist = float('inf')
            closest_node = self.node_list[0]
            for node in self.G.nodes:
                pos = self.G.nodes[node]['pos']
                d = (pacman_pos[0]-pos[0])**2 + (pacman_pos[1]-pos[1])**2
                if d < min_dist:
                    min_dist = d
                    closest_node = node
            self.current_node = closest_node
        else:
            # 如果开局没识别到（极少见），再回退到默认
            self.current_node = self.node_list[0]

        # 初始化历史记录
        self.recent_history.clear() 
        self.recent_history.append(self.current_node)
        
        # 记录初始生命值 (用于检测掉命)
        self.current_lives = info.get("lives", 3)
        
        self.node_visit_counts.clear()
        self.node_visit_counts[self.current_node] = 1
        self.remaining_dots = self.initial_dots.copy()
        
        return self._get_graph_state(observation), info

    def _get_distance_to_nearest_dot(self, node_id):
        """计算当前节点到最近豆子的图上距离(步数)"""
        if not self.remaining_dots:
            return 0
        if node_id in self.remaining_dots:
            return 0
            
        # 使用 BFS 快速搜索
        queue = collections.deque([(node_id, 0)])
        visited = {node_id}
        
        while queue:
            curr, dist = queue.popleft()
            if curr in self.remaining_dots:
                return dist
            # 限制搜索深度
            if dist > 20: 
                return 20
            
            for n in self.G.neighbors(curr):
                if n not in visited:
                    visited.add(n)
                    queue.append((n, dist + 1))
        return 20

    def step(self, action_node_index):
        # 1. 计算移动前的距离 (用于对比奖励)
        prev_dist = self._get_distance_to_nearest_dot(self.current_node)

        neighbors = list(self.G.neighbors(self.current_node))
        sorted_neighbors = sorted(neighbors, key=lambda n: self.G.nodes[n]['pos'])
        
        if action_node_index < len(sorted_neighbors):
            target_node_id = sorted_neighbors[action_node_index]
        else:
            target_node_id = self.current_node

        # 禁忌搜索
        penalty = 0
        if target_node_id in self.recent_history and len(neighbors) > 1:
            penalty = -5 
            valid_options = [n for n in sorted_neighbors if n not in self.recent_history]
            if valid_options:
                target_node_id = random.choice(valid_options)
            else:
                last_node = self.recent_history[-1] if self.recent_history else None
                fallback_options = [n for n in sorted_neighbors if n != last_node]
                if fallback_options:
                    target_node_id = random.choice(fallback_options)

        if target_node_id == self.current_node:
             state = self._get_graph_state(self.last_obs)
             return state, -10 + penalty, False, False, {}

        # 移动执行
        step_count = 0
        total_reward = 0
        done = False
        max_steps_for_this_move = 100 
        
        step_tax = 0.0 
        
        last_dist = float('inf')
        stuck_counter = 0
        
        pacman_pos_start = self.get_pacman_position()
        last_known_pos = pacman_pos_start 
        _, _, _, start_scared = self.get_ghost_info(pacman_pos_start)
        
        reached_target = False

        while step_count < max_steps_for_this_move:
            pacman_pos = self.get_pacman_position()
            
            if pacman_pos is None:
                pacman_pos = last_known_pos
            else:
                last_known_pos = pacman_pos

            joystick_action = 0
            
            # 受困逃脱
            if stuck_counter > 15:
                joystick_action = self.env.action_space.sample()
            else:
                if pacman_pos:
                    try:
                        joystick_action = self.navigator.get_action(pacman_pos, target_node_id)
                    except:
                        joystick_action = 0
            
            observation, reward, terminated, truncated, info = self.env.step(joystick_action)
            self.last_obs = observation 

            if pacman_pos:
                t_pos = self.G.nodes[target_node_id]['pos']
                dist = np.hypot(pacman_pos[0] - t_pos[0], pacman_pos[1] - t_pos[1])
                
                if dist < 12: 
                    reached_target = True
                    break
                
                if abs(dist - last_dist) < 1.0:
                    stuck_counter += 1
                else:
                    stuck_counter = 0 
                last_dist = dist
                
                if stuck_counter > 30: 
                    total_reward -= 10 
                    break
            
            # 追逐模式奖励
            if start_scared > 0.5 and pacman_pos:
                danger, _, _, is_still_scared = self.get_ghost_info(pacman_pos)
                if is_still_scared > 0.5 and danger > 0.5: 
                     total_reward += 2.0 
            
            total_reward += (reward + step_tax)
            step_count += 1
            
            if terminated or truncated:
                done = True
                # 防止自杀
                total_reward -= 500
                break

        # 结算奖励
        if reached_target:
            self.recent_history.append(self.current_node) 
            self.current_node = target_node_id 
            
            # 重复访问惩罚
            visit_count = self.node_visit_counts[target_node_id]
            repetition_penalty = 0
            if visit_count > 0:
                repetition_penalty = -10 * visit_count 
            self.node_visit_counts[target_node_id] += 1
            
            # 吃豆奖励
            dot_reward = 0
            if target_node_id in self.remaining_dots:
                self.remaining_dots.remove(target_node_id)
                dot_reward = 200 
                repetition_penalty = max(0, repetition_penalty - 5)
                
            total_reward += (dot_reward + repetition_penalty)

        # 距离引导奖励
        curr_dist = self._get_distance_to_nearest_dot(self.current_node)
        dist_shaping = 0
        if curr_dist < prev_dist:
            dist_shaping = 2.0   # 靠近奖励
        elif curr_dist > prev_dist:
            dist_shaping = -1.0  # 远离惩罚
            
        total_reward += dist_shaping
        total_reward += penalty

        next_state = self._get_graph_state(self.last_obs)
        return next_state, total_reward, done, truncated, info

    def get_pacman_position(self):
        try:
            img = self.env.unwrapped.ale.getScreenRGB()
            # 黄色 Pacman
            lower = np.array([200, 150, 50])
            upper = np.array([220, 180, 100])
            mask = cv2.inRange(img, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    return (cX, cY)
            return None
        except: return None

    def get_ghost_info(self, pacman_pos):
        if pacman_pos is None: return 0.0, 0.0, 0.0, 0.0
        img = self.env.unwrapped.ale.getScreenRGB()
        
        # 优化后的鬼魂颜色 (包含Inky修复)
        normal_ghosts_colors = [
            (np.array([160, 40, 40]), np.array([255, 130, 130])),   # Blinky
            (np.array([160, 80, 140]), np.array([255, 170, 230])),  # Pinky
            (np.array([40, 120, 160]), np.array([140, 255, 255])),  # Inky 
            (np.array([160, 80, 0]), np.array([255, 180, 120]))     # Sue
        ]
        scared_ghost_lower = np.array([40, 40, 140])
        scared_ghost_upper = np.array([160, 160, 255])
        
        ghost_positions = []
        target_is_scared = False
        
        # 检测正常鬼
        for lower, upper in normal_ghosts_colors:
            try:
                mask = cv2.inRange(img, lower, upper)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in contours:
                    if cv2.contourArea(c) > 10:
                        M = cv2.moments(c)
                        if M["m00"] != 0:
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                            ghost_positions.append((cX, cY, False))
            except: pass
            
        # 检测吓人鬼
        try:
            mask_scared = cv2.inRange(img, scared_ghost_lower, scared_ghost_upper)
            contours_s, _ = cv2.findContours(mask_scared, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours_s:
                if cv2.contourArea(c) > 10:
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        ghost_positions.append((cX, cY, True))
        except: pass

        if not ghost_positions: return 0.0, 0.0, 0.0, 0.0
        
        min_dist = float('inf')
        nearest_ghost = None
        for gx, gy, scared in ghost_positions:
            d = (pacman_pos[0]-gx)**2 + (pacman_pos[1]-gy)**2
            if d < min_dist:
                min_dist = d
                nearest_ghost = (gx, gy)
                target_is_scared = scared
        
        norm_dist = np.sqrt(min_dist)
        danger_level = 1.0 - min(norm_dist / 150.0, 1.0)
        dx = (nearest_ghost[0] - pacman_pos[0]) / 160.0
        dy = (nearest_ghost[1] - pacman_pos[1]) / 210.0
        scared_val = 1.0 if target_is_scared else 0.0
        return danger_level, dx, dy, scared_val

    def _get_graph_state(self, observation):
        pacman_pos = self.get_pacman_position()
        if pacman_pos:
            min_dist = float('inf')
            closest_node = self.current_node
            for node in self.G.nodes:
                pos = self.G.nodes[node]['pos']
                d = (pacman_pos[0]-pos[0])**2 + (pacman_pos[1]-pos[1])**2
                if d < min_dist:
                    min_dist = d
                    closest_node = node
            self.current_node = closest_node

        neighbor_scents = []
        max_search_depth = 10 
        neighbors = list(self.G.neighbors(self.current_node))
        sorted_neighbors = sorted(neighbors, key=lambda n: self.G.nodes[n]['pos'])
        for neighbor in sorted_neighbors:
            queue = collections.deque([(neighbor, 0)])
            visited = {self.current_node, neighbor}
            cnt_dots = 0
            min_dot_dist = max_search_depth
            found_dot = False
            while queue:
                curr, depth = queue.popleft()
                if curr in self.remaining_dots:
                    cnt_dots += 1
                    if not found_dot:
                        min_dot_dist = depth
                        found_dot = True
                if depth >= max_search_depth:
                    continue
                for n in self.G.neighbors(curr):
                    if n not in visited:
                        visited.add(n)
                        queue.append((n, depth + 1))
            norm_dist = 1.0 - (min_dot_dist / max_search_depth)
            norm_density = min(cnt_dots / 5.0, 1.0)
            neighbor_scents.extend([norm_dist, norm_density])
        
        while len(neighbor_scents) < 8:
            neighbor_scents.extend([0.0, 0.0])

        state = [0.0] * 14
        if pacman_pos:
            state[0] = pacman_pos[0] / 160.0
            state[1] = pacman_pos[1] / 210.0
            danger, g_dx, g_dy, is_scared = self.get_ghost_info(pacman_pos)
            state[2] = danger
            state[3] = g_dx
            state[4] = g_dy
            state[5] = is_scared
        else:
            curr_pos = self.G.nodes[self.current_node]['pos']
            state[0] = curr_pos[0] / 160.0
            state[1] = curr_pos[1] / 210.0
            state[2] = 0.0
            state[5] = 0.0
        state[6:] = neighbor_scents[:8]
        return np.array(state, dtype=np.float32)

    def close(self):
        self.env.close()