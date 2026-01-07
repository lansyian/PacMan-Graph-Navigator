import networkx as nx
import numpy as np

class Navigator:
    def __init__(self, graph):
        self.G = graph
        # 定义动作映射 (根据 Gym 的定义)
        # 假设: 1=Up, 2=Right, 3=Left, 4=Down (需核对 Gym 文档)
        self.ACTIONS = {'UP': 1, 'RIGHT': 2, 'LEFT': 3, 'DOWN': 4}
    
    def get_action(self, current_pos, target_node_id):
        """
        决定下一步怎么走。包含隧道特殊处理。
        """
        # 1. 如果没有目标或已经到了，不动
        if target_node_id is None or current_pos is None:
            return 0

        # 2. 规划路径 (A* 或 Shortest Path)
        # 找到最近的图节点作为起点
        start_node = self._find_nearest_node(current_pos)
        
        try:
            path = nx.shortest_path(self.G, source=start_node, target=target_node_id)
        except nx.NetworkXNoPath:
            return 0 # 无路可走
            
        if len(path) < 2:
            return 0 # 已经到了
            
        # 获取下一步的节点信息
        next_node = path[1]
        next_pos = self.G.nodes[next_node]['pos']
        
        # ==========================================================
        # === [核心修复] 隧道穿越判定 (Tunnel/Teleport Logic) ===
        # ==========================================================
        dx = next_pos[0] - current_pos[0]
        dy = next_pos[1] - current_pos[1]
        
        # 计算欧几里得距离
        pixel_dist = (dx**2 + dy**2) ** 0.5
        
        # 如果两点是邻居，但像素距离巨大 (>100)，说明这是隧道两端
        if pixel_dist > 100:
            # 此时逻辑反转：
            # 如果我在右边 (x>80)，要去左边 -> 应该继续向右冲 (Right)
            if current_pos[0] > 80: 
                return 2 # 2 is RIGHT in Atari (通常)
            # 如果我在左边 (x<80)，要去右边 -> 应该继续向左冲 (Left)
            else:
                return 3 # 3 is LEFT in Atari
                
        # ==========================================================
        # === 普通移动逻辑 (如果没有触发隧道) ===
        # ==========================================================
        
    
        
        # 直接进行方向判定
        if abs(dx) > abs(dy):
             return self.ACTIONS['RIGHT'] if dx > 0 else self.ACTIONS['LEFT']
        else:
             return self.ACTIONS['DOWN'] if dy > 0 else self.ACTIONS['UP']
            
        # 简单的方向判定 (优先走距离更远的轴，或者是只要有偏差就走)
        if abs(dx) > abs(dy):
            # 水平移动
            return 2 if dx > 0 else 3 # Right : Left
        else:
            # 垂直移动
            return 5 if dy > 0 else 4 # Down : Up (注意：y轴向下是正)

    def _find_nearest_node(self, pixel_pos):
        # 简单的最近邻搜索
        min_dist = float('inf')
        nearest_node = None
        px, py = pixel_pos
        
        for node, data in self.G.nodes(data=True):
            nx, ny = data['pos']
            dist = (px-nx)**2 + (py-ny)**2
            if dist < min_dist:
                min_dist = dist
                nearest_node = node
        return nearest_node

    def _calculate_move(self, current, target):
        cx, cy = current
        tx, ty = target
        
        # 简单的阈值判定
        dx = tx - cx
        dy = ty - cy
        
        if abs(dx) > abs(dy): # 水平移动优先
            if dx > 0: return self.ACTIONS['RIGHT']
            else:      return self.ACTIONS['LEFT']
        else: # 垂直移动优先
            if dy > 0: return self.ACTIONS['DOWN'] # 图像坐标系 y 向下是正
            else:      return self.ACTIONS['UP']