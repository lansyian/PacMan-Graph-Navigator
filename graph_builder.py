import networkx as nx
import numpy as np
import config

class GraphBuilder:
    def __init__(self):
        self.graph = nx.Graph()

    def check_line_collision(self, u, v, wall_mask):
        # ... (保持之前的代码不变) ...
        # (这里为了节省篇幅省略，请保留原有的 check_line_collision 代码)
        x1, y1 = u
        x2, y2 = v
        dist = int(np.hypot(x2-x1, y2-y1))
        if dist == 0: return False
        xs = np.linspace(x1, x2, dist).astype(int)
        ys = np.linspace(y1, y2, dist).astype(int)
        collision_threshold = 2
        collision_pixels = 0
        for x, y in zip(xs, ys):
            if 0 <= y < wall_mask.shape[0] and 0 <= x < wall_mask.shape[1]:
                if wall_mask[y, x] == 255:
                    collision_pixels += 1
        return collision_pixels <= collision_threshold

    def _add_tunnel_connections(self, nodes, image_width):
        """
        [Debug版] 专门检测并连接左右穿梭隧道。
        """
        left_nodes = []
        right_nodes = []

        print(f"\n--- [Tunnel Debug] Start checking tunnels ---")
        print(f"Image Width: {image_width}")
        print(f"Config Thresholds: LEFT < {config.LEFT_EDGE_THRESHOLD}, RIGHT > {image_width - config.RIGHT_EDGE_THRESHOLD}")
        print(f"Config Y Tolerance: {config.TUNNEL_Y_TOLERANCE}")

        # 1. 分类节点
        for i, (x, y) in enumerate(nodes):
            # 检查左边
            if x <= config.LEFT_EDGE_THRESHOLD:
                left_nodes.append((i, x, y))
                print(f"  -> Found LEFT candidate: Node {i} at ({x}, {y})")
            
            # 检查右边
            if x >= image_width - config.RIGHT_EDGE_THRESHOLD:
                right_nodes.append((i, x, y))
                print(f"  -> Found RIGHT candidate: Node {i} at ({x}, {y})")

        print(f"Candidates found: {len(left_nodes)} Left, {len(right_nodes)} Right")

        # 2. 配对连接
        tunnel_count = 0
        for l_idx, lx, ly in left_nodes:
            for r_idx, rx, ry in right_nodes:
                y_diff = abs(ly - ry)
                print(f"  Comparing Node {l_idx}(y={ly}) with Node {r_idx}(y={ry})... Diff: {y_diff}")
                
                # 检查 Y 轴是否对齐
                if y_diff <= config.TUNNEL_Y_TOLERANCE:
                    self.graph.add_edge(l_idx, r_idx, weight=1.0, type='tunnel')
                    tunnel_count += 1
                    print(f"  >>> SUCCESS: Tunnel created: Node {l_idx} <---> Node {r_idx}")
                else:
                    print(f"  >>> FAILED: Y misalignment too large (Allowed: {config.TUNNEL_Y_TOLERANCE})")
        
        if tunnel_count == 0:
            print("--- [Tunnel Debug] WARNING: No tunnels created! Please adjust config.py ---")
        else:
            print(f"--- [Tunnel Debug] Finished. Total tunnels: {tunnel_count} ---")
        
        return tunnel_count

    def build_graph(self, nodes, wall_mask):
        self.graph.clear()
        image_width = wall_mask.shape[1] # 获取地图宽度
        
        # 1. 添加节点
        for i, node in enumerate(nodes):
            self.graph.add_node(i, pos=node)
            
        # 2. 生成常规边 (基于曼哈顿几何 + 视觉检测)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                u = nodes[i]
                v = nodes[j]
                
                # 曼哈顿筛选
                if u[0] == v[0] or u[1] == v[1]:
                    # 视觉碰撞检测
                    if self.check_line_collision(u, v, wall_mask):
                        # 确保不跨越节点
                        if not self._has_node_between(u, v, nodes):
                            dist = np.linalg.norm(np.array(u) - np.array(v))
                            self.graph.add_edge(i, j, weight=dist, type='corridor')

        # 3. [新增] 生成隧道边
        self._add_tunnel_connections(nodes, image_width)
                            
        return self.graph

    def _has_node_between(self, u, v, all_nodes):
        # ... (保持之前的代码不变) ...
        x1, y1 = u
        x2, y2 = v
        for k in all_nodes:
            if k == u or k == v: continue
            xk, yk = k
            cross_product = (yk - y1) * (x2 - x1) - (xk - x1) * (y2 - y1)
            if abs(cross_product) > 1e-5: continue 
            dot_product = (xk - x1) * (x2 - x1) + (yk - y1) * (y2 - y1)
            squared_length_ba = (x2 - x1)**2 + (y2 - y1)**2
            if 0 < dot_product < squared_length_ba:
                return True
        return False