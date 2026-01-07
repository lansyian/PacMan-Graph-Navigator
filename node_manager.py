import numpy as np
import config

class NodeManager:
    def __init__(self):
        pass
    
    def _cluster_1d(self, coords, threshold):

        if not coords:
            return []
        
        coords = sorted(coords)
        clusters = []
        current_cluster = [coords[0]]
        
        for i in range(1, len(coords)):
            if coords[i] - coords[i-1] <= threshold:
                current_cluster.append(coords[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [coords[i]]
        clusters.append(current_cluster)
        
        # 返回每个簇的平均值 (即对齐后的轴线坐标)
        aligned_values = {}
        for cluster in clusters:
            avg_val = int(np.mean(cluster))
            for val in cluster:
                aligned_values[val] = avg_val
        return aligned_values

    def snap_to_grid(self, raw_nodes):
        
        raw_xs = [n[0] for n in raw_nodes]
        raw_ys = [n[1] for n in raw_nodes]
        
        # 分别对 X 轴和 Y 轴进行吸附
        map_x = self._cluster_1d(raw_xs, config.SNAP_THRESHOLD)
        map_y = self._cluster_1d(raw_ys, config.SNAP_THRESHOLD)
        
        aligned_nodes = []
        node_mapping = {} # 记录旧坐标到新坐标的映射
        
        for i, (rx, ry) in enumerate(raw_nodes):
            nx, ny = map_x[rx], map_y[ry]
            
            # 去重：避免两个点吸附到同一个位置
            if (nx, ny) not in aligned_nodes:
                aligned_nodes.append((nx, ny))
                
        print(f"Node Alignment: {len(raw_nodes)} raw -> {len(aligned_nodes)} aligned nodes.")
        return aligned_nodes