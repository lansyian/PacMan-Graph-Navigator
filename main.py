import matplotlib.pyplot as plt
import networkx as nx
from vision_processor import VisionProcessor
from node_manager import NodeManager
from graph_builder import GraphBuilder
import config
import json
import os
import pickle
import os

def load_nodes_from_json(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"找不到节点文件: {json_path}。请先运行 node_extractor.py")
    with open(json_path, 'r') as f:
        nodes = json.load(f)
    return [tuple(p) for p in nodes]

def main():
    img_path = "standard_reference_map.png" 
    json_path = "real_nodes_corrected.json" 
    
    vision = VisionProcessor()
    node_mgr = NodeManager()
    builder = GraphBuilder()
    
    print(">>> Phase 1: Processing Vision...")
    clean_wall_mask = vision.get_robust_wall_mask(img_path)

    print(">>> Phase 2: Loading & Aligning Nodes...")

    raw_nodes = load_nodes_from_json(json_path)
    
    # (Snap-to-Grid)
    aligned_nodes = node_mgr.snap_to_grid(raw_nodes)
    
    print(">>> Phase 3: Building Graph...")
    G = builder.build_graph(aligned_nodes, clean_wall_mask)
    print(f"Graph Built! Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    if not os.path.exists('data'):
        os.makedirs('data')

    save_path = "data/pacman_graph.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(G, f)
    print(f"SUCCESS: Graph object saved to {save_path}")
    print("This file contains the logic structure (NetworkX object) for navigation.")

    # 可视化图形
    if config.SHOW_PLOTS:
        plt.figure(figsize=(12, 6))
        
        # 左图：Mask
        plt.subplot(1, 2, 1)
        plt.title("Processed Wall Mask")
        plt.imshow(clean_wall_mask, cmap='gray')
        
        # 右图：Graph
        plt.subplot(1, 2, 2)
        plt.title("Topological Graph (Final)")
        plt.imshow(clean_wall_mask, cmap='gray', alpha=0.3) 
        pos = nx.get_node_attributes(G, 'pos')
        
        nx.draw_networkx_nodes(G, pos, node_size=20, node_color='red')
        nx.draw_networkx_edges(G, pos, edge_color='lime', width=1.5, alpha=0.7)

        tunnel_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'tunnel']
        nx.draw_networkx_edges(G, pos, edgelist=tunnel_edges, edge_color='cyan', width=2.0, style='dashed')

        plt.tight_layout()
        
        plt.savefig("data/graph_visualization.png", dpi=300)
        print("Visualization saved to data/graph_visualization.png")
        
        plt.show()

if __name__ == "__main__":
    main()