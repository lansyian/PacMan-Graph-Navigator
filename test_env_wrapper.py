import pickle
from pacman_graph_env import PacmanGraphEnv
import random
import ale_py

def main():
    # 1. 加载图
    print("Loading graph...")
    with open("data/pacman_graph.pkl", "rb") as f:
        G = pickle.load(f)

    # 2. 创建高层环境
    print("Initializing Graph Environment...")
    env = PacmanGraphEnv(G, render_mode='human') # 用 human 模式看效果

    obs, info = env.reset()
    print(f"Initial State: {obs}") # 应该是 [NodeID, -1, -1, -1, -1]

    done = False
    total_score = 0
    
    # 3. 随机漫步测试
    while not done:
        # 随机选择一个节点作为目标
        # env.action_space.sample() 返回的是 0 ~ num_nodes-1 的索引
        action = env.action_space.sample()
        
        target_node_id = list(G.nodes)[action]
        print(f"Manager decides: Go to Node {target_node_id} (Index {action})")
        
        # 这一步 step 可能会持续几秒钟 (Worker 在干活)
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_score += reward
        print(f" -> Step finished. Reward: {reward}, New State: {obs}")
        
        if terminated or truncated:
            print("Game Over!")
            break

    env.close()

if __name__ == "__main__":
    main()