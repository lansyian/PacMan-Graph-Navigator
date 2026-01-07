import pickle
import os
import time
import cv2
import numpy as np
from stable_baselines3 import DQN
from pacman_graph_env import PacmanGraphEnv
import ale_py

def main():
    # 1. 加载地图
    if not os.path.exists("data/pacman_graph.pkl"):
        print("Error: can not find data/pacman_graph.pkl")
        return

    with open("data/pacman_graph.pkl", "rb") as f:
        G = pickle.load(f)

    # 2. 创建环境
    env = PacmanGraphEnv(G, render_mode='rgb_array')

    # 3. 加载模型
    model_path = "models/dqn_pacman/dqn_final.zip"
    if not os.path.exists(model_path):
        print(f"Error: can not find model {model_path}")
        return

    print(f"Loading model from {model_path}...")
    model = DQN.load(model_path, env=env)

    # 4. 开始展示
    episodes = 5
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        score = 0
        
        print(f"Episode {ep + 1} started...")
        
        while not done:
            # 预测
            action, _states = model.predict(obs, deterministic=True)
            
            # 执行
            obs, reward, terminated, truncated, info = env.step(action)
            score += reward
            
            # 1. 获取当前帧图像
            img = env.env.render() 
            
            # 2. 转换颜色
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                # 3. 显示分数
                cv2.putText(img, f"Score: {score:.1f}", (10, 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # 4. 放大
                img = cv2.resize(img, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
                
                # 5. 显示
                cv2.imshow("Pac-Man Graph Agent", img)
            
            # 6. 处理退出事件
            if cv2.waitKey(10) & 0xFF == ord('q'):
                done = True
                break
            
            if terminated or truncated:
                done = True
                print(f"Episode {ep + 1} finished. Score: {score}")
                time.sleep(1.0) 

    env.close()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)

if __name__ == "__main__":
    main()


