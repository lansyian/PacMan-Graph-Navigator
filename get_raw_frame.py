import gymnasium as gym
import cv2
import os
import ale_py

def save_raw_frame():
    # 1. 使用和测试时完全一样的环境 ID
    env_name = 'ALE/MsPacman-v5' 
    # 确保 render_mode 是 rgb_array 以获取原始像素
    env = gym.make(env_name, render_mode='rgb_array') 
    
    observation, info = env.reset()
    
    # 2. 预热游戏，让画面稳定下来 (等 Pacman 出现)
    print("Warming up environment...")
    for _ in range(60):
        observation, _, _, _, _ = env.step(0) # Noop
        
    # 3. 获取当前帧的形状 (这是最重要的信息!)
    h, w, c = observation.shape
    print(f"\n--- Captured Raw Frame Info ---")
    print(f"Height: {h} (Y axis)")
    print(f"Width:  {w} (X axis)")
    print(f"Channels: {c}")
    print("-------------------------------\n")
    
    # 4. 保存图片 (注意要从 RGB 转 BGR 才能用 OpenCV 正确保存)
    save_path = "standard_reference_map.png"
    # 如果 observation 不是 uint8 类型，可能需要转换
    obs_bgr = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, obs_bgr)
    
    print(f"SUCCESS: Standard reference map saved to '{save_path}'.")
    print("Please use THIS image for your node_extractor.py!")
    
    env.close()

if __name__ == "__main__":
    save_raw_frame()