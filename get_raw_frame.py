import gymnasium as gym
import cv2
import os
import ale_py

def save_raw_frame():

    env_name = 'ALE/MsPacman-v5' 
    env = gym.make(env_name, render_mode='rgb_array') 
    
    observation, info = env.reset()
    
    print("Warming up environment...")
    for _ in range(60):
        observation, _, _, _, _ = env.step(0) # Noop
        
    h, w, c = observation.shape
    print(f"\n--- Captured Raw Frame Info ---")
    print(f"Height: {h} (Y axis)")
    print(f"Width:  {w} (X axis)")
    print(f"Channels: {c}")
    print("-------------------------------\n")
    
    # 保存图片
    save_path = "standard_reference_map.png"
    
    obs_bgr = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, obs_bgr)
    
    print(f"SUCCESS: Standard reference map saved to '{save_path}'.")
    print("Please use THIS image for your node_extractor.py!")
    
    env.close()

if __name__ == "__main__":
    save_raw_frame()