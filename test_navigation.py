import gymnasium as gym
import cv2
import networkx as nx
import numpy as np
import random
import pickle
import os
import ale_py

from feature_extractor import FeatureExtractor
from navigator import Navigator

# 全局变量用于鼠标回调
mouse_color = None
# 全局变量用于存储最后一帧的画面，防止暂停时画面闪烁或消失
last_frame_bgr = None

def mouse_callback(event, x, y, flags, param):
    global mouse_color
    frame = param
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # 防止点击超出边界
        h, w, _ = frame.shape
        if y >= h or x >= w: return

        b, g, r = frame[y, x]
        print(f"\n[COLOR PICKER] Clicked at ({x}, {y})")
        print(f"               BGR (OpenCV): [{b}, {g}, {r}]")
        print(f"               RGB (Config): [{r}, {g}, {b}]")
        print(f"--------------------------------------------------")

def load_prebuilt_graph(path="data/pacman_graph.pkl"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到图文件: {path}")
    with open(path, "rb") as f:
        G = pickle.load(f)
    return G

def main():
    env_name = 'ALE/MsPacman-v5' 
    env = gym.make(env_name, render_mode='rgb_array') 
    
    try:
        G = load_prebuilt_graph()
    except FileNotFoundError as e:
        print(e)
        return

    navigator = Navigator(G)
    extractor = FeatureExtractor()

    print(">>> Initializing Environment...")
    observation, info = env.reset()
    
    # 预热
    for _ in range(60):
        observation, _, _, _, _ = env.step(0)

    print("\n==========================================")
    print("      DEBUGGER WITH PAUSE FUNCTION        ")
    print("==========================================")
    print(" [P]   : PAUSE / RESUME Game (暂停/继续)")
    print(" [Click]: Get RGB Color (at any time)")
    print(" [Q]   : Quit")
    print("==========================================\n")

    current_target_node = None
    paused = False 
    
    cv2.namedWindow("Debug View", cv2.WINDOW_NORMAL)
    
    current_obs = observation
    
    try:
        while True:

            if not paused:
                # A. 视觉定位
                pacman_pos = extractor.get_pacman_position(current_obs)
                action = 0

                if pacman_pos is not None:
                    # B. 目标管理
                    if current_target_node is None:
                        all_nodes = list(G.nodes)
                        current_target_node = random.choice(all_nodes)
                        print(f"New Target: {current_target_node}")

                    # C. 导航计算
                    action = navigator.get_action(pacman_pos, current_target_node)
                    
                    # 判断到达
                    target_pos = G.nodes[current_target_node]['pos']
                    target_pos = (int(target_pos[0]), int(target_pos[1]))
                    nearest_node = navigator._find_nearest_node(pacman_pos)
                    dist = np.hypot(pacman_pos[0] - target_pos[0], pacman_pos[1] - target_pos[1])
                    if nearest_node == current_target_node and dist < 12:
                        current_target_node = None

                # D. 执行动作，更新画面
                observation, reward, terminated, truncated, info = env.step(action)
                current_obs = observation
                
                if terminated or truncated:
                    print("Game Over. Resetting...")
                    observation, info = env.reset()
                    current_obs = observation
                    current_target_node = None
                    for _ in range(60): env.step(0)


            # 转为 BGR 供 OpenCV 显示
            debug_frame = cv2.cvtColor(current_obs, cv2.COLOR_RGB2BGR)

            # 绘制辅助信息 (红点、绿圈等)
            # 即使暂停了，也重新画一遍，保证视觉连贯
            if not paused:
                # 重新计算一次位置用于绘图 (因为 paused 时不会更新 pacman_pos)
                draw_pos = extractor.get_pacman_position(current_obs)
            else:
                # 暂停时，我们希望定格，所以再算一次也没关系，画面本身没变
                draw_pos = extractor.get_pacman_position(current_obs)

            if draw_pos:
                cv2.circle(debug_frame, draw_pos, 8, (0, 255, 0), 2) # 绿圈
                if current_target_node is not None:
                    t_pos = G.nodes[current_target_node]['pos']
                    t_pos = (int(t_pos[0]), int(t_pos[1]))
                    cv2.line(debug_frame, draw_pos, t_pos, (0, 255, 255), 1) # 黄线

            # 如果暂停中，画个大大的提示
            if paused:
                overlay = debug_frame.copy()
                cv2.rectangle(overlay, (0, 0), (160, 20), (0, 0, 0), -1)
                debug_frame = cv2.addWeighted(overlay, 0.5, debug_frame, 0.5, 0)
                cv2.putText(debug_frame, "[PAUSED] - CLICK TO PICK COLOR", (5, 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            # 更新鼠标回调的引用帧
            cv2.setMouseCallback("Debug View", mouse_callback, debug_frame)

            # 显示
            display_img = cv2.resize(debug_frame, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Debug View", display_img)
            
            # 键盘监听
            key = cv2.waitKey(20) & 0xFF # 20ms 延迟
            
            if key == ord('q') or key == 27: # q 或 ESC 退出
                break
            elif key == ord('p'): # P 键切换暂停
                paused = not paused
                status = "PAUSED" if paused else "RESUMED"
                print(f">>> Game {status}")

    except KeyboardInterrupt:
        print("\nForce Quit.")
    finally:
        env.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()