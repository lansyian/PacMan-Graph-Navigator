import gymnasium as gym
import cv2
import numpy as np
import ale_py

def nothing(x):
    pass

def main():
    env_name = 'ALE/MsPacman-v5'
    env = gym.make(env_name, render_mode='rgb_array')
    observation, _ = env.reset()

    # 预热让 Pacman 出来
    for _ in range(60):
        observation, _, _, _, _ = env.step(0)

    # 创建窗口
    cv2.namedWindow("Tuner")
    cv2.namedWindow("Mask (White = Detected)")

    # === 核心：创建滑动条 ===
    # 默认值设为大概的黄色
    cv2.createTrackbar("R (Red)", "Tuner", 210, 255, nothing)
    cv2.createTrackbar("G (Green)", "Tuner", 164, 255, nothing)
    cv2.createTrackbar("B (Blue)", "Tuner", 74, 255, nothing)
    cv2.createTrackbar("Tolerance", "Tuner", 30, 100, nothing)

    print(">>> 调参指南：")
    print("1. 拖动滑动条，直到 'Mask' 窗口中 Pacman 变成清晰的白色方块。")
    print("2. 尽量让背景（墙壁、豆子）保持黑色。")
    print("3. 记下终端打印出的 [Final Config] 数值。")
    print("4. 按 'p' 暂停/继续游戏。")
    print("5. 按 'q' 退出。")

    paused = False
    
    while True:
        if not paused:
            observation, _, terminated, truncated, _ = env.step(0) # 只是原地不动，方便观察
            if terminated or truncated:
                env.reset()
                for _ in range(60): env.step(0)

        # 1. 获取滑动条的当前值
        r = cv2.getTrackbarPos("R (Red)", "Tuner")
        g = cv2.getTrackbarPos("G (Green)", "Tuner")
        b = cv2.getTrackbarPos("B (Blue)", "Tuner")
        tol = cv2.getTrackbarPos("Tolerance", "Tuner")

        # 2. 构建掩码
        # 注意：Gym 的 observation 是 RGB 格式
        target_color = np.array([r, g, b])
        lower = np.clip(target_color - tol, 0, 255)
        upper = np.clip(target_color + tol, 0, 255)
        
        mask = cv2.inRange(observation, lower, upper)

        # 3. 计算重心（模拟 feature_extractor 的逻辑）
        M = cv2.moments(mask)
        center = None
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            center = (cx, cy)

        # 4. 可视化
        # 把原始画面转 BGR 方便显示
        display_img = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
        
        # 在原始画面上画圈
        if center:
            cv2.circle(display_img, center, 10, (0, 255, 0), 2)
            cv2.putText(display_img, f"Pos: {center}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        else:
            cv2.putText(display_img, "LOST!", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

        # 放大显示
        display_img = cv2.resize(display_img, (0,0), fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        mask_display = cv2.resize(mask, (0,0), fx=3, fy=3, interpolation=cv2.INTER_NEAREST)

        cv2.imshow("Tuner", display_img)
        cv2.imshow("Mask (White = Detected)", mask_display)

        # 打印当前的配置 (方便你复制)
        print(f"\r[Config] RGB=[{r}, {g}, {b}] | Tolerance={tol}  ", end="")

        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            print(f"\n\n>>> 最终确定的配置: RGB=[{r}, {g}, {b}], Tolerance={tol}")
            break
        elif key == ord('p'):
            paused = not paused

    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()