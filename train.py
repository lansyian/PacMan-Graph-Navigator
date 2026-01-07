import gymnasium as gym
import pickle
import os
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor # 用于记录训练数据
from pacman_graph_env import PacmanGraphEnv
import ale_py
from gymnasium.wrappers import TimeLimit

def main():
    # 0. 清理旧日志
    print(">>> 正在初始化...")

    # 1. 加载地图数据
    graph_path = "data/pacman_graph.pkl"
    if not os.path.exists(graph_path):
        print("Error: 找不到 data/pacman_graph.pkl")
        return

    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    # 2. 创建环境
    # 我们需要两个环境：
    # env: 用于训练 (AI 会尝试乱走来探索)
    # eval_env: 用于考试 (AI 只选它认为最好的路)
    env = PacmanGraphEnv(G, render_mode='rgb_array')
    env = TimeLimit(env, max_episode_steps=2000)
    env = Monitor(env) 

    eval_env = PacmanGraphEnv(G, render_mode='rgb_array')
    eval_env = TimeLimit(eval_env, max_episode_steps=2000)
    eval_env = Monitor(eval_env)

    # 3. 定义路径
    models_dir = "models/dqn_pacman"
    log_dir = "logs"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 4. 初始化 DQN 模型参数 (针对困难环境优化)
    model = DQN(
        "MlpPolicy", 
        env, 
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=5e-5,    
        buffer_size=100000,    
        learning_starts=5000,  
        batch_size=128,       
        gamma=0.99,             
        train_freq=4,          
        target_update_interval=1000, 
        exploration_fraction=0.3, 
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05, 
    )
    
    # 5. 回调函数
    # 定期保存检查点
    checkpoint_callback = CheckpointCallback(
        save_freq=50000, 
        save_path=models_dir,
        name_prefix="dqn_step"
    )
    # 评估回调函数
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=models_dir,
        log_path=log_dir,
        eval_freq=10000,
        deterministic=True, 
        render=False,
        n_eval_episodes=5
    )

    # learning
    try:
        model.learn(
            total_timesteps=500000,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n训练被手动中断...")
        model.save(f"{models_dir}/dqn_interrupted")
        print("已保存中断前的模型。")
    
    # save
    model.save(f"{models_dir}/dqn_final")
    env.close()
    eval_env.close()
    print("训练结束。")

if __name__ == "__main__":
    main()