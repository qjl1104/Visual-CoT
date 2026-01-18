import argparse

# --- 1. 最先导入 AppLauncher ---
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visual-CoT Data Collection")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

# --- 2. 启动仿真器 ---
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- 3. 导入 Isaac Lab 模块 ---
import torch
import pickle
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import FrankaCubeLiftEnvCfg

def main():
    # 配置环境
    env_cfg = FrankaCubeLiftEnvCfg()
    env_cfg.scene.num_envs = 4
    env_cfg.sim.device = "cuda:0"
    
    print(">>> 正在初始化 Isaac Lab 环境 (Franka Cube Lift)...")
    env = ManagerBasedRLEnv(cfg=env_cfg)

    dataset = []
    
    print(">>> 开始采集数据...")
    obs, _ = env.reset()
    
    total_steps = 100
    for step in range(total_steps):
        # 生成随机动作
        actions = torch.rand((env.num_envs, env.action_space.shape[1]), device=env.device) * 2 - 1
        
        # [关键修正] 接收 5 个返回值 (obs, rew, terminated, truncated, extras)
        out = env.step(actions)
        next_obs = out[0]
        # rewards = out[1]
        # terminated = out[2]
        # truncated = out[3]
        # extras = out[4]
        
        # 提取数据
        state_vec = obs["policy"][0].cpu().numpy() 
        action_vec = actions[0].cpu().numpy()
        
        frame_data = {
            "step": step,
            "state": state_vec, 
            "action": action_vec,
            "description": "Random exploration" 
        }
        dataset.append(frame_data)
        
        obs = next_obs
        
        if step % 20 == 0:
            print(f"采集进度: {step}/{total_steps} 步")

    # 保存
    save_path = "dataset_demo.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(dataset, f)
    
    print(f"\n[成功] 数据采集完成！已保存至 {save_path}")
    print(f"数据集大小: {len(dataset)} 帧")
    
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
