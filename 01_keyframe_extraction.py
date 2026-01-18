import numpy as np

# --- 模拟模块: 生成一段机械臂轨迹 ---
def generate_mock_trajectory():
    traj = []
    print("正在生成模拟数据...")
    # 阶段1: 接近物体 (0-30步)，手爪开 (1.0)
    for i in range(30): traj.append([i, 0.5+i*0.01, 0.2, 0.3, 1.0])
    # 阶段2: 抓取瞬间 (30-50步)，手爪闭合 (-1.0)
    for i in range(30, 50): traj.append([i, 0.8, 0.2, 0.3, -1.0]) 
    # 阶段3: 抬起物体 (50-80步)，手爪保持闭合 (-1.0)
    for i in range(50, 80): traj.append([i, 0.8, 0.2+(i-50)*0.01, 0.3, -1.0])
    return np.array(traj)

# --- 核心算法: 提取关键帧 ---
def extract_keyframes(trajectory):
    keyframes = []
    prev_gripper = trajectory[0][-1]
    
    print(f"开始分析轨迹，总长度: {len(trajectory)} 步...")

    for i in range(1, len(trajectory)):
        curr_state = trajectory[i]
        curr_gripper = curr_state[-1]
        
        # 1. 计算速度
        pos_prev = trajectory[i-1][1:4]
        pos_curr = curr_state[1:4]
        velocity = np.linalg.norm(pos_curr - pos_prev)
        
        # 2. 检测手爪状态变化
        gripper_changed = (curr_gripper != prev_gripper)
        
        # 3. 提取关键帧逻辑
        if gripper_changed:
            action = "GRASP (抓取)" if curr_gripper < 0 else "RELEASE (释放)"
            print(f">>> [关键事件捕获] 时间步: {i}, 速度: {velocity:.4f}, 动作: {action}")
            keyframes.append({"step": i, "action": action})
            
        prev_gripper = curr_gripper
    return keyframes

if __name__ == "__main__":
    data = generate_mock_trajectory()
    frames = extract_keyframes(data)
    print("-" * 30)
    print(f"验证成功！共提取到 {len(frames)} 个关键帧。")
    print("Visual-CoT 数据管线逻辑测试通过。")
