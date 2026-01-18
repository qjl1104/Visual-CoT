import numpy as np

def generate_mock_trajectory():
    traj = []
    print("正在生成模拟数据...")
    # 增加更多动态变化
    for i in range(30): traj.append([i, 0.5+i*0.01, 0.2, 0.3, 1.0]) # 接近
    for i in range(30, 40): traj.append([i, 0.8, 0.2, 0.3, -1.0]) # 抓取
    for i in range(40, 60): traj.append([i, 0.8, 0.2+(i-40)*0.02, 0.3, -1.0]) # 快速抬起
    return np.array(traj)

def extract_keyframes(trajectory, vel_threshold=0.015):
    keyframes = []
    prev_gripper = trajectory[0][-1]
    prev_velocity = 0
    
    print(f"开始分析轨迹，总长度: {len(trajectory)} 步...")

    for i in range(1, len(trajectory)):
        curr_state = trajectory[i]
        pos_prev = trajectory[i-1][1:4]
        pos_curr = curr_state[1:4]
        
        # 1. 计算当前速度和加速度 (启发式特征)
        velocity = np.linalg.norm(pos_curr - pos_prev)
        acceleration = abs(velocity - prev_velocity)
        
        # 2. 检测手爪状态变化
        curr_gripper = curr_state[-1]
        gripper_changed = (curr_gripper != prev_gripper)
        
        # 3. 复合提取逻辑: 手爪变化 OR 显著的运动转折 (加速度)
        if gripper_changed or acceleration > vel_threshold:
            reason = "GRIPPER" if gripper_changed else "MOTION_SHIFT"
            action = "GRASP/RELEASE" if gripper_changed else "MOVE_TRANSITION"
            print(f">>> [事件捕获] 步数: {i}, 原因: {reason}, 动作: {action}")
            keyframes.append({"step": i, "reason": reason, "action": action})
            
        prev_gripper = curr_gripper
        prev_velocity = velocity
    return keyframes

if __name__ == "__main__":
    data = generate_mock_trajectory()
    frames = extract_keyframes(data)
    print("-" * 30)
    print(f"验证成功！共提取到 {len(frames)} 个关键帧。")
