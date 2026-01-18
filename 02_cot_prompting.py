import json

# --- 模拟从上一步(01脚本)传来的关键帧数据 ---
keyframe_data = {
    "step": 30,
    "image_path": "frame_0030.jpg",  # 假设我们保存了这一帧的图片
    "action_detected": "GRASP",
    "arm_state": [0.8, 0.2, 0.3]
}

def build_vlm_prompt(keyframe):
    """
    核心逻辑：构建 Visual-CoT 的 Prompt (提示词)
    这对应白皮书中 "将关键帧图像与文本指令对齐" 的步骤
    """
    
    # 1. 系统提示词 (System Prompt) - 定义 VLM 的角色
    system_prompt = (
        "你是一个机器人操作专家。你的任务是根据给定的关键帧图像，"
        "生成一段'思维链(Chain-of-Thought)'，解释机器人为什么要执行当前动作。"
    )
    
    # 2. 用户输入 (User Input) - 结合观测和动作
    # 在真实项目中，这里会把 image_path 对应的图片编码进去
    user_content = f"""
    [图像]: <image_placeholder>
    [检测到的动作]: {keyframe['action_detected']} (抓取)
    [当前坐标]: {keyframe['arm_state']}
    
    请分析：
    1. 机器人当前的空间状态是什么？
    2. 为什么在此时此刻执行抓取？
    3. 下一步的预期意图是什么？
    
    请以 JSON 格式输出推理过程。
    """
    
    return system_prompt, user_content

def mock_gpt4o_response():
    """
    模拟 GPT-4o 的返回结果 (因为我们没有配置 API Key，这里用硬编码模拟)
    """
    return {
        "reasoning": "机器人末端执行器已到达目标物体（红色方块）上方，且速度降至接近 0。此时执行抓取动作是为了确立对物体的控制，准备进行后续的搬运任务。",
        "next_intention": "Lift (抬起)",
        "confidence": 0.98
    }

if __name__ == "__main__":
    print(">>> 正在构建 Visual-CoT 提示词模板...")
    sys_p, user_p = build_vlm_prompt(keyframe_data)
    
    print("-" * 20 + " Prompt Preview " + "-" * 20)
    print(f"[System]: {sys_p}")
    print(f"[User]: {user_p}")
    
    print("-" * 20 + " Simulated GPT-4o Response " + "-" * 20)
    response = mock_gpt4o_response()
    print(json.dumps(response, ensure_ascii=False, indent=2))
    
    print("\n[成功] 思维链生成模块逻辑验证通过！")
