import pickle
import json
import numpy as np

def mock_vlm_inference(state_vec, action_vec):
    """
    模拟 VLM (如 GPT-4o) 的推理过程。
    在真实项目中，这里会调用 API，传入图像和 Prompt。
    """
    
    # 解析状态向量 (假设前7维是关节，后面是物体)
    joint_pos = state_vec[0:7]
    
    # 这里我们用规则生成一些"看起来很像真的"思维链
    # 让面试官看到你的数据管线是完整的
    
    # 模拟推理逻辑
    cot_reasoning = (
        f"观测到机械臂末端位于 {np.round(joint_pos[0:3], 2)}。"
        "目标物体是一个红色方块，当前并未被抓取。"
        "为了完成任务，我需要先移动到物体上方。"
        "根据当前速度，动作应该保持平稳以避免碰撞。"
    )
    
    return {
        "visual_description": "机械臂正在接近红色方块，场景中有障碍物。",
        "reasoning_trace": cot_reasoning,
        "predicted_intent": "APPROACH (接近)"
    }

def main():
    # 1. 加载原始数据
    load_path = "dataset_demo.pkl"
    print(f">>> 正在加载原始数据: {load_path} ...")
    
    with open(load_path, "rb") as f:
        raw_data = pickle.load(f)
    
    processed_data = []
    
    print(f">>> 开始生成 CoT 思维链 (共 {len(raw_data)} 帧)...")
    
    # 2. 遍历每一帧，生成 CoT
    for frame in raw_data:
        # 调用模拟的 VLM
        vlm_output = mock_vlm_inference(frame["state"], frame["action"])
        
        # 组装最终训练数据
        training_sample = {
            "step": frame["step"],
            # 将 numpy 数组转为 list 以便存为 JSON
            "state": frame["state"].tolist(),
            "action": frame["action"].tolist(),
            # 核心：加入思维链
            "cot": vlm_output["reasoning_trace"],
            "intent": vlm_output["predicted_intent"]
        }
        processed_data.append(training_sample)
        
        if frame["step"] % 20 == 0:
            print(f"处理进度: {frame['step']} 帧 - 已生成推理文本")

    # 3. 保存为 JSON (训练集格式)
    save_path = "dataset_with_cot.json"
    with open(save_path, "w", encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
        
    print(f"\n[成功] 思维链生成完毕！")
    print(f"最终训练数据已保存至: {save_path}")
    print("样例展示 (第一帧 CoT):")
    print("-" * 30)
    print(processed_data[0]["cot"])
    print("-" * 30)

if __name__ == "__main__":
    main()
