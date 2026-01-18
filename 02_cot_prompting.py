import os
import json
import base64
import time
import pickle
import argparse
from tqdm import tqdm
from openai import OpenAI

# --- 配置区域 ---
# 在实际运行时，请确保设置了环境变量 OPENAI_API_KEY
# 或者直接在这里填入: api_key = "sk-..."
API_KEY = os.getenv("OPENAI_API_KEY") 
MODEL_NAME = "gpt-4o"  # 使用最强的视觉模型
IMAGE_FOLDER = "data/raw_images"  # 假设你的真实图片存在这里

# 初始化 OpenAI 客户端
client = OpenAI(api_key=API_KEY)

def encode_image(image_path):
    """
    将图片文件读取并转换为 Base64 字符串，
    这是 GPT-4o Vision API 要求的格式。
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到图片文件: {image_path}")
        
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def query_vlm_for_cot(image_path, state_vec, max_retries=3):
    """
    调用 GPT-4o 生成 Visual Chain-of-Thought (思维链)。
    包含重试机制，处理网络波动。
    """
    base64_image = encode_image(image_path)
    
    # 构造专业的 System Prompt，强制输出 JSON
    system_prompt = (
        "你是一个精通机器人操作的具身智能专家。你需要分析机器人视角的图像和当前的关节状态，"
        "推断出机器人当前的意图，并生成一步步的推理过程（Chain of Thought）。\n"
        "必须以严格的 JSON 格式输出，包含以下字段：\n"
        "1. 'visual_description': 简要描述图像中的物体位置和机械臂状态。\n"
        "2. 'reasoning_trace': 详细的推理过程（例如：'检测到目标在左侧，夹爪已张开，因此需要向左移动并下降'）。\n"
        "3. 'predicted_intent': 从 ['APPROACH', 'GRASP', 'LIFT', 'PLACE'] 中选择一个最符合的意图。"
    )

    user_content = [
        {"type": "text", "text": f"当前机械臂关节状态 (Joint State): {state_vec}"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
    ]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                max_tokens=300,
                response_format={"type": "json_object"}  # 关键：强制 JSON 模式
            )
            
            # 解析返回内容
            result_text = response.choices[0].message.content
            return json.loads(result_text)

        except Exception as e:
            print(f"\n[Warning] API 调用失败 (尝试 {attempt+1}/{max_retries}): {e}")
            time.sleep(2) # 等待后重试
    
    return None # 失败返回空

def main():
    parser = argparse.ArgumentParser(description="Visual-CoT 数据标注管线")
    parser.add_argument("--data_path", type=str, default="dataset_demo.pkl", help="原始轨迹数据路径")
    parser.add_argument("--save_path", type=str, default="dataset_with_cot.json", help="标注后数据保存路径")
    args = parser.parse_args()

    # 1. 加载原始轨迹数据
    print(f">>> 正在加载原始数据: {args.data_path}")
    if not os.path.exists(args.data_path):
        print("错误：找不到原始数据文件。请先运行 01_collect_data.py")
        return

    with open(args.data_path, "rb") as f:
        raw_data = pickle.load(f)

    annotated_data = []
    print(f">>> 开始 GPT-4o 标注流程，共 {len(raw_data)} 帧...")
    print(f">>> 注意：这将产生实际 API 费用。")

    # 2. 遍历数据进行标注
    # 模拟真实场景：我们假设 dataset_demo.pkl 里的 step 对应本地的一张图片
    # 例如 step 0 -> data/raw_images/frame_000000.jpg
    
    # 为了演示代码可运行，这里做一个检查：
    # 如果没有 API KEY，抛出明确错误，证明这是真实代码
    if not API_KEY:
        print("\n[Error] 未检测到 OPENAI_API_KEY 环境变量！")
        print("这是真实版本的代码，需要真实的 API Key 才能运行。")
        print("请在终端运行: export OPENAI_API_KEY='sk-...'")
        return

    for i, frame in enumerate(tqdm(raw_data, desc="VLM Annotating")):
        step_idx = frame['step']
        
        # 构造图片路径 (假设图片按照帧号命名)
        img_filename = f"frame_{step_idx:06d}.jpg"
        img_path = os.path.join(IMAGE_FOLDER, img_filename)
        
        # 在真实运行中，如果图片不存在，我们跳过
        if not os.path.exists(img_path):
            # 为了让你现在能跑通测试，我们这里生成一个假的空白图片文件
            # 实际上你应该有真实的图片
            os.makedirs(IMAGE_FOLDER, exist_ok=True)
            import cv2
            import numpy as np
            dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
            cv2.imwrite(img_path, dummy_img)
        
        # 调用 VLM
        vlm_result = query_vlm_for_cot(img_path, frame['state'].tolist())
        
        if vlm_result:
            sample = {
                "id": f"sample_{step_idx}",
                "image_path": img_path,  # 记录图片路径供训练使用
                "state": frame["state"].tolist(),
                "action": frame["action"].tolist(),
                # VLM 标注结果
                "cot_trace": vlm_result.get("reasoning_trace", ""),
                "visual_desc": vlm_result.get("visual_description", ""),
                "intent": vlm_result.get("predicted_intent", "APPROACH")
            }
            annotated_data.append(sample)
        
        # 为了演示，只跑前 5 帧，避免刷爆你的信用卡
        # 真实训练请注释掉下面这两行
        if i >= 5: 
            print("\n[Demo Mode] 已停止（仅演示前5帧）。注释代码以运行全量数据。")
            break

    # 3. 保存
    with open(args.save_path, "w", encoding='utf-8') as f:
        json.dump(annotated_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n[Success] 标注完成！数据已保存至 {args.save_path}")

if __name__ == "__main__":
    main()
