import os
import json
import base64
import time
import pickle
import argparse
from tqdm import tqdm
from openai import OpenAI

# --- 配置区域 ---
# MODEL_NAME = "gpt-4o" 
MODEL_NAME = "gpt-4o-2024-05-13" # 建议指定具体版本，保证效果稳定
IMAGE_FOLDER = "data/raw_images"

def encode_image(image_path):
    """
    将图片文件读取并转换为 Base64 字符串。
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到图片文件: {image_path}")
        
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def query_vlm_for_cot(client, image_path, state_vec, max_retries=3):
    """
    调用 GPT-4o 生成 Visual Chain-of-Thought (思维链)。
    """
    base64_image = encode_image(image_path)
    
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
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content
            return json.loads(result_text)

        except Exception as e:
            print(f"\n[Warning] API 调用失败 (尝试 {attempt+1}/{max_retries}): {e}")
            time.sleep(2)
    
    return None

def main():
    parser = argparse.ArgumentParser(description="Visual-CoT 数据标注管线")
    parser.add_argument("--data_path", type=str, default="dataset_demo.pkl", help="原始轨迹数据路径")
    parser.add_argument("--save_path", type=str, default="dataset_with_cot.json", help="标注后数据保存路径")
    parser.add_argument("--limit", type=int, default=-1, help="仅处理前N帧用于测试 (默认-1表示处理所有)")
    args = parser.parse_args()

    # 0. 检查 API Key (移到这里更安全)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n[Error] 未检测到 OPENAI_API_KEY 环境变量！")
        print("请在终端运行: export OPENAI_API_KEY='sk-...'")
        return

    # 初始化客户端
    client = OpenAI(api_key=api_key)

    # 1. 加载原始轨迹数据
    print(f">>> 正在加载原始数据: {args.data_path}")
    if not os.path.exists(args.data_path):
        print(f"错误：找不到原始数据文件 {args.data_path}")
        return

    with open(args.data_path, "rb") as f:
        raw_data = pickle.load(f)

    annotated_data = []
    total_frames = len(raw_data)
    
    # 如果设置了 limit，则截取数据
    if args.limit > 0:
        raw_data = raw_data[:args.limit]
        print(f">>> [测试模式] 仅处理前 {args.limit} 帧 (总数: {total_frames})")
    else:
        print(f">>> [全量模式] 准备处理所有 {total_frames} 帧")

    print(f">>> 注意：这将产生实际 API 费用。")

    # 2. 遍历数据进行标注
    success_count = 0
    for i, frame in enumerate(tqdm(raw_data, desc="VLM Annotating")):
        step_idx = frame['step']
        img_filename = f"frame_{step_idx:06d}.jpg"
        img_path = os.path.join(IMAGE_FOLDER, img_filename)
        
        # --- 关键修改：不要生成假图片 ---
        if not os.path.exists(img_path):
            # 如果图片真的不存在，应该跳过，而不是造假数据
            # 否则 VLM 会对着黑图产生幻觉，污染数据集
            # print(f"[Skip] 缺少图片: {img_path}") # 可选：打印日志
            continue
        
        # 调用 VLM
        vlm_result = query_vlm_for_cot(client, img_path, frame['state'].tolist())
        
        if vlm_result:
            sample = {
                "id": f"sample_{step_idx}",
                "image_path": img_path,
                "state": frame["state"].tolist(),
                "action": frame["action"].tolist(),
                "cot_trace": vlm_result.get("reasoning_trace", ""),
                "visual_desc": vlm_result.get("visual_description", ""),
                "intent": vlm_result.get("predicted_intent", "APPROACH")
            }
            annotated_data.append(sample)
            success_count += 1

    # 3. 保存
    with open(args.save_path, "w", encoding='utf-8') as f:
        json.dump(annotated_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n[Success] 标注完成！成功: {success_count}/{len(raw_data)}")
    print(f"数据已保存至 {args.save_path}")

if __name__ == "__main__":
    main()
