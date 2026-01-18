import os
import json
import base64
import time
import pickle
import argparse
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor

MODEL_NAME = "gpt-4o-2024-05-13"
IMAGE_FOLDER = "data/raw_images"

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def query_vlm_worker(client, frame):
    """单条数据处理逻辑，便于多线程调用"""
    step_idx = frame['step']
    img_path = os.path.join(IMAGE_FOLDER, f"frame_{step_idx:06d}.jpg")
    
    if not os.path.exists(img_path):
        return None

    try:
        base64_image = encode_image(img_path)
        # 严格的 System Prompt 约束
        system_prompt = (
            "你是一个具身智能专家。请分析图像和关节状态，以 JSON 格式输出：\n"
            "1. 'reasoning_trace': 推理过程。\n"
            "2. 'predicted_intent': 必须在 ['APPROACH', 'GRASP', 'LIFT', 'PLACE'] 中。"
        )
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": f"State: {frame['state'].tolist()}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ],
            response_format={"type": "json_object"}
        )
        
        res = json.loads(response.choices[0].message.content)
        # 格式校验：防止 VLM 乱写 Intent
        valid_intents = ['APPROACH', 'GRASP', 'LIFT', 'PLACE']
        intent = res.get("predicted_intent", "APPROACH").upper()
        if intent not in valid_intents: intent = "APPROACH"

        return {
            "step": step_idx,
            "image_path": img_path,
            "state": frame["state"].tolist(),
            "action": frame["action"].tolist(),
            "cot_trace": res.get("reasoning_trace", ""),
            "intent": intent
        }
    except Exception as e:
        return None

def main():
    # ... (前面的参数解析逻辑保持一致) ...
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    with open("dataset_demo.pkl", "rb") as f:
        raw_data = pickle.load(f)

    print(f">>> 开始多线程标注 (Max Workers: 5)...")
    annotated_data = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(query_vlm_worker, client, f) for f in raw_data]
        for future in tqdm(futures, desc="VLM Annotating"):
            res = future.result()
            if res: annotated_data.append(res)

    with open("dataset_with_cot.json", "w", encoding='utf-8') as f:
        json.dump(annotated_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
