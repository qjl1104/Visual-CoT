import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np

# --- 1. 定义数据集 ---
class RobotReasoningDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # 简单的意图编码 (String -> Int)
        self.intent_map = {"APPROACH (接近)": 0, "GRASP (抓取)": 1, "LIFT (抬起)": 2}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 输入: 状态 (36维)
        state = torch.tensor(item['state'], dtype=torch.float32)
        
        # 目标1: 动作 (8维)
        action = torch.tensor(item['action'], dtype=torch.float32)
        
        # 目标2: 意图 (分类标签) - 这里的 intent 是我们在上一步模拟生成的
        intent_str = item.get('intent', "APPROACH (接近)")
        intent_label = torch.tensor(self.intent_map.get(intent_str, 0), dtype=torch.long)
        
        return state, action, intent_label

# --- 2. 定义 Visual-CoT 策略网络 ---
class VisualCoTPolicy(nn.Module):
    def __init__(self, state_dim=36, action_dim=8, num_intents=3):
        super(VisualCoTPolicy, self).__init__()
        
        # 共享感知骨干网络 (Backbone)
        # 在真实项目中，这里通常是 ResNet 或 ViT 处理图像
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # 任务头 1: 动作预测 (Action Head) - 负责"手"
        self.action_head = nn.Linear(128, action_dim)
        
        # 任务头 2: 意图预测 (Reasoning Head) - 负责"脑"
        # 这就是 Visual-CoT 的精髓：让网络理解"为什么"
        self.intent_head = nn.Linear(128, num_intents)

    def forward(self, x):
        features = self.backbone(x)
        pred_action = self.action_head(features)
        pred_intent = self.intent_head(features)
        return pred_action, pred_intent

# --- 3. 训练流程 ---
def train():
    # 配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> 使用设备: {device}")
    
    # 加载数据
    dataset = RobotReasoningDataset("dataset_with_cot.json")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # 初始化模型
    model = VisualCoTPolicy().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 定义损失函数
    criterion_action = nn.MSELoss()        # 回归损失
    criterion_intent = nn.CrossEntropyLoss() # 分类损失
    
    print(">>> 开始训练 Visual-CoT 策略网络...")
    model.train()
    
    # 模拟训练 50 个 Epoch
    for epoch in range(50):
        total_loss = 0
        
        for states, actions, intents in dataloader:
            states, actions, intents = states.to(device), actions.to(device), intents.to(device)
            
            # 前向传播
            pred_actions, pred_intents = model(states)
            
            # 计算多任务损失
            loss_a = criterion_action(pred_actions, actions)
            loss_i = criterion_intent(pred_intents, intents)
            
            # Loss 加权融合 (动作精度优先)
            loss = loss_a + 0.5 * loss_i
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/50], Loss: {total_loss/len(dataloader):.4f} (Action Loss: {loss_a.item():.4f}, Intent Loss: {loss_i.item():.4f})")

    # 保存模型
    torch.save(model.state_dict(), "visual_cot_policy.pth")
    print("\n[成功] 模型训练完成！已保存至 visual_cot_policy.pth")

if __name__ == "__main__":
    train()
