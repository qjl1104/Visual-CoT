import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import json
import os
import numpy as np

# --- 1. 定义真实的多模态数据集 ---
class VisualCotDataset(Dataset):
    def __init__(self, json_path, transform=None):
        """
        Args:
            json_path: 包含标注数据的 json 文件路径
            transform: 图像预处理管线
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.transform = transform
        
        # 如果没有传入 transform，使用 ImageNet 标准预处理
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)), # ResNet 输入尺寸
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
            ])

        # 意图标签映射
        self.intent_map = {"APPROACH": 0, "GRASP": 1, "LIFT": 2, "PLACE": 3}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # --- A. 加载图像 (Visual Input) ---
        img_path = item['image_path']
        # 鲁棒性处理：如果图片不存在（比如在测试环境），生成黑图
        if os.path.exists(img_path):
            image = Image.open(img_path).convert('RGB')
        else:
            image = Image.new('RGB', (224, 224), color='black')
            
        if self.transform:
            image = self.transform(image)
            
        # --- B. 加载本体感知 (Proprioception Input) ---
        state = torch.tensor(item['state'], dtype=torch.float32)
        
        # --- C. 加载 Ground Truth ---
        action = torch.tensor(item['action'], dtype=torch.float32)
        
        intent_str = item.get('intent', "APPROACH")
        # 处理可能出现的未定义标签
        intent_label = self.intent_map.get(intent_str.split()[0], 0) 
        intent_target = torch.tensor(intent_label, dtype=torch.long)
        
        return image, state, action, intent_target

# --- 2. 真实的视觉-本体融合策略网络 ---
class VisualCotPolicy(nn.Module):
    def __init__(self, state_dim=36, action_dim=8, num_intents=4, pretrained=True):
        super(VisualCotPolicy, self).__init__()
        
        print(f"Initializing Visual-CoT Policy with ResNet18 Backbone (Pretrained={pretrained})...")
        
        # --- 视觉流 (Visual Stream) ---
        # 使用 ResNet18 提取图像特征
        resnet = models.resnet18(pretrained=pretrained)
        # 去掉最后的全连接层 (fc)，只保留卷积特征
        self.visual_backbone = nn.Sequential(*list(resnet.children())[:-1])
        # ResNet18 的输出特征维度是 512
        self.visual_dim = 512
        
        # --- 状态流 (Proprioception Stream) ---
        # 将低维状态映射到高维空间
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # --- 多模态融合 (Fusion Module) ---
        # 拼接 视觉特征 (512) + 状态特征 (64)
        self.fusion_dim = self.visual_dim + 64
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1), # 防止过拟合
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # --- 多任务头 (Heads) ---
        # 1. 动作回归头
        self.action_head = nn.Linear(128, action_dim)
        # 2. 意图分类头 (CoT Auxiliary Task)
        self.intent_head = nn.Linear(128, num_intents)

    def forward(self, img, state):
        # 1. 处理图像: [B, 3, 224, 224] -> [B, 512, 1, 1] -> [B, 512]
        vis_feat = self.visual_backbone(img)
        vis_feat = torch.flatten(vis_feat, 1)
        
        # 2. 处理状态: [B, 36] -> [B, 64]
        state_feat = self.state_encoder(state)
        
        # 3. 融合: [B, 576]
        fused = torch.cat([vis_feat, state_feat], dim=1)
        embedding = self.fusion_mlp(fused)
        
        # 4. 预测
        pred_action = self.action_head(embedding)
        pred_intent = self.intent_head(embedding)
        
        return pred_action, pred_intent

# --- 3. 训练主循环 ---
def train():
    # 检查 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Training on device: {device}")
    
    # 初始化数据集和加载器
    # 真实训练时，num_workers 可以设置得更高 (例如 4 或 8)
    try:
        dataset = VisualCotDataset("dataset_with_cot.json")
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    except FileNotFoundError:
        print("错误：未找到 'dataset_with_cot.json'。请先运行 02_generate_cot.py")
        return

    # 初始化模型
    model = VisualCotPolicy(pretrained=True).to(device)
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # 损失函数
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    
    print(f">>> Start Training (Batches: {len(dataloader)})...")
    
    epochs = 10 # 演示用，实际建议 50+
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (imgs, states, actions, intents) in enumerate(dataloader):
            # 搬运数据到 GPU
            imgs, states = imgs.to(device), states.to(device)
            actions, intents = actions.to(device), intents.to(device)
            
            # 前向传播
            pred_actions, pred_intents = model(imgs, states)
            
            # 计算损失
            loss_action = mse_loss(pred_actions, actions)
            loss_intent = ce_loss(pred_intents, intents)
            
            # 联合损失: Action为主，Intent为辅
            loss = loss_action + 0.1 * loss_intent
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} [{batch_idx}/{len(dataloader)}] "
                      f"Loss: {loss.item():.4f} (Act: {loss_action.item():.3f}, Int: {loss_intent.item():.3f})")
        
        avg_loss = total_loss / len(dataloader)
        print(f"=== Epoch {epoch+1} Average Loss: {avg_loss:.4f} ===")

    # 保存权重
    torch.save(model.state_dict(), "visual_cot_policy.pth")
    print("\n[Done] Model saved to visual_cot_policy.pth")

if __name__ == "__main__":
    train()
