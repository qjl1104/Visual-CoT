import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image
import json
import os
import numpy as np

class VisualCotDataset(Dataset):
    def __init__(self, json_path, stack_size=3, chunk_size=20):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
        self.stack_size = stack_size
        self.chunk_size = chunk_size
        
        # 【工业级优化 1】：动作归一化 (Action Normalization)
        # 统计数据集中所有动作的极值，将其映射到 [-1, 1] 之间，防止大数值关节主导 Loss
        all_actions = np.array([item['action'] for item in self.data])
        self.action_min = torch.tensor(all_actions.min(axis=0), dtype=torch.float32)
        self.action_max = torch.tensor(all_actions.max(axis=0), dtype=torch.float32)
        self.action_scale = self.action_max - self.action_min
        self.action_scale[self.action_scale == 0] = 1.0 # 防止除零
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.intent_map = {"APPROACH": 0, "GRASP": 1, "LIFT": 2, "PLACE": 3}

    def normalize_action(self, act):
        act_tensor = torch.tensor(act, dtype=torch.float32)
        return (act_tensor - self.action_min) / self.action_scale * 2.0 - 1.0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # --- 获取历史图像 (Frame Stacking) ---
        frames = []
        for i in range(idx - self.stack_size + 1, idx + 1):
            curr_idx = max(0, i)
            img_path = self.data[curr_idx]['image_path']
            img = Image.open(img_path).convert('RGB') if os.path.exists(img_path) else Image.new('RGB', (224, 224))
            frames.append(self.transform(img))
        stacked_imgs = torch.cat(frames, dim=0)
        
        item = self.data[idx]
        state = torch.tensor(item['state'], dtype=torch.float32)
        intent = torch.tensor(self.intent_map.get(item.get('intent', 'APPROACH'), 0), dtype=torch.long)
        
        # --- 【工业级优化 2】：轨迹越界保护 (Episode Boundary Padding) ---
        action_chunk = []
        # 假设数据中有 episode_id 字段。如果没有，默认为 0 (针对单轨迹 Demo)
        current_episode = item.get('episode_id', 0) 
        last_valid_act = self.normalize_action(item['action'])
        
        for i in range(idx, idx + self.chunk_size):
            if i < len(self.data):
                next_ep = self.data[i].get('episode_id', 0)
                if next_ep == current_episode:
                    # 同一条轨迹，正常读取
                    act = self.normalize_action(self.data[i]['action'])
                    last_valid_act = act
                else:
                    # 跨越了轨迹边界，停止读取未来动作，用最后一个有效动作 Padding
                    act = last_valid_act
            else:
                # 超出数据集总长度
                act = last_valid_act
                
            action_chunk.append(act)
            
        action_seq = torch.stack(action_chunk) # [chunk_size, action_dim]
        return stacked_imgs, state, action_seq, intent

class VisualCotPolicy(nn.Module):
    # 网络结构与上一版相同，保留修改过的 ResNet (9通道输入) 和 Chunking 输出头
    def __init__(self, state_dim=36, action_dim=8, stack_size=3, chunk_size=20):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        
        weights = ResNet18_Weights.IMAGENET1K_V1
        resnet = models.resnet18(weights=weights)
        self.visual_backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        self.visual_backbone[0] = nn.Conv2d(3 * stack_size, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, 256), nn.ReLU(), nn.Linear(256, 128))
        self.fusion_mlp = nn.Sequential(nn.Linear(512 + 128, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, 256), nn.ReLU())
        
        self.action_head = nn.Linear(256, action_dim * chunk_size)
        self.intent_head = nn.Linear(256, 4)

    def forward(self, img, state):
        vis_feat = torch.flatten(self.visual_backbone(img), 1)
        state_feat = self.state_encoder(state)
        emb = self.fusion_mlp(torch.cat([vis_feat, state_feat], dim=1))
        
        pred_actions = self.action_head(emb).view(-1, self.chunk_size, self.action_dim)
        pred_intent = self.intent_head(emb)
        return pred_actions, pred_intent

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = VisualCotDataset("dataset_with_cot.json", stack_size=3, chunk_size=20)
    
    # 保存归一化参数供推理时使用
    torch.save({'action_min': dataset.action_min, 'action_scale': dataset.action_scale}, 'norm_stats.pth')
    
    train_size = int(0.8 * len(dataset))
    train_idx, val_idx = torch.utils.data.random_split(range(len(dataset)), [train_size, len(dataset) - train_size])
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=16, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=16)

    model = VisualCotPolicy(stack_size=3, chunk_size=20).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion_act = nn.MSELoss()
    criterion_int = nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()
        for batch_i, (imgs, states, action_chunks, intents) in enumerate(train_loader):
            imgs, states, action_chunks, intents = imgs.to(device), states.to(device), action_chunks.to(device), intents.to(device)
            pred_actions, pred_intents = model(imgs, states)
            loss = criterion_act(pred_actions, action_chunks) + 0.1 * criterion_int(pred_intents, intents)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        
        print(f"Epoch {epoch+1} finished.")
    torch.save(model.state_dict(), "visual_cot_policy_chunked.pth")

if __name__ == "__main__":
    train()
