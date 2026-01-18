import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image
import json
import os

class VisualCotDataset(Dataset):
    def __init__(self, json_path, stack_size=3):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.stack_size = stack_size
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.intent_map = {"APPROACH": 0, "GRASP": 1, "LIFT": 2, "PLACE": 3}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 简化的时序叠帧：取当前帧及前 N 帧，不足则重复第一帧
        frames = []
        for i in range(idx - self.stack_size + 1, idx + 1):
            curr_idx = max(0, i)
            img_path = self.data[curr_idx]['image_path']
            img = Image.open(img_path).convert('RGB') if os.path.exists(img_path) else Image.new('RGB', (224, 224))
            frames.append(self.transform(img))
        
        # [C*S, H, W] -> 例如 [9, 224, 224]
        stacked_imgs = torch.cat(frames, dim=0)
        item = self.data[idx]
        state = torch.tensor(item['state'], dtype=torch.float32)
        action = torch.tensor(item['action'], dtype=torch.float32)
        intent = torch.tensor(self.intent_map.get(item['intent'], 0), dtype=torch.long)
        
        return stacked_imgs, state, action, intent

class VisualCotPolicy(nn.Module):
    def __init__(self, state_dim=36, action_dim=8, stack_size=3):
        super().__init__()
        # 修改 ResNet 第一层以接收多帧输入
        weights = ResNet18_Weights.IMAGENET1K_V1
        resnet = models.resnet18(weights=weights)
        self.visual_backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # 关键修改：调整输入通道数 (3 * stack_size)
        original_conv = resnet.conv1
        self.visual_backbone[0] = nn.Conv2d(3 * stack_size, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, 256), nn.ReLU(), nn.Linear(256, 128))
        self.fusion_mlp = nn.Sequential(nn.Linear(512 + 128, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 128))
        self.action_head = nn.Linear(128, action_dim)
        self.intent_head = nn.Linear(128, 4)

    def forward(self, img, state):
        vis_feat = torch.flatten(self.visual_backbone(img), 1)
        state_feat = self.state_encoder(state)
        fused = torch.cat([vis_feat, state_feat], dim=1)
        emb = self.fusion_mlp(fused)
        return self.action_head(emb), self.intent_head(emb)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_dataset = VisualCotDataset("dataset_with_cot.json")
    
    # 划分训练集与验证集 (8:2)
    train_size = int(0.8 * len(full_dataset))
    train_idx, val_idx = torch.utils.data.random_split(range(len(full_dataset)), [train_size, len(full_dataset) - train_size])
    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=16, shuffle=True)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=16)

    model = VisualCotPolicy().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion_act = nn.MSELoss()
    criterion_int = nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()
        for imgs, states, actions, intents in train_loader:
            imgs, states, actions, intents = imgs.to(device), states.to(device), actions.to(device), intents.to(device)
            p_act, p_int = model(imgs, states)
            loss = criterion_act(p_act, actions) + 0.2 * criterion_int(p_int, intents)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        
        # 每个 Epoch 进行简单验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, states, actions, intents in val_loader:
                p_act, _ = model(imgs.to(device), states.to(device))
                val_loss += criterion_act(p_act, actions.to(device)).item()
        print(f"Epoch {epoch+1} | Val MSE: {val_loss/len(val_loader):.4f}")

    torch.save(model.state_dict(), "visual_cot_policy_v2.pth")

if __name__ == "__main__":
    train()
