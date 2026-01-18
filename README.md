# Visual-CoT: Multi-Modal Robotic Manipulation with Chain-of-Thought Reasoning

**Visual-CoT** 是一个基于 **NVIDIA Isaac Lab** 的具身智能项目，旨在通过引入多模态大模型（VLM）生成的思维链（CoT）作为中间推理步骤，增强机器人操作策略的可解释性。

## 🚀 核心特性 (Key Features)
- **高性能仿真**: 基于 NVIDIA Isaac Lab 和 **RTX 5080 (WSL2)** 实现 **90,000+ steps/s** 并行采集。
- **思维链驱动**: 将操作任务分解为“观测-推理-意图”的可解释步骤。
- **多任务学习**: 联合训练动作回归 (Action) 与意图分类 (Intent)。

## 🛠️ 流程 (Pipeline)
1. **数据采集**: 在仿真环境中采集 Franka 机械臂的状态与动作。
2. **CoT 生成**: 模拟 VLM 生成推理文本和意图标签。
3. **策略训练**: 训练多任务神经网络 (Action Loss < 0.1, Intent Acc > 99%)。

## 💻 环境依赖
- OS: Ubuntu 24.04 (WSL2)
- GPU: NVIDIA RTX 5080
- Python: 3.10+ / PyTorch / Isaac Lab

## ▶️ 使用方法
\`\`\`bash
python 01_collect_data.py   # 采集数据
python 02_generate_cot.py   # 生成标注
python 03_train_policy.py   # 训练模型
\`\`\`

## 📝 License
MIT License.
