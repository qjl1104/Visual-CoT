这是一份为您重新编写的、完全基于最新 Visual-CoT 代码库（包含 Action Chunking、Frame Stacking 和时序融合等工业级特性）的 README.md。已完全排除了 FinSight 项目的干扰。你可以直接复制以下内容替换掉错误的 README.md：Markdown# Visual-CoT: Multi-Modal Robotic Manipulation with Chain-of-Thought Distillation
# 基于视觉思维链蒸馏的时序增强具身控制系统

[![Isaac Lab](https://img.shields.io/badge/Sim-NVIDIA_Isaac_Lab-green)](https://developer.nvidia.com/isaac-sim)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch_2.0-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Project Status**: Phase 4 (Action Chunking & Sim-to-Real Ensembling) Completed.
> **Hardware**: Verified on NVIDIA RTX Series (WSL2/Ubuntu).

**Visual-CoT** 旨在解决端到端具身智能模型（End-to-End VLA）缺乏可解释性及大模型在边缘端推理延迟过高的问题。本项目创新性地提出了一条**“大模型思维链标注 + 轻量级模型多任务蒸馏 + 动作分块（Action Chunking）”**的工业级落地管线。

通过在 NVIDIA Isaac Lab 中采集海量数据，利用 GPT-4o 离线生成思维链（CoT）作为中间监督信号，训练轻量级 ResNet 策略网络，最终在端侧实现了 **30Hz** 的高平滑度实时闭环控制。

---

## 🚀 核心特性 (Key Features)

* **🧠 Multi-Task Knowledge Distillation (多任务知识蒸馏)**: 摒弃高延迟的大模型端侧推理，将 GPT-4o 的系统性推理能力（System 2）转化为轻量级网络（Modified ResNet-18）的快速直觉反应（System 1）。联合优化“动作回归（MSE）”与“意图分类（CrossEntropy）”。
* **⏱️ Temporal-Aware Perception (时序感知增强)**: 采用 **Frame Stacking（多帧堆叠）** 机制，重构网络底层输入维度，隐式捕捉机械臂与物体的速度、加速度等动态物理特征。
* **📦 Action Chunking & Episode Protection (动作分块与轨迹保护)**: 策略网络单次预测未来多步动作序列（Chunking）。数据集层面引入了严格的**动作归一化（Action Normalization）**与**轨迹越界保护（Episode Boundary Padding）**，防止跨任务数据污染。
* **🌊 Exponential Temporal Ensembling (指数级时序平滑融合)**: 在推理部署阶段，构建了专用的时序融合引擎，通过指数衰减权重对相互重叠的预测动作序列进行加权平均，彻底消除机械臂端侧高频控制时的物理抖动。
* **⚡ High-Throughput Simulation (高吞吐仿真)**: 基于 NVIDIA Isaac Lab 构建，支持数千个环境的并行数据采集（90k+ FPS）。

---

## 🛠️ 系统架构 (System Pipeline)

项目包含从数据采集、VLM 标注到策略训练与真机部署的完整闭环：

```mermaid
graph TD
    A[Phase 1: Isaac Lab 并行仿真] -->|State & Action| B(Raw Trajectories);
    B --> C[Phase 2: GPT-4o 视觉思维链生成];
    C -->|Visual Desc + Reasoning + Intent| D(Annotated Dataset);
    D -->|Frame Stacking & Normalization| E[Phase 3: 多任务蒸馏训练];
    E -->|Visual-CoT Policy| F[Phase 4: 边缘端部署];
    F -->|Action Chunking| G[Temporal Ensembling 时序融合];
    G -->|30Hz Smooth Control| H((Real Robot / Sim))
    
    style A fill:#d4f1f4,stroke:#333
    style C fill:#f4e7d4,stroke:#333
    style E fill:#d4f4d7,stroke:#333
    style G fill:#f3d4f4,stroke:#333
▶️ 快速开始 (Quick Start)1. 数据采集 (Data Collection)在 Isaac Lab 仿真环境中并行采集机械臂操控数据。Bash# 启动 headless 模式进行高速并行采集
python 01_collect_data.py
2. 生成思维链标注 (CoT Generation)调用 GPT-4o Vision 接口，为原始轨迹自动打上意图（Intent）和推理过程（Reasoning Trace）标签。Bash# 请确保已设置环境变量: export OPENAI_API_KEY="sk-..."
python 02_generate_cot.py
3. 策略网络训练 (Policy Training)利用 Frame Stacking 和 Action Chunking 机制，训练多任务轻量级策略网络。内置数据归一化与验证集监控。Bashpython 03_train_policy.py
4. 实时融合推理 (Real-time Ensembling Inference)模拟真实部署环境，通过 ActionEnsembler 验证多步预测的加权平滑效果。Bashpython 04_inference_ensembling.py
📊 性能表现 (Performance Metrics)指标 (Metric)结果 (Value)备注 (Note)仿真吞吐量 (Sim Speed)90k+ FPS基于 RTX 5080 (4096 Envs)端侧控制频率 (Control Freq)30 Hz+纯视觉输入下的稳定闭环推理延迟 (Inference Latency)< 10 ms相比 7B VLA 模型降低 95% 以上动作预测视野 (Chunk Size)20 Steps覆盖约 0.67 秒的未来动作规划👤 作者 (Author)Jiale Qian (钱家乐)Email: 12011626@mail.sustech.edu.cnGithub: qjl1104Institution: Southern University of Science and Technology (SUSTech)📝 许可证 (License)本项目基于 MIT License 开源。
