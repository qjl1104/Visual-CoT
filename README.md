Markdown
# 🦅 FinSight: Enterprise GraphRAG & Agentic Reasoning System

FinSight 是一个面向复杂金融文档（如招股书、授信合同、审计报告）的下一代智能审查与问答系统。

不同于传统仅依赖向量相似度的 RAG，FinSight 深度融合了 **GraphRAG（知识图谱增强）**、**Hybrid Search（多路混合检索）** 以及 **Self-Reflective Agent（自反思智能体）**，能够完美兼顾微观的“细节条款”查询与宏观的“业务全景”理解，实现零幻觉的金融级合规审查。

## 🚀 核心特性 (Key Features)

* **双层记忆索引 (Dual Memory Index)**: 底层结合 FAISS 向量数据库与 Neo4j 图数据库，实现非结构化语义与结构化实体关系的统一存储。
* **GraphRAG 宏观感知**: 基于 DeepSeek-V3 构建高精度知识图谱，并运用 Neo4j GDS 的 Leiden 算法进行社区聚类，自动生成宏观业务摘要 (Community Summaries)。
* **混合检索与深度重排序 (Hybrid Search & Reranking)**: 采用“向量 (BGE-Small) + 图谱实体 + 社区摘要”的三路召回架构，并引入 BGE-Reranker-Base 交叉编码器 (Cross-Encoder) 进行精准打分去噪。
* **自反思智能体 (Self-Reflective Agent)**: 基于 LangChain 构筑原生的“检索 -> 裁判评分 -> 查询重写” System 2 慢思考闭环，有效解决长程复杂逻辑问题的回答遗漏。
* **工业级落地特性**: 
  * **增量更新 (Incremental Update)**: 基于锚点探测 (Anchor Detection) 的图谱局部刷新。
  * **数据治理 (Entity Resolution)**: 基于 LLM 的同义实体自动对齐。
  * **知识蒸馏 (Knowledge Distillation)**: 包含从超大参数模型 (Teacher) 提取 CoT 数据微调小模型 (Student) 的完整实验管线。

## 🛠️ 安装指南 (Installation)

**1. 环境准备**
确保已安装 Python 3.10+ 和 Neo4j Desktop (或使用 Docker 部署 Neo4j)。

```bash
git clone [https://github.com/your-username/FinSight.git](https://github.com/your-username/FinSight.git)
cd FinSight
pip install -r requirements.txt
2. 配置环境变量
复制项目根目录下的 .env.example 为 .env (注意已被 .gitignore 忽略，需手动创建)，并填入配置信息：

Ini, TOML
# Neo4j 数据库配置
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here

# LLM API 配置 (本项目基于 DeepSeek 构建)
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxx
🏃‍♂️ 快速开始 (Quick Start)
本项目采用模块化设计，完美拆分了数据治理流水线与应用推理层。

阶段一：构建索引流水线 (Indexing Pipeline)
按顺序运行以下脚本，完成从原始 PDF 到双层检索索引的构建：

python 1_chunking.py —— 文档解析与 Token 级切分。

python 2_extract_triplets.py —— DeepSeek 驱动的三元组 (实体/关系) 抽取。

python 3_import_graph.py —— 将 JSON 三元组写入 Neo4j。

python 4_community_detection.py —— 运行 Leiden 算法进行社区划分。

python 5_generate_summaries.py —— 为子图社区生成自然语言摘要。

python 6_build_vector_index.py —— 构建 FAISS 本地向量库 (BGE-Small)。

阶段二：应用启动 (Run Application)
启动基于 Streamlit 构建的可视化审查面板：

Bash
streamlit run app.py
阶段三：核心算法与高级特性验证 (Advanced Capabilities)
独立运行以下脚本，深入体验 FinSight 的底层算法优势：

实体对齐: python 9_entity_resolution.py (清洗并合并图谱中的同义实体)。

自动化评测: python 10_evaluate.py (运行 FinBench 测试集，验证召回率提升)。

Reranker 去噪: python 11_rerank.py (观察 Cross-Encoder 如何精准过滤无关文档)。

增量入库: python 12_incremental_update.py (模拟新知识入库时的锚点挂载与局部摘要刷新)。

自反思流: python 13_agent_feedback_loop.py (体验 Agent 发现证据不足时自动 Rewrite Query 的过程)。

模型蒸馏: python 14_distillation_pipeline.py (对比 Zero-Shot 与 Teacher-Student 蒸馏后的抽取表现)。

📄 技术架构图 (Architecture)
代码段
graph TD
    A[PDF 招股书/合同] -->|PyPDF & TikToken| B(文本切片 Chunks)
    
    %% 索引构建层
    subgraph Indexing Pipeline
        B -->|Embedding| C[FAISS 向量库]
        B -->|LLM Extraction| D[实体与关系提取]
        D -->|Cypher| E[Neo4j 知识图谱]
        E -->|Leiden Algorithm| F[社区聚类检测]
        F -->|LLM Summarization| G[社区宏观摘要]
    end

    %% 推理层
    subgraph Agentic Reasoning Workflow
        User[用户提问] --> H[Hybrid Search 混合检索]
        H -->|Vector Search| C
        H -->|Graph Traversal| E
        H -->|Macro Context| G
        
        C & E & G --> I[BGE-Reranker 交叉编码打分]
        I --> J{裁判模型评估 Grade}
        J -->|Evidence Insufficient| K[Query Rewrite 检索词重写]
        K --> H
        J -->|Evidence Sufficient| L[DeepSeek 生成最终回答]
    end
📜 License
MIT License
