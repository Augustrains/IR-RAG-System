# IR-RAG-System

一个面向信息检索教材与专业知识文档的 RAG 系统。项目聚焦教材类 PDF 在真实落地中常见的几个难点：版面结构复杂、图表信息利用不足、检索与生成链路缺少闭环优化。围绕这些问题，系统构建了从文档解析、结构化切分、索引构建、双路召回、重排序，到前端问答展示与效果评测的完整工程流程。

这个项目的目标不是简单做一个“能回答问题”的 Demo，而是尽可能回答得更准、更可追溯，并且让检索器、排序器和生成模型都可以独立评测和持续优化。

## 项目定位

本项目聚焦于 Information Retrieval 教材场景，希望系统同时具备以下能力：

- 检索更稳定：兼顾关键词匹配与语义召回
- 回答更可信：支持引用页码、命中文档和关联图表
- 面向教材更友好：处理章节结构、脚注、图表标题和跨页内容
- 具备迭代能力：支持索引更新、训练数据构建和效果评测闭环

## 项目亮点

### 1. 面向教材 PDF 的结构化解析

项目不是直接对 PDF 粗暴切块，而是先进行针对教材场景的版面清洗与结构恢复，包括：

- 页眉页脚、页码等噪声过滤
- 图表标题识别与跨行合并
- 同一视觉行碎片重组
- 跨页内容修复，减少正文和图表说明的断裂

这样做的意义在于，后续检索与生成依赖的是更接近原始语义结构的文档块，而不是简单的固定长度文本片段。

### 2. 结构先验 + 语义信息结合的切分策略

项目在切分阶段没有只采用通用滑窗，而是结合教材结构做语义切分：

- 区分 section heading、sentence、list item、caption 等角色
- 综合结构规则、词汇变化和语义相似度决定切分边界
- 构建 parent/child 层级块，兼顾召回范围与上下文完整性

这让知识块更符合教材的表达逻辑，也为后续召回与重排提供了更好的输入单元。

### 3. 双语检索设计，适配术语混杂场景

IR 教材中经常出现中英文术语混合表达，例如 `Boolean retrieval`、`postings list`、`precision/recall`。为此，项目在检索层做了双语设计：

- BM25 按中英文分别建立索引
- Milvus 按中英文分别建立 hybrid collection
- 查询时先做原语言检索，再做跨语种补召回
- 最终对多路结果统一去重和合并

相比单语检索，这种设计更适合教材、论文和技术文档等术语跨语言混用的知识库场景。

### 4. 混合检索 + 精排，提高上下文质量

主链路采用“多路召回 + 重排序”方案：

- 稀疏检索：BM25 负责术语、关键词和精确表达匹配
- 稠密/混合检索：Milvus 负责语义召回
- 候选融合：统一去重、合并多路结果
- 精排：使用本地 BGE-M3 reranker 对候选文档做相关性排序

这套方案的重点不在于堆模型，而在于利用不同检索方式的互补性，提升教材问答这种“既需要术语精确匹配，也需要语义理解”的场景表现。

### 5. 图表增强检索与可追溯回答

教材类问题往往不只依赖正文，也可能依赖某个 Figure/Table 附近的说明文字。项目为此做了两层增强：

- 在解析阶段抽取图表 caption、图片路径、页码等元信息
- 在切块阶段识别正文中的图表引用，并把图表短描述注入正文块

最终系统输出的不只是答案，还会补充：

- 引用页码
- 命中的文档片段
- 关联图表信息
- 本地图片路径

这使得系统不仅关注“能不能生成”，也关注“回答是否可验证、可解释”。

### 6. 不只是问答 Demo，而是完整优化闭环

除了在线问答主链路，项目还实现了持续优化所需的离线能力：

- 从教材内容自动构建 QA 对
- 对问题做泛化扩写，提高训练集覆盖度
- 构建 reranker 训练数据
- 提供检索器、排序器、生成模型和整体链路的独立评测脚本

这说明项目已经从“可运行原型”推进到了“可训练、可评测、可迭代”的工程系统。

## 系统架构

```text
PDF教材
  -> 文档清洗与版面规则学习
  -> 跨页合并与图表/脚注保留
  -> 结构化语义分块
  -> MongoDB 存储结构化文档
  -> BM25 双语索引
  -> Milvus 双语 Hybrid 索引
  -> 用户问题输入
  -> BM25召回 + Milvus召回
  -> 候选去重合并
  -> BGE-M3 Reranker 精排
  -> 本地大模型生成回答
  -> 后处理输出答案、页码、关联图表与命中文档
```

## 核心流程

### 1. 文档构建链路

文档处理与索引更新主流程位于 [`src/build_index.py`](./src/build_index.py)，主要包括：

- 解析 PDF 并清洗版面噪声
- 修复跨页断裂的正文与图表单元
- 进行结构化语义切分
- 将文档块同步到 MongoDB
- 更新 BM25 索引与 Milvus 向量库

### 2. 在线问答链路

主入口位于 [`main.py`](./main.py)，当前流程为：

1. BM25 召回候选文档
2. Milvus hybrid 检索召回候选文档
3. 合并与去重
4. 用 BGE-M3 reranker 精排
5. 将候选块拼接成带页码、图表和脚注信息的上下文
6. 调用本地 Qwen3-8B 生成回答
7. 对回答做后处理，提取引用编号、页码、图表与命中文档

### 3. 前端展示链路

项目包含一个可直接演示的 Flask 前端：

- 类 ChatGPT 风格会话界面
- 支持历史会话保存与恢复
- 支持展示引用片段
- 支持根据本地 `image_path` 展示关联图表


## 环境依赖 / 安装说明

### 基础环境

建议使用以下运行环境：

- Python 3.10 及以上
- Linux 环境或具备 CUDA 支持的服务器环境
- MongoDB 6.x 或兼容版本
- Milvus 2.x 或兼容版本
- 可用的 GPU 环境，用于 embedding、reranker 和本地生成模型推理

### Python 依赖安装

项目依赖较多，且包含文档处理、检索、训练和前端相关模块。可以先在虚拟环境中安装基础依赖：

```bash
pip install -r ir_rag_chat_frontend/requirements.txt
```

如果后续需要训练或运行第三方训练工程，还需要分别安装：

```bash
pip install -r RAG-Retrieval-master/requirements.txt
pip install -r LlamaFactory-main/requirements/metrics.txt
```

实际使用时，可根据自己只运行“问答主链路”“评测脚本”还是“训练脚本”按需裁剪依赖。

### 数据库与向量库准备

系统依赖 MongoDB 存储结构化文档，依赖 Milvus 存储向量索引与 hybrid 检索结构。开始前需要确保：

- MongoDB 服务已启动并可连接
- Milvus 服务或本地数据库模式已正确配置
- `src/client/mongodb_config.py` 与 `src/path.py` 中的相关路径和连接参数已与本地环境匹配

### 模型准备

项目默认依赖以下几类本地模型：

- 生成模型：Qwen3-8B
- 检索 embedding 模型：BGE-M3
- 排序模型：BGE reranker 或其微调版本

相关模型路径由 [`src/path.py`](./src/path.py) 统一管理。使用前需要根据本地环境确认：

- 模型文件已下载到本地
- 路径配置与实际目录一致
- 显存足够支持当前推理设置

### 数据准备

项目默认假设已经具备以下数据目录：

- 原始 PDF 教材
- 文档处理中间产物目录
- 评测数据目录
- QA 训练与测试数据目录

如果是首次运行，通常需要先准备 PDF 原始文件，再通过构建脚本生成后续索引与中间数据。

## 快速开始

完成上面的环境与依赖准备后，可以按以下步骤快速运行项目。

### 1. 构建文档与索引

先执行文档处理与索引构建流程：

```bash
python3 src/build_index.py
```

这一步会完成：

- PDF 清洗与结构恢复
- 文档分块
- MongoDB 文档同步
- BM25 索引更新
- Milvus 向量索引更新

### 2. 启动命令行问答

索引构建完成后，可以直接运行命令行问答主入口：

```bash
python3 main.py
```

系统会依次执行多路检索、候选重排、上下文拼接和本地模型生成，并输出答案、引用页码和相关图表信息。

### 3. 启动前端演示

如果希望通过 Web 界面体验问答流程，可运行前端服务：

```bash
python3 ir_rag_chat_frontend/app.py
```

启动后可在浏览器中查看会话界面、历史记录、引用片段以及图表展示效果。

### 4. 运行评测脚本

项目的评测脚本统一放在 [`src/evaluation`](./src/evaluation) 下，可分别对检索器、排序器、生成模型和整体链路进行评测。

常用流程包括：

- 使用 `build_rag_eval_inputs.py` 生成评测输入
- 使用 `eval_qwen_from_rag_inputs.py` 跑 base 或 LoRA 问答模型
- 使用 `eval_retriever.py` 与 `eval_rank.py` 做模块级评测


## 训练数据生成

当前仓库中，训练生成模型和训练 reranker 的数据构建脚本都已经存在，但 README 之前没有把这部分流程单独展开。若你希望复现实验或继续迭代模型，通常需要按下面两条链路分别准备数据。

### 1. 生成问答模型训练数据

这一部分对应 [`src/gen_qa`](./src/gen_qa) 和 [`LlamaFactory-main/examples/train_lora/qwen3_ir_rag_lora_sft.yaml`](./LlamaFactory-main/examples/train_lora/qwen3_ir_rag_lora_sft.yaml)。

推荐顺序如下：

1. 先对切分后的文档块做 QA 可用性过滤  
   输入通常是文档切分结果，脚本会把样本分为 `core / extra / drop`，并输出到：
   - `data/processed_docs/qa_filter_outputs/core_docs.jsonl`
   - `data/processed_docs/qa_filter_outputs/extra_docs.jsonl`

```bash
python3 src/gen_qa/filter.py
```

2. 基于过滤结果生成初始 QA 对  
   `core` 文档默认生成 4 个 QA，`extra` 文档默认生成 1 个 QA，输出：
   - `data/qa_pairs/qa_pair.jsonl`

```bash
python3 src/gen_qa/generate_qa.py
```

3. 对 QA 做质量打分  
   该步骤会对 `qa_pair.jsonl` 中的问答逐条评分，输出：
   - `data/qa_pairs/qa_score.jsonl`

```bash
python3 src/gen_qa/score.py
```

4. 对高分问题做泛化并切分训练/测试集  
   默认仅对 `score >= 4` 的问题做泛化，最终导出：
   - `data/qa_pairs/train_qa_pair.json`
   - `data/qa_pairs/test_qa_pair.json`

```bash
python3 src/gen_qa/question_generalizer.py
```

5. 构造成 LLaMA-Factory 可直接训练的 Alpaca 格式  
   [`src/gen_qa/build_alpaca_augmented_dataset.py`](./src/gen_qa/build_alpaca_augmented_dataset.py) 默认读取 `test_qa_keywords.jsonl`，脚本里已经预留了 train/test 两套路径，通常需要根据当前要生成的是训练集还是测试集切换输入输出路径。产物包括：
   - `data/qa_pairs/train_augmented_master.jsonl` 或 `test_augmented_master.jsonl`
   - `data/qa_pairs/train_augmented_alpaca.jsonl` 或 `test_augmented_alpaca.jsonl`

```bash
python3 src/gen_qa/build_alpaca_augmented_dataset.py
```

6. 可选：混入拒答负样本  
   [`src/gen_qa/augment_with_negative_samples.py`](./src/gen_qa/augment_with_negative_samples.py) 会把通用闲聊或越域问题整理成 `output="无答案"` 的样本，并并入最终训练/测试集，输出：
   - `data/qa_pairs/train_augmented_with_neg_alpaca.jsonl`
   - `data/qa_pairs/test_augmented_with_neg_alpaca.jsonl`

```bash
python3 src/gen_qa/augment_with_negative_samples.py
```

完成后，可将最终 Alpaca 数据挂到 `LlamaFactory-main/data/ir_rag/` 下，并结合现有配置启动 LoRA 训练。仓库中现成配置默认使用：

- 训练集：`train_augmented_with_neg_alpaca.jsonl`
- 测试集：`test_augmented_with_neg_alpaca.jsonl`
- 配置文件：`LlamaFactory-main/examples/train_lora/qwen3_ir_rag_lora_sft.yaml`

### 2. 生成 reranker 训练数据

这一部分对应 [`src/gen_qa/generate_rank.py`](./src/gen_qa/generate_rank.py)、[`scripts/prepare_rag_retrieval_reranker_data.py`](./scripts/prepare_rag_retrieval_reranker_data.py) 和 [`scripts/train_rag_retrieval_bge_reranker_v2_m3.sh`](./scripts/train_rag_retrieval_bge_reranker_v2_m3.sh)。

推荐顺序如下：

1. 基于 QA 数据构造排序标签原始集  
   [`src/gen_qa/generate_rank.py`](./src/gen_qa/generate_rank.py) 会读取：
   - `data/qa_pairs/train_qa_pair.json`
   - `data/qa_pairs/test_qa_pair.json`
   - `data/qa_pairs/qa_generalized.jsonl`
   - `data/qa_pairs/train_augmented_master.jsonl`
   - `data/qa_pairs/test_augmented_master.jsonl`

   然后调用 BM25 + Milvus 召回候选文档，再由 LLM 给候选打 `1 / 0 / -1` 相关性标签，最终输出：
   - `data/qa_pairs/reranker/train_rank_labels.jsonl`
   - `data/qa_pairs/reranker/val_rank_labels.jsonl`
   - `data/qa_pairs/reranker/test_rank_labels.jsonl`

```bash
python3 src/gen_qa/generate_rank.py
```

2. 转换成 RAG-Retrieval 训练格式  
   训练脚本默认使用 grouped 格式数据。执行：

```bash
python3 scripts/prepare_rag_retrieval_reranker_data.py --write-pointwise
```

默认会把 `-1/0/1` 标签映射为 `0/1/2`，并导出到：

- `data/qa_pairs/rag_retrieval_reranker/train_grouped.jsonl`
- `data/qa_pairs/rag_retrieval_reranker/val_grouped.jsonl`
- `data/qa_pairs/rag_retrieval_reranker/test_grouped.jsonl`
- `data/qa_pairs/rag_retrieval_reranker/train_pointwise.jsonl`
- `data/qa_pairs/rag_retrieval_reranker/val_pointwise.jsonl`
- `data/qa_pairs/rag_retrieval_reranker/test_pointwise.jsonl`

3. 启动 reranker 训练  
   现成配置文件 [`scripts/configs/rag_retrieval_bge_reranker_v2_m3.yaml`](./scripts/configs/rag_retrieval_bge_reranker_v2_m3.yaml) 默认读取上一步导出的 `train_grouped.jsonl` 与 `val_grouped.jsonl`：

```bash
bash scripts/train_rag_retrieval_bge_reranker_v2_m3.sh
```

训练输出默认写入：

- `output/rag_retrieval_bge_reranker_v2_m3`

### 3. 两条数据链路的关系

- 生成模型训练数据的核心目标是让 Qwen 学会基于教材知识块稳定回答问题，以及在越域问题上输出“无答案”。
- reranker 训练数据的核心目标是让排序模型学会区分“强相关证据块”和“仅部分相关块”。
- 两者都基于同一批教材切分块衍生，但训练目标不同，因此数据构造方式也不同。
- 如果你只想跑在线问答，不需要先走完整训练流程；如果你要复现实验结果，建议把这两条数据链路都补齐。

## 项目目录

```text
IR-RAG-System/
├── main.py                          # 命令行问答主入口
├── src/
│   ├── build_index.py               # 文档构建与索引更新主流程
│   ├── document_split/              # PDF解析、清洗、图表标题识别
│   ├── document_merge/              # 跨页内容修复
│   ├── chunking/                    # 语义切分、图表引用增强
│   ├── retriever/                   # BM25 / Milvus 检索器
│   ├── reranker/                    # BGE-M3 / Qwen3 reranker
│   ├── gen_qa/                      # QA生成、评分、泛化、训练集构建
│   ├── evaluation/                  # 检索器、排序器、生成模型评测
│   └── client/                      # LLM、MongoDB 等客户端封装
├── ir_rag_chat_frontend/            # Web 演示前端
├── data/                            # 本地数据、索引、中间产物
├── models/                          # 本地模型目录（不上传）
├── output/                          # 训练输出目录（不上传）
├── LlamaFactory-main/               # 训练相关第三方工程
└── RAG-Retrieval-master/            # 排序器训练相关第三方工程
```

## 技术栈

- Python
- PyMuPDF / `fitz` 进行 PDF 解析
- MongoDB 存储结构化文档
- Milvus 构建混合检索索引
- BGE-M3 作为 embedding / hybrid retrieval 基座
- BGE-M3 Cross-Encoder 作为 reranker
- Qwen3-8B 作为本地生成模型
- Flask + HTML/CSS/JS 构建演示前端

## 我的工作

项目实现覆盖了完整的 RAG 工程链路，主要包括：

- 面向复杂 PDF 的文档解析、清洗与结构恢复
- 结构化语义切分与 parent/child 文档组织
- 双语 BM25 与双语 Milvus hybrid 检索
- 本地 reranker 重排与上下文构造
- 本地生成模型推理与答案后处理
- 检索器、排序器、生成模型与整体链路评测

## 效果评测

当前项目的评测分为四个层次：检索器评测、排序器评测、生成模型评测，以及整体链路评测。相关脚本统一放在 [`src/evaluation`](./src/evaluation) 下，报告样例放在 [`src/evaluation/report`](./src/evaluation/report) 下。

### 检索器评测

检索器评测使用统一测试集，对 BM25 与 Milvus 的新旧版本进行横向对比，指标包括 `Recall@1/3/5`、`MRR`、`Hit@1/3`。

#### BM25：新版本 vs 旧版本

| Metric | bm25 | old_bm25 | Delta |
|---|---:|---:|---:|
| Recall@1 | 0.4242 | 0.3917 | +0.0325 |
| Recall@3 | 0.5072 | 0.4861 | +0.0212 |
| Recall@5 | 0.5507 | 0.5078 | +0.0430 |
| MRR | 0.6862 | 0.6425 | +0.0437 |
| Hit@1 | 0.6289 | 0.5849 | +0.0440 |
| Hit@3 | 0.7233 | 0.6981 | +0.0252 |

结论：新版 `bm25` 在所有核心指标上全面优于 `old_bm25`，说明项目在双语索引、文本归一化和去重聚合上的优化有效提高了前几名候选的覆盖率和排序质量。

#### Milvus：新版本 vs 旧版本

| Metric | milvus | old_milvus | Delta |
|---|---:|---:|---:|
| Recall@1 | 0.4583 | 0.4095 | +0.0487 |
| Recall@3 | 0.6077 | 0.4751 | +0.1326 |
| Recall@5 | 0.6512 | 0.5013 | +0.1499 |
| MRR | 0.7635 | 0.6840 | +0.0796 |
| Hit@1 | 0.6730 | 0.6478 | +0.0252 |
| Hit@3 | 0.8428 | 0.7170 | +0.1258 |

结论：新版 `milvus` 在所有核心指标上都优于 `old_milvus`，说明项目在双语 hybrid 检索、跨语种补召回、扩召回和 parent 聚合上的改造能够更稳定地提升前排候选质量。

### 排序器评测

排序器评测用于比较基座 reranker 与任务微调后的 reranker，指标包括 `Hits@1/3/5`、`Recall@5/10`、`MRR`、`MAP`、`NDCG@5/10`。

| 模型 | Hits@1 | Hits@3 | Hits@5 | Recall@5 | Recall@10 | MRR | MAP | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 微调前 `bge-reranker-v2-m3` | 0.9057 | 0.9560 | 0.9811 | 0.5763 | 0.7533 | 0.9330 | 0.7949 | 0.8153 | 0.8300 |
| 微调后模型 | 0.9686 | 1.0000 | 1.0000 | 0.6224 | 0.8077 | 0.9822 | 0.8655 | 0.8994 | 0.9071 |

结论：微调后的排序模型在全部指标上都优于原始基座模型，说明当前任务数据上的监督微调有效增强了相关文档的前列排序能力，也提升了整体重排质量。

### 生成模型评测

生成模型评测主要比较两类模型：

- 基础模型：`Qwen3-8B`
- 微调模型：`Qwen3-8B + LoRA`

评测关注的核心指标包括：

- `Exact Match`
- `Token F1`
- `ROUGE-L`
- `No Answer Accuracy`
- `Generation Time`
- `End-to-End Time`

从整体链路结果看，问答模型微调后在以下方面带来了稳定收益：

- 更严格地依据上下文作答
- 更少输出与教材证据不够贴合的泛化回答
- 在“无答案”场景下边界判断更稳定
- 更适配当前项目的问答任务形式与输出风格

### 整体链路评测

为了比较当前系统不同模块组合的收益，项目对七种设置做了统一对比，重点观察“检索、排序、生成”三部分协同优化后的最终效果。

| 编号 | 实验设置 | EM | Token F1 | ROUGE-L | 无答案准确率 | 平均端到端时延(s) |
|---|---|---:|---:|---:|---:|---:|
| 1 | 只用基础问答模型 | 0.287 | 0.462 | 0.438 | 0.611 | 1.84 |
| 2 | 原版检索器 + 原始问答模型 | 0.401 | 0.589 | 0.561 | 0.712 | 2.37 |
| 3 | 修改后检索器 + 原始问答模型 | 0.458 | 0.647 | 0.621 | 0.768 | 2.51 |
| 4 | 修改后检索器 + 旧排序器 + 原始问答模型 | 0.486 | 0.673 | 0.649 | 0.782 | 2.79 |
| 5 | 修改后检索器 + 新排序器 + 原始问答模型 | 0.531 | 0.718 | 0.694 | 0.821 | 2.73 |
| 6 | 修改后检索器 + 旧排序器 + 微调后问答模型 | 0.552 | 0.736 | 0.711 | 0.836 | 2.96 |
| 7 | 修改后检索器 + 新排序器 + 微调后问答模型 | 0.604 | 0.781 | 0.758 | 0.874 | 2.91 |

整体结论：

- 引入检索后，系统效果相较纯基础问答模型有显著提升
- 修改后的检索器能够稳定提高有效上下文质量
- 微调后的排序器进一步提升了证据排序质量
- 微调后的问答模型进一步增强了最终回答质量与无答案识别能力
- 最优方案“修改后检索器 + 新排序器 + 微调后问答模型”在效果上整体领先，同时端到端时延仍保持在可接受范围内

这组结果说明，当前项目的优势并不只是“接入了 RAG”，而是形成了从检索、排序到生成的完整协同优化链路，并且这些优化可以通过独立评测和整体评测共同体现出来。

## 项目总结

本项目的核心价值在于，它不是只依赖单一大模型能力，而是围绕教材类文档场景，把文档处理、检索、排序、生成和评测组织成了一条完整链路。系统在保证回答可追溯性的同时，也为后续持续优化检索器、排序器和问答模型提供了明确的工程基础。

## 后续可优化方向

- 引入更细粒度的 chunk 质量评估与自动回流机制
- 在图表增强基础上继续探索真正的多模态检索与回答
- 进一步优化 query rewrite / query expansion 策略
- 增加在线日志与检索可观测性，方便定位失败 case
- 将本地脚本式流程逐步整理为更标准的服务化架构

## 说明

出于仓库体积与安全考虑，以下内容不会上传到 GitHub：

- 本地模型权重
- 训练输出与检查点
- PDF 原始文件与生成数据
- 本地索引与数据库文件
- 缓存、日志与 API Key 配置
