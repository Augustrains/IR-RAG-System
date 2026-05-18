# IR-RAG-System 流程图

下面的流程图基于当前仓库的真实代码链路整理，重点覆盖三部分：

- 在线问答主链路：`main.py`、`ir_rag_chat_frontend/app.py`
- 文档构建与索引更新链路：`src/build_index.py`
- 离线数据蒸馏与评测链路：`src/gen_qa/`、`src/evaluation/`

```mermaid
flowchart TD
    A[PDF 教材 / 专业文档] --> B[src/document_split/document_splitter.py<br/>PDF 解析、版面清洗、图表/脚注抽取]
    B --> C[src/document_merge/merge.py<br/>跨页正文与 caption 合并]
    C --> D[src/chunking/chunk_pipeline.py<br/>语义切分 + parent/child 组织 + 图表引用增强]

    D --> E[结构化 chunks]
    D --> F[MongoDB<br/>manual_text]
    E --> G[src/retriever/bm25_retriever.py<br/>双语 BM25 索引]
    E --> H[src/retriever/milvus_retriever.py<br/>双语 Milvus Hybrid 索引]

    subgraph Online["在线问答链路"]
        I[用户问题<br/>CLI 或 Flask 前端] --> J[BM25 检索<br/>原语言直查 + 跨语种补召回]
        I --> K[Milvus Hybrid 检索<br/>Dense + Sparse + RRF]
        G --> J
        H --> K
        J --> L[src/utils.py<br/>merge_docs 去重合并]
        K --> L
        L --> M[src/reranker/bge_m3_reranker.py<br/>BGE-M3 Reranker 精排]
        M --> N[上下文拼接<br/>页码 / 图表 / 脚注]
        N --> O[src/client/llm_local_client.py<br/>本地 Qwen 推理]
        O --> P[src/utils.py<br/>post_processing]
        P --> Q[最终输出<br/>答案 + 引用页码 + 命中文档 + 关联图片]
    end

    subgraph Build["索引构建链路"]
        R[src/build_index.py] --> B
        R --> C
        R --> D
        R --> F
        R --> G
        R --> H
    end

    subgraph Offline["离线蒸馏与优化闭环"]
        D --> S[src/gen_qa/filter.py<br/>筛选核心/扩展文档块]
        S --> T[src/gen_qa/generate_qa.py<br/>教师模型生成 QA]
        T --> U[src/gen_qa/score.py<br/>质量打分]
        U --> V[src/gen_qa/question_generalizer.py<br/>问题泛化]
        V --> W[src/gen_qa/augment_with_negative_samples.py<br/>构造检索/排序训练样本]
        W --> X[LlamaFactory-main<br/>Qwen3-8B + LoRA 问答模型训练]
        W --> Y[RAG-Retrieval-master<br/>Reranker 训练]
        X --> O
        Y --> M
        W --> Z[src/evaluation/*.py<br/>检索器 / 排序器 / 生成效果评测]
        Z --> G
        Z --> H
        Z --> M
        Z --> O
    end
```

## 简化理解

1. 文档先经过 `解析 -> 清洗 -> 跨页修复 -> 语义切分`，形成可检索的结构化 chunk。
2. chunk 一路写入 MongoDB，另一路分别进入双语 BM25 和双语 Milvus。
3. 在线问答时，系统并行做 BM25 与 Milvus 召回，合并后再用 reranker 精排。
4. 精排后的上下文送入本地大模型生成答案，最后补齐页码、命中文档和图表引用。
5. 离线部分再基于 chunk 自动生成 QA、做问题泛化、构造训练集，用于持续优化问答模型和 reranker。

## 对应代码入口

- 在线问答主入口：[main.py](/hard_data1/user/yangguobin/LLM/IR-RAG-System-main/main.py)
- Web 前端入口：[ir_rag_chat_frontend/app.py](/hard_data1/user/yangguobin/LLM/IR-RAG-System-main/ir_rag_chat_frontend/app.py)
- 文档构建入口：[src/build_index.py](/hard_data1/user/yangguobin/LLM/IR-RAG-System-main/src/build_index.py)
- Milvus 检索器：[src/retriever/milvus_retriever.py](/hard_data1/user/yangguobin/LLM/IR-RAG-System-main/src/retriever/milvus_retriever.py)
- BM25 检索器：[src/retriever/bm25_retriever.py](/hard_data1/user/yangguobin/LLM/IR-RAG-System-main/src/retriever/bm25_retriever.py)
