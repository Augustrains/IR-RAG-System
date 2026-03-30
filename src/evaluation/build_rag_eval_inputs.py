"""
构建 RAG 问答评测输入。

功能：
1. 从指定测试集读取问题与标准答案
2. 可自由启用/禁用检索器与排序器
3. 可自由选择使用哪些检索器与排序器
4. 对每条问题执行：
   - 检索
   - 去重合并
   - 可选排序
   - 拼接成最终给大模型的完整 prompt
5. 输出逐条 JSONL 结果，包含问题、标准答案、检索文档、最终 prompt 与耗时
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from langchain_core.documents import Document

from src.path import Qwen3_Reranker_path, bge_reranker_tuned_model_path
from src.retriever.bm25_retriever import BM25 as CurrentBM25
from src.retriever.milvus_retriever import MilvusRetriever as CurrentMilvusRetriever
from src.retriever.old_bm25_retriever import BM25 as OldBM25
from src.retriever.old_milvus_retriever import MilvusRetriever as OldMilvusRetriever
from src.reranker.bge_m3_reranker import BGEM3ReRanker
from src.reranker.qwen3_4B_reranker import Qwen3ReRankervLLM
from src.utils import merge_docs


BASE_EVAL_DIR = Path("/root/autodl-tmp/IR-RAG-System/src/evaluation")
DEFAULT_DATASET_PATH = (
    "/root/autodl-tmp/IR-RAG-System/LlamaFactory-main/data/ir_rag/test_augmented_with_neg_alpaca.jsonl"
)
DEFAULT_OUTPUT_DIR = BASE_EVAL_DIR / "rag_eval_inputs"
DEFAULT_RETRIEVAL_TOPK = 3
DEFAULT_RERANK_TOPK = 5

RETRIEVER_CHOICES = ("bm25", "old_bm25", "milvus", "old_milvus")
RERANKER_CHOICES = ("bge_m3", "qwen3_4b")

LLM_CHAT_PROMPT = """
### 检索信息
{context}

### 任务
你是《Introduction to Information Retrieval》教材的问答助手。你必须综合考虑每条检索结果中的正文、页码、图表信息、脚注信息，再回答用户问题。

### 回答要求
1. 只能依据给定的检索信息作答，不允许编造。
2. 如果检索信息不足以支持回答，直接输出“无答案”。
3. 如果答案依赖图表或脚注，在最终答案中自然说明，并给出对应引用编号。
4. 不要输出思考过程、分析过程、解释过程。
5. 不要输出 <think>、</think> 或任何额外标签。
6. 必须严格按照以下格式输出，且只能输出这一段：
<answer>
这里写最终回答
</answer>

### 用户问题
{query}
""".strip()


@dataclass
class EvalSample:
    sample_id: str
    instruction: str
    question: str
    answer: str
    system_prompt: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 RAG 评测输入 JSONL")
    parser.add_argument("--dataset-path", type=str, default=DEFAULT_DATASET_PATH, help="测试集 JSONL 路径")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="输出目录根路径")
    parser.add_argument("--run-name", type=str, default="", help="可选，自定义运行名")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 条，0 表示全量")

    parser.add_argument(
        "--enable-retriever",
        action="store_true",
        help="是否启用检索；不启用时 context 为空字符串",
    )
    parser.add_argument(
        "--retrievers",
        type=str,
        default="bm25,milvus",
        help="逗号分隔的检索器列表，可选 bm25,old_bm25,milvus,old_milvus",
    )
    parser.add_argument("--retrieval-topk", type=int, default=DEFAULT_RETRIEVAL_TOPK, help="每个检索器的 topk")

    parser.add_argument(
        "--enable-reranker",
        action="store_true",
        help="是否启用排序器；仅在启用检索后生效",
    )
    parser.add_argument(
        "--reranker",
        type=str,
        default="bge_m3",
        choices=RERANKER_CHOICES,
        help="排序器名称",
    )
    parser.add_argument("--rerank-topk", type=int, default=DEFAULT_RERANK_TOPK, help="排序后保留的 topk")
    parser.add_argument("--bge-reranker-model-path", type=str, default=bge_reranker_tuned_model_path)
    parser.add_argument("--qwen-reranker-model-path", type=str, default=Qwen3_Reranker_path)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def save_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    return str(text or "").strip()


def parse_name_list(raw: str, allowed: Sequence[str]) -> List[str]:
    names = [item.strip() for item in str(raw or "").split(",") if item.strip()]
    invalid = [name for name in names if name not in allowed]
    if invalid:
        raise ValueError(f"不支持的名称: {invalid}，允许值: {list(allowed)}")
    return names


def load_samples(dataset_path: str, limit: int = 0) -> List[EvalSample]:
    samples: List[EvalSample] = []
    with open(dataset_path, "r", encoding="utf-8") as file:
        for index, line in enumerate(file):
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            question = normalize_text(row.get("input", ""))
            answer = normalize_text(row.get("output", ""))
            instruction = normalize_text(row.get("instruction", ""))
            system_prompt = normalize_text(row.get("system", ""))
            if not question:
                continue
            samples.append(
                EvalSample(
                    sample_id=f"s{index:04d}",
                    instruction=instruction,
                    question=question,
                    answer=answer,
                    system_prompt=system_prompt,
                )
            )
            if limit > 0 and len(samples) >= limit:
                break
    return samples


def build_run_id(args: argparse.Namespace) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = Path(args.dataset_path).stem
    custom = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in args.run_name).strip("_")
    if custom:
        return f"{timestamp}_{custom}_{dataset_name}"
    return f"{timestamp}_rag_eval_inputs_{dataset_name}"


def build_doc_context(idx: int, doc: Document) -> str:
    metadata = doc.metadata or {}
    page_no = metadata.get("orig_page_no") or metadata.get("page_no") or metadata.get("page") or "未知"
    chunk_level = metadata.get("chunk_level") or "unknown"
    source = metadata.get("source") or ""
    figure_refs = metadata.get("figure_refs") or metadata.get("images_info") or []
    footnotes = metadata.get("related_footnotes") or []

    lines = [
        f"【{idx}】",
        f"页码: {page_no}",
        f"分块层级: {chunk_level}",
    ]
    if source:
        lines.append(f"来源: {source}")

    lines.append("正文:")
    lines.append(doc.page_content)

    if figure_refs:
        lines.append("图表信息:")
        for figure in figure_refs:
            if not isinstance(figure, dict):
                continue
            fig_page = figure.get("orig_page_no") or figure.get("page_no") or page_no
            fig_label = figure.get("caption_label") or ""
            fig_text = figure.get("caption_text") or figure.get("title") or ""
            fig_path = figure.get("image_path") or figure.get("path") or figure.get("img_path") or ""
            lines.append(f"- 页码: {fig_page}; 标识: {fig_label}; 描述: {fig_text}; 路径: {fig_path}")

    if footnotes:
        lines.append("脚注信息:")
        for footnote in footnotes:
            lines.append(f"- {footnote}")

    return "\n".join(lines)


def build_context(docs: Sequence[Document]) -> str:
    return "\n\n".join(build_doc_context(idx + 1, doc) for idx, doc in enumerate(docs))


def serialize_doc(doc: Document) -> Dict[str, Any]:
    metadata = doc.metadata or {}
    return {
        "page_content": doc.page_content,
        "metadata": metadata,
        "unique_id": metadata.get("unique_id"),
        "page_no": metadata.get("orig_page_no") or metadata.get("page_no") or metadata.get("page"),
        "source": metadata.get("source"),
    }


def build_retriever(name: str):
    if name == "bm25":
        return CurrentBM25(docs=None)
    if name == "old_bm25":
        return OldBM25(docs=[], retrieve=True)
    if name == "milvus":
        return CurrentMilvusRetriever(docs=None, rebuild=False)
    if name == "old_milvus":
        return OldMilvusRetriever(docs=[], retrieve=True)
    raise ValueError(f"unsupported retriever: {name}")


def build_reranker(name: str, args: argparse.Namespace):
    if name == "bge_m3":
        return BGEM3ReRanker(model_path=args.bge_reranker_model_path)
    if name == "qwen3_4b":
        return Qwen3ReRankervLLM(model_path=args.qwen_reranker_model_path)
    raise ValueError(f"unsupported reranker: {name}")


def collect_retrieved_docs(
    query: str,
    retriever_names: Sequence[str],
    retriever_map: Dict[str, Any],
    topk: int,
) -> tuple[list[Document], dict[str, list[Document]], dict[str, float]]:
    per_retriever_docs: Dict[str, List[Document]] = {}
    per_retriever_times: Dict[str, float] = {}
    merged_docs: List[Document] = []

    for retriever_name in retriever_names:
        start = time.perf_counter()
        docs = retriever_map[retriever_name].retrieve_topk(query, topk=topk)
        elapsed = time.perf_counter() - start

        normalized_docs = [doc for doc in docs if isinstance(doc, Document)]
        per_retriever_docs[retriever_name] = normalized_docs
        per_retriever_times[retriever_name] = elapsed
        merged_docs = merge_docs(merged_docs, normalized_docs)

    return merged_docs, per_retriever_docs, per_retriever_times


def main() -> None:
    args = parse_args()
    if args.limit < 0:
        raise ValueError("--limit 不能小于 0")
    if args.retrieval_topk <= 0:
        raise ValueError("--retrieval-topk 必须大于 0")
    if args.rerank_topk <= 0:
        raise ValueError("--rerank-topk 必须大于 0")

    samples = load_samples(args.dataset_path, limit=args.limit)
    if not samples:
        raise ValueError(f"测试集为空: {args.dataset_path}")

    retriever_names = parse_name_list(args.retrievers, RETRIEVER_CHOICES) if args.enable_retriever else []
    if args.enable_retriever and not retriever_names:
        raise ValueError("启用检索时，--retrievers 不能为空")

    print(">>> 加载检索器...")
    retriever_map = {name: build_retriever(name) for name in retriever_names}

    reranker = None
    if args.enable_reranker:
        if not args.enable_retriever:
            raise ValueError("启用排序器前必须先启用检索器")
        print(">>> 加载排序器...")
        reranker = build_reranker(args.reranker, args)

    run_id = build_run_id(args)
    run_dir = Path(args.output_dir) / run_id
    ensure_dir(run_dir)

    records: List[Dict[str, Any]] = []
    summary = {
        "sample_count": 0,
        "retriever_enabled": args.enable_retriever,
        "retrievers": retriever_names,
        "reranker_enabled": bool(reranker is not None),
        "reranker": args.reranker if reranker is not None else "",
        "avg_retrieval_time_seconds": 0.0,
        "avg_rerank_time_seconds": 0.0,
        "avg_prompt_build_time_seconds": 0.0,
        "avg_total_time_seconds": 0.0,
    }

    total_retrieval_time = 0.0
    total_rerank_time = 0.0
    total_prompt_build_time = 0.0
    total_total_time = 0.0

    print(">>> 开始逐条构建 RAG 评测输入...")
    for index, sample in enumerate(samples, start=1):
        sample_start = time.perf_counter()

        merged_docs: List[Document] = []
        per_retriever_docs: Dict[str, List[Document]] = {}
        per_retriever_times: Dict[str, float] = {}
        retrieval_time = 0.0
        if args.enable_retriever:
            retrieval_stage_start = time.perf_counter()
            merged_docs, per_retriever_docs, per_retriever_times = collect_retrieved_docs(
                query=sample.question,
                retriever_names=retriever_names,
                retriever_map=retriever_map,
                topk=args.retrieval_topk,
            )
            retrieval_time = time.perf_counter() - retrieval_stage_start

        rerank_time = 0.0
        final_docs = merged_docs
        if reranker is not None and merged_docs:
            rerank_stage_start = time.perf_counter()
            final_docs = reranker.rank(sample.question, merged_docs, topk=args.rerank_topk)
            rerank_time = time.perf_counter() - rerank_stage_start

        prompt_build_start = time.perf_counter()
        context = build_context(final_docs)
        final_prompt = LLM_CHAT_PROMPT.format(context=context, query=sample.question)
        prompt_build_time = time.perf_counter() - prompt_build_start

        total_time = time.perf_counter() - sample_start

        record = {
            "sample_id": sample.sample_id,
            "question": sample.question,
            "gold_answer": sample.answer,
            "alpaca_sample": {
                "instruction": "请严格依据给定检索信息回答问题，不要输出思考过程，只输出最终答案。",
                "input": final_prompt,
                "output": sample.answer,
                "system": sample.system_prompt
                or "你是《Introduction to Information Retrieval》教材的问答助手。",
            },
            "retrieval_time_seconds": retrieval_time,
            "rerank_time_seconds": rerank_time,
            "prompt_build_time_seconds": prompt_build_time,
            "total_time_seconds": total_time,
        }
        records.append(record)

        total_retrieval_time += retrieval_time
        total_rerank_time += rerank_time
        total_prompt_build_time += prompt_build_time
        total_total_time += total_time

        print(
            f"[{index}/{len(samples)}] sample_id={sample.sample_id} "
            f"retrieval={retrieval_time:.3f}s rerank={rerank_time:.3f}s prompt={prompt_build_time:.3f}s"
        )

    if hasattr(reranker, "stop"):
        try:
            reranker.stop()
        except Exception:
            pass

    sample_count = len(records)
    summary["sample_count"] = sample_count
    if sample_count > 0:
        summary["avg_retrieval_time_seconds"] = total_retrieval_time / sample_count
        summary["avg_rerank_time_seconds"] = total_rerank_time / sample_count
        summary["avg_prompt_build_time_seconds"] = total_prompt_build_time / sample_count
        summary["avg_total_time_seconds"] = total_total_time / sample_count

    generated_at = datetime.now().isoformat(timespec="seconds")
    config_payload = {
        "run_id": run_id,
        "generated_at": generated_at,
        "dataset_path": args.dataset_path,
        "limit": args.limit,
        "retriever_enabled": args.enable_retriever,
        "retrievers": retriever_names,
        "retrieval_topk": args.retrieval_topk,
        "reranker_enabled": bool(reranker is not None),
        "reranker": args.reranker if reranker is not None else "",
        "rerank_topk": args.rerank_topk if reranker is not None else 0,
        "bge_reranker_model_path": args.bge_reranker_model_path,
        "qwen_reranker_model_path": args.qwen_reranker_model_path,
        "prompt_template": LLM_CHAT_PROMPT,
    }
    result_payload = {
        "run_id": run_id,
        "generated_at": generated_at,
        "summary": summary,
    }

    config_path = run_dir / "config.json"
    result_path = run_dir / "result.json"
    records_path = run_dir / "records.jsonl"

    save_json(config_path, config_payload)
    save_json(result_path, result_payload)
    save_jsonl(records_path, records)

    print("\n===== RAG Eval Input Build Results =====")
    print(f"run_id:                    {run_id}")
    print(f"sample_count:              {summary['sample_count']}")
    print(f"avg_retrieval_time:        {summary['avg_retrieval_time_seconds']:.6f}")
    print(f"avg_rerank_time:           {summary['avg_rerank_time_seconds']:.6f}")
    print(f"avg_prompt_build_time:     {summary['avg_prompt_build_time_seconds']:.6f}")
    print(f"avg_total_time:            {summary['avg_total_time_seconds']:.6f}")
    print(f"\n>>> 配置已保存: {config_path}")
    print(f">>> 汇总已保存: {result_path}")
    print(f">>> 逐条结果已保存: {records_path}")


if __name__ == "__main__":
    main()
