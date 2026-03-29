"""
统一检索器评测脚本。

支持对 src/retriever 下的四个检索器进行统一评测：
- bm25
- old_bm25
- milvus
- old_milvus

评测数据来自：
/root/autodl-tmp/IR-RAG-System/data/qa_pairs/rag_retrieval_reranker/test_grouped.jsonl

评测方式：
- 对 BM25 检索器，使用 grouped test set 中所有 hits 构成本地评测语料库
- 对 Milvus 检索器，直接调用真实后端的 retrieve_topk，并用返回文本与测试集目标文档比对
- 以 label == 2 的文档作为目标文档
- 计算 Recall@1/3/5、MRR、Hit@1、Hit@3
- 每次评测生成独立 run 目录
- run 目录内同时保存 result.json 与 config.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

from langchain.schema import Document

from src.retriever.bm25_retriever import BM25 as CurrentBM25
from src.retriever.old_bm25_retriever import BM25 as OldBM25
import src.retriever.milvus_retriever as current_milvus_module
import src.retriever.old_milvus_retriever as old_milvus_module


BASE_EVAL_DIR = Path("/root/autodl-tmp/IR-RAG-System/src/evaluation")
DEFAULT_DATA_PATH = "/root/autodl-tmp/IR-RAG-System/data/qa_pairs/rag_retrieval_reranker/test_grouped.jsonl"
DEFAULT_RUNS_DIR = BASE_EVAL_DIR / "runs"
DEFAULT_TOPK = 5
TARGET_LABEL = 2
RETRIEVER_CHOICES = ("bm25", "old_bm25", "milvus", "old_milvus")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统一评测当前项目中的检索器效果")
    parser.add_argument(
        "--retriever",
        type=str,
        default="bm25",
        choices=RETRIEVER_CHOICES,
        help="选择待评测检索器，包括 old_bm25、bm25、old_milvus、milvus",
    )
    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH, help="测试集路径")
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK, help="最大检索深度，至少应覆盖 5")
    parser.add_argument(
        "--runs-dir",
        type=str,
        default=str(DEFAULT_RUNS_DIR),
        help="评测运行目录根路径；每次运行会生成 runs/<run_id>/",
    )
    return parser.parse_args()


def stable_doc_id(text: str) -> str:
    return hashlib.md5(text.strip().encode("utf-8")).hexdigest()


def load_grouped_testset(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            query = str(row.get("query", "")).strip()
            hits = row.get("hits", [])
            if not query or not hits:
                continue
            rows.append(row)
    return rows


def build_corpus_and_labels(
    rows: Sequence[Dict[str, Any]],
) -> tuple[list[Document], dict[str, str], dict[str, dict[str, int]]]:
    docs: List[Document] = []
    doc_store: Dict[str, str] = {}
    qrels: Dict[str, Dict[str, int]] = {}

    for idx, row in enumerate(rows):
        query_id = str(row.get("query_id") or f"q{idx}")
        qrels[query_id] = {}

        for hit_idx, hit in enumerate(row.get("hits", [])):
            content = str(hit.get("content", "")).strip()
            label = int(hit.get("label", 0))
            if not content:
                continue

            doc_id = stable_doc_id(content)
            qrels[query_id][doc_id] = label

            if doc_id not in doc_store:
                doc_store[doc_id] = content
                docs.append(
                    Document(
                        page_content=content,
                        metadata={
                            "unique_id": doc_id,
                            "parent_id": doc_id,
                            "source": f"eval_source::{doc_id}",
                            "chunk_type": "eval",
                            "content_hash": doc_id,
                            "hit_index": hit_idx,
                        },
                    )
                )

    return docs, doc_store, qrels


def content_to_doc_id(text: str) -> str:
    return stable_doc_id(text) if text.strip() else ""


def recall_at_k(relevant_doc_ids: Sequence[str], retrieved_doc_ids: Sequence[str], k: int) -> float:
    if not relevant_doc_ids:
        return 0.0
    topk_ids = set(retrieved_doc_ids[:k])
    hit_count = sum(1 for doc_id in relevant_doc_ids if doc_id in topk_ids)
    return hit_count / len(relevant_doc_ids)


def hit_at_k(relevant_doc_ids: Sequence[str], retrieved_doc_ids: Sequence[str], k: int) -> float:
    relevant = set(relevant_doc_ids)
    return 1.0 if any(doc_id in relevant for doc_id in retrieved_doc_ids[:k]) else 0.0


def reciprocal_rank(relevant_doc_ids: Sequence[str], retrieved_doc_ids: Sequence[str]) -> float:
    relevant = set(relevant_doc_ids)
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def build_run_id(retriever_name: str, data_path: str, topk: int) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_name = Path(data_path).stem
    return f"{timestamp}_{retriever_name}_{data_name}_top{topk}"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def build_retriever(retriever_name: str, docs: Sequence[Document]):
    if retriever_name == "bm25":
        return CurrentBM25(docs=None)
    if retriever_name == "old_bm25":
        return OldBM25(docs=[], retrieve=True)
    if retriever_name == "milvus":
        return current_milvus_module.MilvusRetriever(docs=None, rebuild=False)
    if retriever_name == "old_milvus":
        return old_milvus_module.MilvusRetriever(docs=[], retrieve=True)
    raise ValueError(f"unsupported retriever: {retriever_name}")


def evaluate_retriever_on_grouped_testset(
    retriever_name: str,
    data_path: str = DEFAULT_DATA_PATH,
    topk: int = DEFAULT_TOPK,
    runs_dir: str | Path = DEFAULT_RUNS_DIR,
) -> Dict[str, Any]:
    rows = load_grouped_testset(data_path)
    if not rows:
        raise ValueError(f"测试集为空: {data_path}")

    docs, _, qrels = build_corpus_and_labels(rows)
    retriever = build_retriever(retriever_name, docs)

    metrics_tracker = {
        "recall@1": [],
        "recall@3": [],
        "recall@5": [],
        "mrr": [],
        "hit@1": [],
        "hit@3": [],
    }
    for idx, row in enumerate(rows):
        query_id = str(row.get("query_id") or f"q{idx}")
        query = str(row.get("query", "")).strip()
        label_map = qrels[query_id]

        target_doc_ids = [doc_id for doc_id, label in label_map.items() if label == TARGET_LABEL]
        retrieved_docs = retriever.retrieve_topk(query, topk=topk)

        retrieved_doc_ids = []
        for doc in retrieved_docs:
            content = str(getattr(doc, "page_content", "") or "").strip()
            doc_id = content_to_doc_id(content)
            if not doc_id:
                continue
            retrieved_doc_ids.append(doc_id)

        metrics_tracker["recall@1"].append(recall_at_k(target_doc_ids, retrieved_doc_ids, 1))
        metrics_tracker["recall@3"].append(recall_at_k(target_doc_ids, retrieved_doc_ids, 3))
        metrics_tracker["recall@5"].append(recall_at_k(target_doc_ids, retrieved_doc_ids, 5))
        metrics_tracker["mrr"].append(reciprocal_rank(target_doc_ids, retrieved_doc_ids))
        metrics_tracker["hit@1"].append(hit_at_k(target_doc_ids, retrieved_doc_ids, 1))
        metrics_tracker["hit@3"].append(hit_at_k(target_doc_ids, retrieved_doc_ids, 3))

    aggregated_metrics = {
        metric_name: (sum(values) / len(values) if values else 0.0)
        for metric_name, values in metrics_tracker.items()
    }

    run_id = build_run_id(retriever_name, data_path, topk)
    run_dir = Path(runs_dir) / run_id
    result_path = run_dir / "result.json"
    config_path = run_dir / "config.json"
    generated_at = datetime.now().isoformat(timespec="seconds")

    config_payload = {
        "run_id": run_id,
        "generated_at": generated_at,
        "retriever": retriever_name,
        "data_path": data_path,
        "topk": topk,
        "retriever_mode": "real_backend",
        "target_label": TARGET_LABEL,
        "query_count": len(rows),
        "corpus_doc_count": len(docs),
        "metrics": list(metrics_tracker.keys()),
    }

    result_payload = {
        "run_id": run_id,
        "summary": {
            "retriever": retriever_name,
            "data_path": data_path,
            "topk": topk,
            "retriever_mode": "real_backend",
            "target_label": TARGET_LABEL,
            "query_count": len(rows),
            "corpus_doc_count": len(docs),
            "generated_at": generated_at,
        },
        "metrics": aggregated_metrics,
    }

    save_json(config_path, config_payload)
    save_json(result_path, result_payload)

    return {
        "run_id": run_id,
        "metrics": aggregated_metrics,
        "run_dir": str(run_dir),
        "result_path": str(result_path),
        "config_path": str(config_path),
    }


def main() -> None:
    args = parse_args()
    if args.topk < 5:
        raise ValueError("topk 至少应为 5，否则无法正确计算 Recall@5")

    summary = evaluate_retriever_on_grouped_testset(
        retriever_name=args.retriever,
        data_path=args.data_path,
        topk=args.topk,
        runs_dir=args.runs_dir,
    )

    print("\n===== Retrieval Evaluation Results =====")
    print(f"retriever: {args.retriever}")
    for metric_name, metric_value in summary["metrics"].items():
        print(f"{metric_name}: {metric_value:.6f}")
    print(f"\n>>> 运行目录已保存到: {summary['run_dir']}")
    print(f">>> 结果 JSON 已保存到: {summary['result_path']}")
    print(f">>> 配置 JSON 已保存到: {summary['config_path']}")


if __name__ == "__main__":
    main()
