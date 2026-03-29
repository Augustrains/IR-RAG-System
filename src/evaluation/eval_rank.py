"""
排序器评估脚本。

默认用于评估基于 Cross-Encoder / BGE reranker 的排序效果，支持：
1. 读取 grouped 或 pointwise 格式的数据集
2. 加载本地微调后的 SequenceClassification 模型
3. 计算常用排序指标：
   - hits@k
   - recall@k
   - mrr
   - map
   - ndcg@k
4. 自动保存：
   - 每次评测产出一个独立 run 目录
   - 目录内同时保存 result.json 与 config.json
5. 可选保存逐 query 排序明细到结果 JSON 中

示例：
python eval_rank.py
python -m src.evaluation.eval_rank --save-details
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


BASE_EVAL_DIR = Path("/root/autodl-tmp/IR-RAG-System/src/evaluation")
DEFAULT_MODEL_PATH = "/root/autodl-tmp/IR-RAG-System/output/rag_retrieval_bge_reranker_v2_m3/model"
DEFAULT_DATA_PATH = "/root/autodl-tmp/IR-RAG-System/data/qa_pairs/rag_retrieval_reranker/test_grouped.jsonl"
DEFAULT_RUNS_DIR = BASE_EVAL_DIR / "runs"
DEFAULT_MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 64
DEFAULT_BINARY_THRESHOLD = 1.0
DEFAULT_METRICS = (
    "hits@1",
    "hits@3",
    "hits@5",
    "recall@5",
    "recall@10",
    "mrr",
    "map",
    "ndcg@5",
    "ndcg@10",
)


@dataclass
class CandidateItem:
    content: str
    label: float
    score: float | None = None


@dataclass
class QuerySample:
    query_id: str
    query: str
    candidates: List[CandidateItem]


class LocalReranker:
    """面向本地 SequenceClassification 模型的轻量级 reranker 封装。"""

    def __init__(
        self,
        model_path: str,
        device: str | None = None,
        max_length: int = DEFAULT_MAX_LENGTH,
        use_fp16: bool = True,
    ) -> None:
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.model.to(self.device)

        if use_fp16 and self.device.startswith("cuda"):
            self.model.half()

    @torch.no_grad()
    def compute_scores(
        self,
        query_doc_pairs: Sequence[Tuple[str, str]],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> List[float]:
        scores: List[float] = []

        for start in range(0, len(query_doc_pairs), batch_size):
            batch_pairs = query_doc_pairs[start : start + batch_size]
            encoded = self.tokenizer(
                batch_pairs,
                padding=True,
                truncation="only_second",
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            logits = self.model(**encoded).logits
            logits = logits.squeeze(-1).detach().float().cpu().tolist()
            if isinstance(logits, float):
                logits = [logits]
            scores.extend(logits)

        return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估本地 reranker 的排序效果")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH, help="已训练 reranker 模型目录")
    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH, help="测试集 JSONL 路径")
    parser.add_argument(
        "--data-format",
        type=str,
        default="grouped",
        choices=["grouped", "pointwise"],
        help="测试集格式",
    )
    parser.add_argument("--device", type=str, default=None, help="推理设备，例如 cuda:0 / cpu")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="打分 batch size")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH, help="tokenizer 最大长度")
    parser.add_argument(
        "--binary-threshold",
        type=float,
        default=DEFAULT_BINARY_THRESHOLD,
        help="将 label 视为相关文档的阈值，默认 label >= 1 视为 relevant",
    )
    parser.add_argument(
        "--save-details",
        action="store_true",
        help="是否在结果 JSON 中保存逐 query 排序明细",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default=str(DEFAULT_RUNS_DIR),
        help="评测运行目录根路径；每次运行会生成 runs/<run_id>/",
    )
    return parser.parse_args()


def load_grouped_samples(data_path: str) -> List[QuerySample]:
    samples: List[QuerySample] = []
    with open(data_path, "r", encoding="utf-8") as file:
        for idx, line in enumerate(file):
            row = json.loads(line)
            query = row["query"].strip()
            hits = row.get("hits", [])
            candidates = [
                CandidateItem(
                    content=hit["content"].strip(),
                    label=float(hit.get("label", 0.0)),
                )
                for hit in hits
                if hit.get("content", "").strip()
            ]
            if not query or not candidates:
                continue
            query_id = row.get("query_id") or row.get("query_ids", [None])[0] or f"q{idx}"
            samples.append(QuerySample(query_id=str(query_id), query=query, candidates=candidates))
    return samples


def load_pointwise_samples(data_path: str) -> List[QuerySample]:
    grouped: Dict[str, QuerySample] = {}
    with open(data_path, "r", encoding="utf-8") as file:
        for idx, line in enumerate(file):
            row = json.loads(line)
            query = row["query"].strip()
            content = row["content"].strip()
            label = float(row.get("label", 0.0))
            if not query or not content:
                continue
            query_id = row.get("query_id") or f"q{idx}"
            if query_id not in grouped:
                grouped[query_id] = QuerySample(query_id=str(query_id), query=query, candidates=[])
            grouped[query_id].candidates.append(CandidateItem(content=content, label=label))
    return list(grouped.values())


def load_samples(data_path: str, data_format: str) -> List[QuerySample]:
    if data_format == "grouped":
        return load_grouped_samples(data_path)
    if data_format == "pointwise":
        return load_pointwise_samples(data_path)
    raise ValueError(f"Unsupported data format: {data_format}")


def dcg_at_k(labels: Sequence[float], k: int) -> float:
    score = 0.0
    for rank, label in enumerate(labels[:k], start=1):
        gain = (2.0 ** float(label)) - 1.0
        discount = math.log2(rank + 1.0)
        score += gain / discount
    return score


def ndcg_at_k(labels: Sequence[float], k: int) -> float:
    actual = dcg_at_k(labels, k)
    ideal = dcg_at_k(sorted(labels, reverse=True), k)
    if ideal <= 0:
        return 0.0
    return actual / ideal


def hits_at_k(binary_labels: Sequence[int], k: int) -> float:
    return 1.0 if any(binary_labels[:k]) else 0.0


def recall_at_k(binary_labels: Sequence[int], k: int) -> float:
    total_relevant = sum(binary_labels)
    if total_relevant == 0:
        return 0.0
    return sum(binary_labels[:k]) / total_relevant


def reciprocal_rank(binary_labels: Sequence[int]) -> float:
    for index, label in enumerate(binary_labels, start=1):
        if label:
            return 1.0 / index
    return 0.0


def average_precision(binary_labels: Sequence[int]) -> float:
    total_relevant = sum(binary_labels)
    if total_relevant == 0:
        return 0.0

    hit_count = 0
    precision_sum = 0.0
    for index, label in enumerate(binary_labels, start=1):
        if label:
            hit_count += 1
            precision_sum += hit_count / index
    return precision_sum / total_relevant


def build_query_pairs(samples: Sequence[QuerySample]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for sample in samples:
        for candidate in sample.candidates:
            pairs.append((sample.query, candidate.content))
    return pairs


def attach_scores(samples: Sequence[QuerySample], scores: Sequence[float]) -> None:
    offset = 0
    for sample in samples:
        for candidate in sample.candidates:
            candidate.score = float(scores[offset])
            offset += 1
    if offset != len(scores):
        raise ValueError("score 数量与候选文档数量不一致")


def sort_candidates(sample: QuerySample) -> List[CandidateItem]:
    return sorted(
        sample.candidates,
        key=lambda item: (item.score if item.score is not None else float("-inf")),
        reverse=True,
    )


def evaluate_samples(
    samples: Sequence[QuerySample],
    binary_threshold: float,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    metric_values: Dict[str, List[float]] = {metric_name: [] for metric_name in DEFAULT_METRICS}
    details: List[Dict[str, Any]] = []

    for sample in samples:
        ranked = sort_candidates(sample)
        labels = [item.label for item in ranked]
        binary_labels = [1 if item.label >= binary_threshold else 0 for item in ranked]

        metric_values["hits@1"].append(hits_at_k(binary_labels, 1))
        metric_values["hits@3"].append(hits_at_k(binary_labels, 3))
        metric_values["hits@5"].append(hits_at_k(binary_labels, 5))
        metric_values["recall@5"].append(recall_at_k(binary_labels, 5))
        metric_values["recall@10"].append(recall_at_k(binary_labels, 10))
        metric_values["mrr"].append(reciprocal_rank(binary_labels))
        metric_values["map"].append(average_precision(binary_labels))
        metric_values["ndcg@5"].append(ndcg_at_k(labels, 5))
        metric_values["ndcg@10"].append(ndcg_at_k(labels, 10))

        details.append(
            {
                "query_id": sample.query_id,
                "query": sample.query,
                "num_candidates": len(ranked),
                "num_relevant": int(sum(binary_labels)),
                "ranked_candidates": [
                    {
                        "rank": rank,
                        "score": float(item.score if item.score is not None else 0.0),
                        "label": float(item.label),
                        "content": item.content,
                    }
                    for rank, item in enumerate(ranked, start=1)
                ],
            }
        )

    averaged_metrics = {
        metric_name: (sum(values) / len(values) if values else 0.0)
        for metric_name, values in metric_values.items()
    }
    return averaged_metrics, details


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_run_id(model_path: str, data_path: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = Path(model_path).name
    data_name = Path(data_path).stem
    return f"{timestamp}_{model_name}_{data_name}"


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def validate_inputs(args: argparse.Namespace) -> None:
    model_path = Path(args.model_path)
    data_path = Path(args.data_path)
    if not model_path.exists():
        raise FileNotFoundError(f"模型目录不存在: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"测试集不存在: {data_path}")


def main() -> None:
    args = parse_args()
    validate_inputs(args)

    print(">>> 开始加载测试集...")
    samples = load_samples(args.data_path, args.data_format)
    if not samples:
        raise ValueError("评测数据为空，请检查 data-path 与 data-format 是否匹配")

    num_queries = len(samples)
    num_candidates = sum(len(sample.candidates) for sample in samples)
    print(f">>> 查询数: {num_queries}")
    print(f">>> 候选文档总数: {num_candidates}")

    print(">>> 开始加载 reranker 模型...")
    reranker = LocalReranker(
        model_path=args.model_path,
        device=args.device,
        max_length=args.max_length,
    )
    print(f">>> 推理设备: {reranker.device}")

    print(">>> 开始批量打分...")
    query_pairs = build_query_pairs(samples)
    scores = reranker.compute_scores(query_pairs, batch_size=args.batch_size)
    attach_scores(samples, scores)

    print(">>> 开始计算排序指标...")
    metrics, details = evaluate_samples(samples, binary_threshold=args.binary_threshold)

    print("\n===== Reranker Evaluation Results =====")
    print(f"model_path: {args.model_path}")
    print(f"data_path:  {args.data_path}")
    print(f"data_format:{args.data_format}")
    print(f"threshold:  {args.binary_threshold}")
    for metric_name in DEFAULT_METRICS:
        print(f"{metric_name}: {metrics[metric_name]:.6f}")

    run_id = build_run_id(args.model_path, args.data_path)
    run_dir = Path(args.runs_dir) / run_id
    result_path = run_dir / "result.json"
    config_path = run_dir / "config.json"

    generated_at = datetime.now().isoformat(timespec="seconds")
    config_payload = {
        "run_id": run_id,
        "model_path": args.model_path,
        "data_path": args.data_path,
        "data_format": args.data_format,
        "device": reranker.device,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "binary_threshold": args.binary_threshold,
        "save_details": args.save_details,
        "num_queries": num_queries,
        "num_candidates": num_candidates,
        "metrics": list(DEFAULT_METRICS),
        "generated_at": generated_at,
    }

    result_payload: Dict[str, Any] = {
        "run_id": run_id,
        "summary": {
            "model_path": args.model_path,
            "data_path": args.data_path,
            "data_format": args.data_format,
            "num_queries": num_queries,
            "num_candidates": num_candidates,
            "binary_threshold": args.binary_threshold,
            "generated_at": generated_at,
        },
        "metrics": metrics,
    }
    if args.save_details:
        result_payload["details"] = details

    save_json(config_path, config_payload)
    save_json(result_path, result_payload)

    print(f"\n>>> 结果 JSON 已保存到: {result_path}")
    print(f">>> 配置 JSON 已保存到: {config_path}")


if __name__ == "__main__":
    main()
