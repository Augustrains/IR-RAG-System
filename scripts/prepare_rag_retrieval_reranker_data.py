#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def map_relevance_label(label: int, label_scheme: str) -> int:
    if label_scheme == "raw":
        return int(label)
    if label_scheme == "graded_012":
        mapping = {-1: 0, 0: 1, 1: 2}
        if int(label) not in mapping:
            raise ValueError(f"unsupported relevance label for graded_012: {label}")
        return mapping[int(label)]
    raise ValueError(f"unsupported label_scheme: {label_scheme}")


def normalize_query_ids(row: Dict) -> Tuple[str, ...]:
    query_ids = row.get("query_ids")
    if isinstance(query_ids, list):
        normalized = [str(query_id).strip() for query_id in query_ids if str(query_id).strip()]
        if normalized:
            return tuple(normalized)

    query_id = str(row.get("query_id", "")).strip()
    if query_id:
        return (query_id,)

    return ()


def iter_blocks(path: Path) -> Iterable[List[Dict]]:
    current_block: List[Dict] = []
    current_key = None

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except Exception as exc:
                raise ValueError(f"failed to parse {path} line {line_no}: {exc}") from exc

            block_key = (str(row.get("query", "")), normalize_query_ids(row))
            if current_block and block_key != current_key:
                yield current_block
                current_block = []

            current_block.append(row)
            current_key = block_key

    if current_block:
        yield current_block


def convert_grouped(block: List[Dict], label_scheme: str) -> Dict:
    first = block[0]
    query_ids = list(normalize_query_ids(first))
    hits = []
    for row in block:
        hits.append({
            "content": row["document"],
            "label": map_relevance_label(int(row["relevance"]), label_scheme),
        })

    payload = {
        "query": first["query"],
        "hits": hits,
    }
    if query_ids:
        payload["query_id"] = query_ids[0]
        payload["query_ids"] = query_ids
    return payload


def convert_pointwise(block: List[Dict], label_scheme: str) -> List[Dict]:
    first = block[0]
    query_ids = list(normalize_query_ids(first))
    rows = []
    for row in block:
        payload = {
            "query": row["query"],
            "content": row["document"],
            "label": map_relevance_label(int(row["relevance"]), label_scheme),
        }
        if query_ids:
            payload["query_id"] = query_ids[0]
            payload["query_ids"] = query_ids
        rows.append(payload)
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def process_split(input_path: Path, grouped_path: Path, pointwise_path: Path | None, label_scheme: str) -> Dict:
    grouped_rows = []
    pointwise_rows = []
    block_count = 0
    pair_count = 0

    for block in iter_blocks(input_path):
        block_count += 1
        pair_count += len(block)
        grouped_rows.append(convert_grouped(block, label_scheme))
        if pointwise_path is not None:
            pointwise_rows.extend(convert_pointwise(block, label_scheme))

    grouped_count = write_jsonl(grouped_path, grouped_rows)
    pointwise_count = None
    if pointwise_path is not None:
        pointwise_count = write_jsonl(pointwise_path, pointwise_rows)

    return {
        "input": str(input_path),
        "grouped_output": str(grouped_path),
        "grouped_rows": grouped_count,
        "pointwise_output": str(pointwise_path) if pointwise_path is not None else None,
        "pointwise_rows": pointwise_count,
        "query_blocks": block_count,
        "query_doc_pairs": pair_count,
        "label_scheme": label_scheme,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert IR-RAG reranker labels into RAG-Retrieval training data.")
    parser.add_argument("--train-input", default="/root/autodl-tmp/IR-RAG-System/data/qa_pairs/reranker/train_rank_labels.jsonl")
    parser.add_argument("--val-input", default="/root/autodl-tmp/IR-RAG-System/data/qa_pairs/reranker/val_rank_labels.jsonl")
    parser.add_argument("--test-input", default="/root/autodl-tmp/IR-RAG-System/data/qa_pairs/reranker/test_rank_labels.jsonl")
    parser.add_argument("--output-dir", default="/root/autodl-tmp/IR-RAG-System/data/qa_pairs/rag_retrieval_reranker")
    parser.add_argument("--write-pointwise", action="store_true", help="Also export pointwise JSONL alongside grouped data.")
    parser.add_argument(
        "--label-scheme",
        choices=["graded_012", "raw"],
        default="graded_012",
        help="Map raw reranker labels before export. graded_012 maps -1/0/1 to 0/1/2.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_inputs = {
        "train": Path(args.train_input),
        "val": Path(args.val_input),
        "test": Path(args.test_input),
    }

    summary = {}
    for split, input_path in split_inputs.items():
        grouped_path = output_dir / f"{split}_grouped.jsonl"
        pointwise_path = output_dir / f"{split}_pointwise.jsonl" if args.write_pointwise else None
        summary[split] = process_split(input_path, grouped_path, pointwise_path, args.label_scheme)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
