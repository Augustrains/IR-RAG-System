"""
面向 Qwen 问答模型的单模型评测脚本。

特点：
1. 每次只评测一个模型，适合分别评测微调前 / 微调后的效果
2. 直接基于 Transformers / PEFT 加载本地模型，不依赖 vLLM
3. 兼容 LlamaFactory 训练产物：基座模型 + LoRA adapter
4. 自动保存 config/result/predictions/report 到 runs 目录

示例：
python -m src.evaluation.eval_qwen_qa \
  --mode base \
  --base-model-path /root/autodl-tmp/IR-RAG-System/models/Qwen3-8B

python -m src.evaluation.eval_qwen_qa \
  --mode lora \
  --base-model-path /root/autodl-tmp/IR-RAG-System/models/Qwen3-8B \
  --adapter-path /root/autodl-tmp/IR-RAG-System/output/qwen3_8b_lora_ir_rag
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_EVAL_DIR = Path("/root/autodl-tmp/IR-RAG-System/src/evaluation")
DEFAULT_RUNS_DIR = BASE_EVAL_DIR / "runs"
DEFAULT_DATASET_PATH = (
    "/root/autodl-tmp/IR-RAG-System/LlamaFactory-main/data/ir_rag/test_augmented_with_neg_alpaca.jsonl"
)
DEFAULT_BASE_MODEL_PATH = "/root/autodl-tmp/IR-RAG-System/models/Qwen3-8B"
DEFAULT_ADAPTER_PATH = "/root/autodl-tmp/IR-RAG-System/output/qwen3_8b_lora_ir_rag"
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_MAX_SAMPLES_IN_REPORT = 20
NO_ANSWER_TEXT = "无答案"


@dataclass
class QASample:
    sample_id: str
    instruction: str
    question: str
    expected_answer: str
    system_prompt: str


@dataclass
class PredictionRecord:
    sample_id: str
    question: str
    instruction: str
    expected_answer: str
    predicted_answer: str
    normalized_expected_answer: str
    normalized_predicted_answer: str
    exact_match: float
    token_f1: float
    rouge_l_f1: float
    answerable_expected: bool
    answerable_predicted: bool
    no_answer_match: float
    status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评测单个 Qwen 问答模型在测试集上的回答效果")
    parser.add_argument("--mode", type=str, default="base", choices=["base", "lora"], help="评测基座或 LoRA")
    parser.add_argument("--base-model-path", type=str, default=DEFAULT_BASE_MODEL_PATH, help="基座模型路径")
    parser.add_argument("--adapter-path", type=str, default=DEFAULT_ADAPTER_PATH, help="LoRA adapter 路径")
    parser.add_argument("--dataset-path", type=str, default=DEFAULT_DATASET_PATH, help="Alpaca 风格测试集路径")
    parser.add_argument("--runs-dir", type=str, default=str(DEFAULT_RUNS_DIR), help="评测运行目录根路径")
    parser.add_argument("--run-name", type=str, default="", help="可选，自定义运行名")
    parser.add_argument("--limit", type=int, default=0, help="仅评测前 N 条，0 表示全量")
    parser.add_argument("--device", type=str, default="auto", help="cuda / cuda:0 / cpu / auto")
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="模型加载精度",
    )
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help="单条最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.0, help="生成温度，0 表示贪心解码")
    parser.add_argument("--top-p", type=float, default=1.0, help="采样 top_p")
    parser.add_argument("--report-max-samples", type=int, default=DEFAULT_MAX_SAMPLES_IN_REPORT, help="报告展示样例数")
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
    text = str(text or "").strip()
    text = text.replace("\u3000", " ")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def tokenize_mixed(text: str) -> List[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    return re.findall(r"[\u4e00-\u9fff]|[a-z0-9]+", normalized)


def is_no_answer_text(text: str) -> bool:
    normalized = normalize_text(text)
    compact = normalized.replace(" ", "")
    return compact in {
        normalize_text(NO_ANSWER_TEXT),
        "无答案。",
        "无答案[]",
        "无答案【】",
        "不知道",
        "不清楚",
        "无法回答",
        "无法根据当前知识回答",
    } or compact.startswith("无答案")


def exact_match_score(expected: str, predicted: str) -> float:
    return 1.0 if normalize_text(expected) == normalize_text(predicted) else 0.0


def token_f1_score(expected: str, predicted: str) -> float:
    expected_tokens = tokenize_mixed(expected)
    predicted_tokens = tokenize_mixed(predicted)
    if not expected_tokens and not predicted_tokens:
        return 1.0
    if not expected_tokens or not predicted_tokens:
        return 0.0

    overlap: Dict[str, int] = {}
    for token in expected_tokens:
        overlap[token] = overlap.get(token, 0) + 1

    common = 0
    for token in predicted_tokens:
        count = overlap.get(token, 0)
        if count > 0:
            common += 1
            overlap[token] = count - 1

    if common == 0:
        return 0.0

    precision = common / len(predicted_tokens)
    recall = common / len(expected_tokens)
    return (2 * precision * recall) / (precision + recall)


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, start=1):
            cached = dp[j]
            if x == y:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = cached
    return dp[-1]


def rouge_l_f1_score(expected: str, predicted: str) -> float:
    expected_tokens = tokenize_mixed(expected)
    predicted_tokens = tokenize_mixed(predicted)
    if not expected_tokens and not predicted_tokens:
        return 1.0
    if not expected_tokens or not predicted_tokens:
        return 0.0

    lcs = lcs_length(expected_tokens, predicted_tokens)
    if lcs == 0:
        return 0.0

    precision = lcs / len(predicted_tokens)
    recall = lcs / len(expected_tokens)
    return (2 * precision * recall) / (precision + recall)


def classify_status(expected: str, predicted: str, exact_match: float, no_answer_match: float) -> str:
    expected_is_no_answer = is_no_answer_text(expected)
    predicted_is_no_answer = is_no_answer_text(predicted)

    if exact_match >= 1.0:
        return "exact_match"
    if expected_is_no_answer and predicted_is_no_answer and no_answer_match >= 1.0:
        return "correct_no_answer"
    if expected_is_no_answer and not predicted_is_no_answer:
        return "missed_no_answer"
    if not expected_is_no_answer and predicted_is_no_answer:
        return "false_no_answer"
    return "content_mismatch"


def build_prompt(sample: QASample) -> str:
    instruction = sample.instruction.strip()
    question = sample.question.strip()
    if instruction and question:
        return f"{instruction}\n\n{question}"
    return instruction or question


def load_qa_samples(dataset_path: str, limit: int = 0) -> List[QASample]:
    samples: List[QASample] = []
    with open(dataset_path, "r", encoding="utf-8") as file:
        for index, line in enumerate(file):
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            question = str(row.get("input", "")).strip()
            expected = str(row.get("output", "")).strip()
            instruction = str(row.get("instruction", "")).strip()
            system_prompt = str(row.get("system", "")).strip()
            if not question or not expected:
                continue
            samples.append(
                QASample(
                    sample_id=f"s{index:04d}",
                    instruction=instruction,
                    question=question,
                    expected_answer=expected,
                    system_prompt=system_prompt,
                )
            )
            if limit > 0 and len(samples) >= limit:
                break
    return samples


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def resolve_dtype(dtype: str, device: str) -> torch.dtype:
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    if device.startswith("cuda"):
        return torch.bfloat16
    return torch.float32


class LocalQwenGenerator:
    def __init__(
        self,
        mode: str,
        base_model_path: str,
        adapter_path: str,
        device: str,
        dtype: torch.dtype,
    ) -> None:
        self.mode = mode
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path
        self.device = device
        self.dtype = dtype

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        if mode == "lora":
            model = PeftModel.from_pretrained(model, adapter_path)

        self.model = model.to(device)
        self.model.eval()

    @torch.no_grad()
    def generate(self, sample: QASample, max_new_tokens: int, temperature: float, top_p: float) -> str:
        messages = []
        if sample.system_prompt:
            messages.append({"role": "system", "content": sample.system_prompt})
        messages.append({"role": "user", "content": build_prompt(sample)})

        if hasattr(self.tokenizer, "apply_chat_template"):
            model_inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        else:
            prompt = ""
            if sample.system_prompt:
                prompt += f"系统：{sample.system_prompt}\n"
            prompt += f"用户：{build_prompt(sample)}\n助手："
            model_inputs = self.tokenizer(prompt, return_tensors="pt").input_ids

        model_inputs = model_inputs.to(self.device)
        attention_mask = torch.ones_like(model_inputs, device=self.device)
        do_sample = temperature > 0

        generation_kwargs = {
            "input_ids": model_inputs,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p

        generated = self.model.generate(**generation_kwargs)
        generated_tokens = generated[0][model_inputs.shape[-1] :]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def evaluate_predictions(records: Sequence[PredictionRecord]) -> Dict[str, Any]:
    total = len(records)
    if total == 0:
        raise ValueError("没有可评测的预测结果")

    answerable_rows = [row for row in records if row.answerable_expected]
    unanswerable_rows = [row for row in records if not row.answerable_expected]
    false_no_answer = sum(1 for row in records if row.answerable_expected and not row.answerable_predicted)
    missed_no_answer = sum(1 for row in records if (not row.answerable_expected) and row.answerable_predicted)

    status_distribution: Dict[str, int] = {}
    for row in records:
        status_distribution[row.status] = status_distribution.get(row.status, 0) + 1

    token_f1_values = [row.token_f1 for row in records]
    return {
        "sample_count": total,
        "answerable_count": len(answerable_rows),
        "unanswerable_count": len(unanswerable_rows),
        "exact_match": sum(row.exact_match for row in records) / total,
        "token_f1": sum(row.token_f1 for row in records) / total,
        "rouge_l_f1": sum(row.rouge_l_f1 for row in records) / total,
        "no_answer_accuracy": sum(row.no_answer_match for row in records) / total,
        "avg_token_f1_answerable": (
            sum(row.token_f1 for row in answerable_rows) / len(answerable_rows) if answerable_rows else 0.0
        ),
        "avg_token_f1_unanswerable": (
            sum(row.token_f1 for row in unanswerable_rows) / len(unanswerable_rows) if unanswerable_rows else 0.0
        ),
        "false_no_answer_count": false_no_answer,
        "missed_no_answer_count": missed_no_answer,
        "exact_match_count": int(sum(row.exact_match for row in records)),
        "token_f1_std": statistics.pstdev(token_f1_values) if total > 1 else 0.0,
        "status_distribution": status_distribution,
    }


def build_run_id(mode: str, base_model_path: str, dataset_path: str, run_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = Path(base_model_path).name
    dataset_name = Path(dataset_path).stem
    custom = re.sub(r"[^a-zA-Z0-9_-]+", "_", run_name).strip("_")
    if custom:
        return f"{timestamp}_{custom}_{mode}_{model_name}_{dataset_name}"
    return f"{timestamp}_{mode}_{model_name}_{dataset_name}"


def build_report_markdown(
    run_id: str,
    config_payload: Dict[str, Any],
    result_payload: Dict[str, Any],
    records: Sequence[PredictionRecord],
    max_samples: int,
) -> str:
    metrics = result_payload["metrics"]
    worst_cases = sorted(records, key=lambda row: (row.exact_match, row.token_f1, row.rouge_l_f1))[:max_samples]
    best_cases = sorted(records, key=lambda row: (row.exact_match, row.token_f1, row.rouge_l_f1), reverse=True)[
        : min(5, max_samples)
    ]

    lines = [
        "# Qwen QA Evaluation Report",
        "",
        f"- run_id: `{run_id}`",
        f"- generated_at: `{result_payload['generated_at']}`",
        f"- mode: `{config_payload['mode']}`",
        f"- base_model_path: `{config_payload['base_model_path']}`",
        f"- adapter_path: `{config_payload['adapter_path']}`",
        f"- dataset_path: `{config_payload['dataset_path']}`",
        f"- sample_count: `{metrics['sample_count']}`",
        "",
        "## Summary Metrics",
        "",
        f"- exact_match: `{metrics['exact_match']:.4f}`",
        f"- token_f1: `{metrics['token_f1']:.4f}`",
        f"- rouge_l_f1: `{metrics['rouge_l_f1']:.4f}`",
        f"- no_answer_accuracy: `{metrics['no_answer_accuracy']:.4f}`",
        f"- avg_token_f1_answerable: `{metrics['avg_token_f1_answerable']:.4f}`",
        f"- avg_token_f1_unanswerable: `{metrics['avg_token_f1_unanswerable']:.4f}`",
        f"- false_no_answer_count: `{metrics['false_no_answer_count']}`",
        f"- missed_no_answer_count: `{metrics['missed_no_answer_count']}`",
        "",
        "## Best Cases",
        "",
    ]

    for row in best_cases:
        lines.extend(
            [
                f"### {row.sample_id}",
                f"- status: `{row.status}`",
                f"- exact_match: `{row.exact_match:.1f}`",
                f"- token_f1: `{row.token_f1:.4f}`",
                f"- question: {row.question}",
                f"- gold: {row.expected_answer}",
                f"- pred: {row.predicted_answer}",
                "",
            ]
        )

    lines.extend(["## Worst Cases", ""])
    for row in worst_cases:
        lines.extend(
            [
                f"### {row.sample_id}",
                f"- status: `{row.status}`",
                f"- exact_match: `{row.exact_match:.1f}`",
                f"- token_f1: `{row.token_f1:.4f}`",
                f"- rouge_l_f1: `{row.rouge_l_f1:.4f}`",
                f"- question: {row.question}",
                f"- gold: {row.expected_answer}",
                f"- pred: {row.predicted_answer}",
                "",
            ]
        )

    return "\n".join(lines).strip() + "\n"


def validate_args(args: argparse.Namespace) -> None:
    if not Path(args.base_model_path).exists():
        raise FileNotFoundError(f"基座模型不存在: {args.base_model_path}")
    if args.mode == "lora" and not Path(args.adapter_path).exists():
        raise FileNotFoundError(f"LoRA adapter 不存在: {args.adapter_path}")
    if not Path(args.dataset_path).exists():
        raise FileNotFoundError(f"测试集不存在: {args.dataset_path}")
    if args.limit < 0:
        raise ValueError("--limit 不能小于 0")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens 必须大于 0")
    if args.report_max_samples <= 0:
        raise ValueError("--report-max-samples 必须大于 0")


def main() -> None:
    args = parse_args()
    validate_args(args)

    print(">>> 开始加载测试集...")
    samples = load_qa_samples(args.dataset_path, limit=args.limit)
    if not samples:
        raise ValueError("测试集为空，请检查 dataset-path")
    print(f">>> 样本数: {len(samples)}")

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    print(f">>> 推理设备: {device}")
    print(f">>> 推理精度: {dtype}")

    print(">>> 开始加载模型...")
    generator = LocalQwenGenerator(
        mode=args.mode,
        base_model_path=args.base_model_path,
        adapter_path=args.adapter_path,
        device=device,
        dtype=dtype,
    )

    run_id = build_run_id(args.mode, args.base_model_path, args.dataset_path, args.run_name)
    run_dir = Path(args.runs_dir) / run_id
    ensure_dir(run_dir)

    print(">>> 开始逐条生成答案并计算指标...")
    records: List[PredictionRecord] = []
    for index, sample in enumerate(samples, start=1):
        predicted_answer = generator.generate(
            sample=sample,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        exact_match = exact_match_score(sample.expected_answer, predicted_answer)
        token_f1 = token_f1_score(sample.expected_answer, predicted_answer)
        rouge_l_f1 = rouge_l_f1_score(sample.expected_answer, predicted_answer)
        no_answer_match = (
            1.0 if is_no_answer_text(sample.expected_answer) == is_no_answer_text(predicted_answer) else 0.0
        )

        record = PredictionRecord(
            sample_id=sample.sample_id,
            question=sample.question,
            instruction=sample.instruction,
            expected_answer=sample.expected_answer,
            predicted_answer=predicted_answer,
            normalized_expected_answer=normalize_text(sample.expected_answer),
            normalized_predicted_answer=normalize_text(predicted_answer),
            exact_match=exact_match,
            token_f1=token_f1,
            rouge_l_f1=rouge_l_f1,
            answerable_expected=not is_no_answer_text(sample.expected_answer),
            answerable_predicted=not is_no_answer_text(predicted_answer),
            no_answer_match=no_answer_match,
            status=classify_status(sample.expected_answer, predicted_answer, exact_match, no_answer_match),
        )
        records.append(record)

        print(
            f"[{index}/{len(samples)}] sample_id={sample.sample_id} "
            f"em={record.exact_match:.1f} token_f1={record.token_f1:.4f} status={record.status}"
        )

    metrics = evaluate_predictions(records)
    generated_at = datetime.now().isoformat(timespec="seconds")

    config_payload = {
        "run_id": run_id,
        "generated_at": generated_at,
        "mode": args.mode,
        "base_model_path": args.base_model_path,
        "adapter_path": args.adapter_path if args.mode == "lora" else "",
        "dataset_path": args.dataset_path,
        "run_name": args.run_name,
        "limit": args.limit,
        "device": device,
        "dtype": str(dtype),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "report_max_samples": args.report_max_samples,
    }

    result_payload = {
        "run_id": run_id,
        "generated_at": generated_at,
        "summary": {
            "mode": args.mode,
            "base_model_path": args.base_model_path,
            "adapter_path": args.adapter_path if args.mode == "lora" else "",
            "dataset_path": args.dataset_path,
            "sample_count": len(records),
        },
        "metrics": metrics,
    }

    config_path = run_dir / "config.json"
    result_path = run_dir / "result.json"
    predictions_path = run_dir / "predictions.jsonl"
    report_path = run_dir / "report.md"

    save_json(config_path, config_payload)
    save_json(result_path, result_payload)
    save_jsonl(predictions_path, (asdict(record) for record in records))
    report_path.write_text(
        build_report_markdown(run_id, config_payload, result_payload, records, args.report_max_samples),
        encoding="utf-8",
    )

    print("\n===== Qwen QA Evaluation Results =====")
    print(f"run_id:             {run_id}")
    print(f"mode:               {args.mode}")
    print(f"sample_count:       {metrics['sample_count']}")
    print(f"exact_match:        {metrics['exact_match']:.6f}")
    print(f"token_f1:           {metrics['token_f1']:.6f}")
    print(f"rouge_l_f1:         {metrics['rouge_l_f1']:.6f}")
    print(f"no_answer_accuracy: {metrics['no_answer_accuracy']:.6f}")
    print(f"false_no_answer:    {metrics['false_no_answer_count']}")
    print(f"missed_no_answer:   {metrics['missed_no_answer_count']}")
    print(f"\n>>> 配置已保存: {config_path}")
    print(f">>> 结果已保存: {result_path}")
    print(f">>> 逐条预测已保存: {predictions_path}")
    print(f">>> Markdown 报告已保存: {report_path}")


if __name__ == "__main__":
    main()
