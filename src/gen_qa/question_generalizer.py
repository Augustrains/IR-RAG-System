"""
基于已有评分后的 QA 文件，对高质量 question 做问题泛化，并导出训练/测试集。

输入：
- /root/autodl-tmp/IR-RAG-System/data/qa_pairs/qa_score.jsonl

中间输出：
- /root/autodl-tmp/IR-RAG-System/data/qa_pairs/qa_generalized.jsonl
- /root/autodl-tmp/IR-RAG-System/data/qa_pairs/progress/qa_generalized_progress.json

最终输出：
- /root/autodl-tmp/IR-RAG-System/data/qa_pairs/train_qa_pair.json
- /root/autodl-tmp/IR-RAG-System/data/qa_pairs/test_qa_pair.json

规则：
1. 只对 score >= 4 的问题做泛化；
2. 每个通过筛选的问题，最终保留“原始问题 + 泛化问题”，并拼接同一个 answer；
3. 最终按 9:1 划分训练集与测试集；
4. 样本格式为：{"unique_id": ..., "question": ..., "answer": ...}


最终QA条数： 1439
训练集QA数： 1295
测试集QA数： 144

"""

import os
import re
import json
import time
import random
import hashlib
import threading
import concurrent.futures
from datetime import datetime

from tqdm.auto import tqdm
from openai import OpenAI

random.seed(42)

# =========================
# Config
# =========================

MAX_WORKERS = 20
GENERALIZE_BATCH_SIZE = 12
MAX_BATCH_CHARS = 10000
LLM_MAX_RETRY = 3
MIN_SCORE_TO_GENERALIZE = 4
TRAIN_SPLIT_RATIO = 0.9

QA_SCORE_INPUT_PATH = "/root/autodl-tmp/IR-RAG-System/data/qa_pairs/qa_score.jsonl"
GENERALIZE_OUTPUT_PATH = "/root/autodl-tmp/IR-RAG-System/data/qa_pairs/qa_generalized.jsonl"
PROGRESS_PATH = "/root/autodl-tmp/IR-RAG-System/data/qa_pairs/progress/qa_generalized_progress.json"
TRAIN_OUTPUT_PATH = "/root/autodl-tmp/IR-RAG-System/data/qa_pairs/train_qa_pair.json"
TEST_OUTPUT_PATH = "/root/autodl-tmp/IR-RAG-System/data/qa_pairs/test_qa_pair.json"

llm_client = OpenAI(
    api_key=os.environ["DOUBAO_API_KEY"],
    base_url=os.environ["DOUBAO_BASE_URL"]
)

# =========================
# Prompt Templates
# =========================

GENERALIZE_BATCH_PROMPT_TPL = """
你是一个擅长问句改写与训练数据增强的助手。

我会给你多个信息检索（Information Retrieval, IR）领域的原始问题。
这些问题来自教材型 QA 数据，很多问题包含专业术语，例如：
- inverted index
- Boolean retrieval
- postings list
- tf-idf
- BM25
- precision / recall / MAP / NDCG
- tokenization / normalization / stemming
- wildcard query / phrase query / proximity query

你的任务：
对每个原始问题生成 5 个“语义等价或高度近义”的不同问法，用于训练检索与问答系统的查询泛化能力。

【生成要求】
1. 必须和原问题表达同一个知识点，不能扩大或改变原意；
2. 可以改变句式、措辞和表达风格，但不能引入原问题中没有的新知识；
3. 尽量让问法更自然、更像真实用户提问；
4. 允许口语化改写，但不要过度随意；
5. 对于专业术语，优先保留原术语，不要随意替换成不准确表达；
6. 如果原问题是中文，就输出中文泛化；如果原问题是英文，就输出英文泛化；
7. 不要回答问题，你的任务只是改写问题；
8. 每个问题必须返回 5 个不同问法；
9. 不要输出与原问题完全相同的句子；
10. 不要输出“这是什么意思”“请解释一下这句话”这种空泛问法。

【输出格式】
只输出一个 JSON 对象，不要输出任何额外说明。
JSON 的 key 必须是我给出的 sample_id，value 必须是长度为 5 的字符串数组。

格式示例：
{
  "sample_id_1": [
    "...",
    "...",
    "...",
    "...",
    "..."
  ],
  "sample_id_2": [
    "...",
    "...",
    "...",
    "...",
    "..."
  ]
}

下面是多个问题：
{{questions}}

请输出结果：
"""

GENERALIZE_SINGLE_PROMPT_TPL = """
你是一个擅长问句改写与训练数据增强的助手。

下面给你一个信息检索（IR）领域的问题：
{{question}}

你的任务：
生成 5 个与原问题语义等价或高度近义的不同问法。

要求：
1. 不能改变原问题的知识点；
2. 可以改变句式、措辞和表达风格；
3. 尽量更自然、更像真实用户提问；
4. 保留关键专业术语，不要引入新知识；
5. 如果原问题是中文，就输出中文泛化；如果原问题是英文，就输出英文泛化；
6. 不要回答问题；
7. 不要输出与原问题完全相同的句子。

只输出 JSON 数组：
[
  "...",
  "...",
  "...",
  "...",
  "..."
]
"""

# =========================
# Utils
# =========================

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def extract_json_from_text(text: str):
    text = (text or "").strip()
    if not text:
        raise ValueError("响应为空，无法解析 JSON")

    fenced = re.search(r"```(?:json)?\s*([\[{].*[\]}])\s*```", text, flags=re.S | re.I)
    if fenced:
        return json.loads(fenced.group(1))

    tagged = re.search(r"<result>\s*([\[{].*[\]}])\s*</result>", text, flags=re.S | re.I)
    if tagged:
        return json.loads(tagged.group(1))

    try:
        return json.loads(text)
    except Exception:
        pass

    candidates = re.findall(r"[\[{].*[\]}]", text, flags=re.S)
    for c in candidates:
        try:
            return json.loads(c)
        except Exception:
            continue

    raise ValueError("未找到可解析的 JSON")


def chat(prompt: str, max_retry: int = LLM_MAX_RETRY, debug: bool = False,
         temperature: float = 0.7, top_p: float = 0.95):
    def do_chat(p: str):
        completion = llm_client.chat.completions.create(
            model=os.environ["DOUBAO_MODEL_NAME"],
            messages=[
                {
                    "role": "system",
                    "content": "你是一个严谨的人工智能助手，擅长生成高质量的问题泛化训练数据。"
                },
                {"role": "user", "content": p}
            ],
            top_p=top_p,
            temperature=temperature,
        )
        return completion.choices[0].message.content

    remain = max_retry
    while remain > 0:
        try:
            return do_chat(prompt)
        except Exception as e:
            remain -= 1
            sleep_seconds = random.randint(1, 4)
            if debug:
                print(f"[chat error] {e}; remain_retry={remain}; sleep={sleep_seconds}s")
            if remain > 0:
                time.sleep(sleep_seconds)

    return None


def append_result_line_atomic(output_path: str, item: dict):
    ensure_parent_dir(output_path)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def build_query_unique_id(question: str) -> str:
    return hashlib.md5(question.encode("utf-8")).hexdigest()

# =========================
# Input Loader
# =========================

def load_qa_samples(path: str):
    """
    从 qa_score.jsonl 中读取 QA。
    每个 question 作为一个独立 sample：
    sample_id = 原文档unique_id#qa_idx
    仅保留 score >= MIN_SCORE_TO_GENERALIZE 的问题。
    """
    samples = []
    if not os.path.exists(path):
        print(f"[warn] input not found: {path}")
        return samples

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except Exception as e:
                print(f"[load error] {path} line {line_no}: {e}")
                continue

            doc_uid = item.get("unique_id")
            raw_resp = item.get("raw_resp", "")
            score_list = item.get("score", [])

            if not doc_uid:
                print(f"[skip] {path} line {line_no}: missing unique_id")
                continue

            try:
                qa_list = json.loads(raw_resp)
            except Exception as e:
                print(f"[skip] {path} line {line_no}: raw_resp parse failed: {e}")
                continue

            if not isinstance(qa_list, list) or not isinstance(score_list, list):
                continue

            for qa_idx, qa in enumerate(qa_list):
                if not isinstance(qa, dict):
                    continue

                score = score_list[qa_idx] if qa_idx < len(score_list) else 0
                try:
                    score = int(score)
                except Exception:
                    score = 0

                if score < MIN_SCORE_TO_GENERALIZE:
                    continue

                question = normalize_text(qa.get("question", ""))
                answer = normalize_text(qa.get("answer", ""))

                if not question or not answer:
                    continue

                sample_id = f"{doc_uid}#{qa_idx}"
                samples.append({
                    "sample_id": sample_id,
                    "doc_unique_id": doc_uid,
                    "qa_idx": qa_idx,
                    "question": question,
                    "answer": answer,
                    "score": score,
                })

    return samples

# =========================
# Prompt Builder
# =========================

def build_batch_questions_text(batch_samples):
    parts = []
    for sample in batch_samples:
        sid = sample["sample_id"]
        question = sample["question"]
        parts.append(
            f'<sample sample_id="{sid}">\n'
            f'{question}\n'
            f'</sample>'
        )
    return "\n\n".join(parts)


def build_batch_prompt(batch_samples):
    questions_text = build_batch_questions_text(batch_samples)
    return GENERALIZE_BATCH_PROMPT_TPL.replace("{{questions}}", questions_text).strip()


def build_single_prompt(sample):
    return GENERALIZE_SINGLE_PROMPT_TPL.replace("{{question}}", sample["question"]).strip()

# =========================
# Batch Decision
# =========================

def can_use_batch(batch_samples):
    if len(batch_samples) <= 1:
        return False

    total_chars = 0
    for sample in batch_samples:
        total_chars += len(sample["question"])

    return total_chars <= MAX_BATCH_CHARS


def build_batches(samples):
    return list(chunked(samples, GENERALIZE_BATCH_SIZE))

# =========================
# Validation / Format
# =========================

def validate_question_list(resp, source_question: str):
    if not isinstance(resp, list):
        return []

    cleaned = []
    source_norm = normalize_text(source_question).lower()

    for item in resp:
        if not isinstance(item, str):
            item = str(item)

        question = normalize_text(item)
        if not question:
            continue

        if question.lower() == source_norm:
            continue

        cleaned.append(question)

    deduped = []
    seen = set()
    for question in cleaned:
        key = question.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(question)

    return deduped[:5]


def pad_to_five(question_list):
    question_list = list(question_list or [])
    if not question_list:
        return []

    while len(question_list) < 5:
        question_list.append(question_list[-1])

    return question_list[:5]


def format_numbered_lines(question_list):
    question_list = pad_to_five(question_list)
    if not question_list:
        return ""

    lines = []
    for idx, question in enumerate(question_list, start=1):
        lines.append(f"{idx}. {question}")
    return "\n".join(lines)


def parse_numbered_lines(raw_resp: str):
    questions = []
    for line in (raw_resp or "").splitlines():
        line = normalize_text(re.sub(r"^\s*\d+\.\s*", "", line))
        if line:
            questions.append(line)
    return questions


def parse_batch_response(raw_resp, batch_samples):
    parsed = extract_json_from_text(raw_resp)
    if not isinstance(parsed, dict):
        return {}

    result = {}
    expected_ids = [sample["sample_id"] for sample in batch_samples]
    question_map = {sample["sample_id"]: sample["question"] for sample in batch_samples}

    for sample_id in expected_ids:
        question_list = parsed.get(sample_id, [])
        result[sample_id] = validate_question_list(question_list, question_map[sample_id])

    return result


def parse_single_response(raw_resp, sample):
    parsed = extract_json_from_text(raw_resp)
    return validate_question_list(parsed, sample["question"])

# =========================
# Resume / Progress
# =========================

def load_progress(progress_path: str):
    if not os.path.exists(progress_path):
        return set()

    try:
        with open(progress_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        processed = data.get("processed_sample_ids", [])
        return set(processed)
    except Exception as e:
        print(f"[load progress error] {e}")
        return set()


def load_existing_results(result_path: str):
    existing_source_uid_set = set()

    if not os.path.exists(result_path):
        return existing_source_uid_set

    with open(result_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except Exception as e:
                print(f"[resume load error] {result_path} line {line_no}: {e}")
                continue

            source_uid = normalize_text(item.get("source_uid", ""))
            if source_uid:
                existing_source_uid_set.add(source_uid)

    return existing_source_uid_set


def save_progress(progress_path: str, processed_sample_ids, total_samples: int):
    ensure_parent_dir(progress_path)

    progress = {
        "total_samples": total_samples,
        "processed_samples": len(processed_sample_ids),
        "remaining_samples": max(total_samples - len(processed_sample_ids), 0),
        "processed_sample_ids": sorted(processed_sample_ids),
        "last_update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    tmp_path = progress_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

    os.replace(tmp_path, progress_path)

# =========================
# Main Generalization
# =========================

def gen_generalized_questions(samples, output_path, progress_path):
    ensure_parent_dir(output_path)
    ensure_parent_dir(progress_path)

    file_lock = threading.Lock()
    total_samples = len(samples)

    processed_sample_ids = load_progress(progress_path)
    existing_source_uid_set = load_existing_results(output_path)

    print(f"progress中已处理样本数：{len(processed_sample_ids)} / {total_samples}")
    print(f"结果文件中已存在source_uid数：{len(existing_source_uid_set)}")

    remaining_samples = []
    for sample in samples:
        sample_id = sample["sample_id"]
        if sample_id in processed_sample_ids:
            continue
        if sample_id in existing_source_uid_set:
            continue
        remaining_samples.append(sample)

    print(f"待处理剩余样本数：{len(remaining_samples)}")

    save_progress(progress_path, processed_sample_ids, total_samples)

    batches = build_batches(remaining_samples)
    if not batches:
        print("没有待处理 batch，直接结束。")
        return

    future_to_task = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for batch_samples in batches:
            if can_use_batch(batch_samples):
                prompt = build_batch_prompt(batch_samples)
                future = executor.submit(chat, prompt, LLM_MAX_RETRY, True)
                future_to_task[future] = {
                    "mode": "batch",
                    "batch_samples": batch_samples,
                }
            else:
                for sample in batch_samples:
                    prompt = build_single_prompt(sample)
                    future = executor.submit(chat, prompt, LLM_MAX_RETRY, True)
                    future_to_task[future] = {
                        "mode": "single",
                        "batch_samples": [sample],
                    }

        pbar = tqdm(
            concurrent.futures.as_completed(future_to_task),
            total=len(future_to_task),
            desc="生成问题泛化",
            dynamic_ncols=True
        )

        for future in pbar:
            task_info = future_to_task[future]
            mode = task_info["mode"]
            batch_samples = task_info["batch_samples"]

            try:
                raw_resp = future.result()
            except Exception as e:
                print(f"[future result error] {e}")
                raw_resp = None

            if raw_resp is None:
                batch_parsed = {
                    sample["sample_id"]: []
                    for sample in batch_samples
                }
            else:
                try:
                    if mode == "batch":
                        batch_parsed = parse_batch_response(raw_resp, batch_samples)
                    else:
                        sample = batch_samples[0]
                        batch_parsed = {
                            sample["sample_id"]: parse_single_response(raw_resp, sample)
                        }
                except Exception as e:
                    print(f"[parse error] {e}")
                    batch_parsed = {
                        sample["sample_id"]: []
                        for sample in batch_samples
                    }

            for sample in batch_samples:
                sample_id = sample["sample_id"]
                question_text = sample["question"]
                generalized_questions = batch_parsed.get(sample_id, [])
                generalized_text = format_numbered_lines(generalized_questions)

                item = {
                    "unique_id": question_text,
                    "source_uid": sample_id,
                    "raw_resp": generalized_text,
                }

                with file_lock:
                    if sample_id in processed_sample_ids:
                        continue

                    append_result_line_atomic(output_path, item)
                    processed_sample_ids.add(sample_id)
                    save_progress(progress_path, processed_sample_ids, total_samples)

            processed_count = len(processed_sample_ids)
            pbar.set_postfix({
                "done": processed_count,
                "remain": max(total_samples - processed_count, 0),
            })


def load_generalized_result_map(result_path: str):
    result_map = {}
    if not os.path.exists(result_path):
        return result_map

    with open(result_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except Exception as e:
                print(f"[load generalized error] {result_path} line {line_no}: {e}")
                continue

            source_uid = normalize_text(item.get("source_uid", ""))
            if not source_uid:
                continue

            result_map[source_uid] = parse_numbered_lines(item.get("raw_resp", ""))

    return result_map


def build_final_qa_pairs(samples, generalized_map):
    qa_pairs = []

    for sample in samples:
        question_candidates = [sample["question"]]
        question_candidates.extend(generalized_map.get(sample["sample_id"], []))

        deduped_questions = []
        seen = set()
        for question in question_candidates:
            question = normalize_text(question)
            if not question:
                continue
            key = question.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped_questions.append(question)

        for question in deduped_questions:
            qa_pairs.append({
                "unique_id": build_query_unique_id(question),
                "question": question,
                "answer": sample["answer"],
            })

    return qa_pairs


def save_train_test_pairs(qa_pairs, train_path: str, test_path: str):
    ensure_parent_dir(train_path)
    ensure_parent_dir(test_path)

    items = list(qa_pairs)
    rng = random.Random(42)
    rng.shuffle(items)

    if not items:
        train_items = []
        test_items = []
    else:
        split_idx = int(len(items) * TRAIN_SPLIT_RATIO)
        split_idx = max(1, split_idx) if len(items) > 1 else len(items)
        split_idx = min(split_idx, len(items))
        train_items = items[:split_idx]
        test_items = items[split_idx:]

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_items, f, ensure_ascii=False, indent=2)

    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_items, f, ensure_ascii=False, indent=2)

    return train_items, test_items


def main():
    samples = load_qa_samples(QA_SCORE_INPUT_PATH)

    print(f"score>=4 的样本数: {len(samples)}")
    if samples:
        print("sample question:", samples[0]["question"])
        print("sample internal sample_id:", samples[0]["sample_id"])
        print("sample source doc unique_id:", samples[0]["doc_unique_id"])

    gen_generalized_questions(
        samples=samples,
        output_path=GENERALIZE_OUTPUT_PATH,
        progress_path=PROGRESS_PATH
    )

    generalized_map = load_generalized_result_map(GENERALIZE_OUTPUT_PATH)
    qa_pairs = build_final_qa_pairs(samples, generalized_map)
    train_pairs, test_pairs = save_train_test_pairs(
        qa_pairs=qa_pairs,
        train_path=TRAIN_OUTPUT_PATH,
        test_path=TEST_OUTPUT_PATH,
    )

    print("生成完成")
    print("泛化中间结果路径：", GENERALIZE_OUTPUT_PATH)
    print("进度保存路径：", PROGRESS_PATH)
    print("最终QA条数：", len(qa_pairs))
    print("训练集QA数：", len(train_pairs))
    print("测试集QA数：", len(test_pairs))
    print("训练集保存路径：", TRAIN_OUTPUT_PATH)
    print("测试集保存路径：", TEST_OUTPUT_PATH)


if __name__ == "__main__":
    main()
