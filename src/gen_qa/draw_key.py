"""
基于已经生成好的 QA 数据，批量抽取每条 QA 的核心关键词。
输出格式：
{
    "unique_id": "...",
    "question": "...",
    "answer": "...",
    "keywords": ["...", "..."]
}
"""

import os
import re
import json
import time
import random
import threading
import concurrent.futures
from datetime import datetime

from tqdm.auto import tqdm
from openai import OpenAI

random.seed(42)

MAX_WORKERS = 5
BATCH_SIZE = 12
LLM_MAX_RETRY = 3


class FatalLLMError(RuntimeError):
    pass


class ConsecutiveLLMFailureError(RuntimeError):
    pass

#处理测试集
# QA_INPUT_PATH = "/root/autodl-tmp/IR-RAG-System/data/qa_pairs/test_qa_pair.json"
# KEYWORDS_OUTPUT_PATH = "/root/autodl-tmp/IR-RAG-System/data/qa_pairs/test_qa_keywords.jsonl"
#PROGRESS_PATH = "/root/autodl-tmp/IR-RAG-System/data/qa_pairs/progress/test_keywords_progress.json"
#处理训练集
QA_INPUT_PATH = "/root/autodl-tmp/IR-RAG-System/data/qa_pairs/train_qa_pair.json"
KEYWORDS_OUTPUT_PATH = "/root/autodl-tmp/IR-RAG-System/data/qa_pairs/train_qa_keywords.jsonl"
PROGRESS_PATH = "/root/autodl-tmp/IR-RAG-System/data/qa_pairs/progress/train_keywords_progress.json"

llm_client = OpenAI(
    api_key=os.environ["DOUBAO_API_KEY"],
    base_url=os.environ["DOUBAO_BASE_URL"]
)

# =========================
# Batch Prompt Template
# =========================

KEYWORDS_BATCH_PROMPT_TPL = """
你是一名专业的信息检索（Information Retrieval）与搜索系统方向的NLP工程师，任务是从给定的问答对中提取最核心的专业关键词。

请严格按照以下要求执行：

【抽取原则】
1. 优先提取信息检索核心术语（如 "inverted index", "Boolean retrieval", "tf-idf"）
2. 保留重要模型/方法名称（如 "vector space model", "BM25", "language model"）
3. 提取关键技术或机制（如 "tokenization", "stemming", "ranking", "indexing"）
4. 包含重要概念或定义对象（如 "document collection", "query", "term frequency"）
5. 保留具有区分性的英文术语，优先使用原始英文表达
6. 可以综合 question 和 answer 来抽取，不要只看问题
7. 必须为每个 unique_id 都返回结果，不允许遗漏任何一条

【重点关注领域】
- 检索模型（Boolean / Vector / Probabilistic / Language Model）
- 索引结构（inverted index / dictionary / postings list）
- 排序与评分（tf-idf / BM25 / cosine similarity）
- 文本处理（tokenization / normalization / stemming）
- 查询处理（query expansion / relevance feedback）
- 系统架构（retrieval system / search engine pipeline）

【过滤规则】
1. 过滤通用词（如 "use", "include", "example", "method"）
2. 避免无信息词（如 "this", "that", "they"）
3. 避免过长短语，优先核心名词短语
4. 不要输出完整句子
5. 不要机械重复 question 中无意义的疑问表达
6. 如果没有有效关键词，返回空数组 []

【输出要求】
1. 只输出一个 JSON 对象，不要输出任何额外说明
2. JSON 的 key 必须是我给出的 unique_id
3. value 必须是关键词数组
4. 每条数据关键词数量不超过 5 个
5. 严格按照下面格式输出：

{
  "unique_id_1": ["keyword1", "keyword2"],
  "unique_id_2": ["keyword1", "keyword2", "keyword3"]
}

下面是多个 QA 数据：
{{qa_pairs}}

请输出结果：
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

    fenced = re.search(r"```(?:json)?\s*([\[{].*?[\]}])\s*```", text, flags=re.S | re.I)
    if fenced:
        return json.loads(fenced.group(1))

    tagged = re.search(r"<result>\s*([\[{].*?[\]}])\s*</result>", text, flags=re.S | re.I)
    if tagged:
        return json.loads(tagged.group(1))

    try:
        return json.loads(text)
    except Exception:
        pass

    candidates = re.findall(r"[\[{].*?[\]}]", text, flags=re.S)
    for c in candidates:
        try:
            return json.loads(c)
        except Exception:
            continue

    raise ValueError("未找到可解析的 JSON")


def is_fatal_llm_error(exc: Exception) -> bool:
    message = str(exc)
    lowered = message.lower()
    return (
        "allocationquota.freetieronly" in lowered
        or "free tier" in lowered
        or "parameter.enable_thinking" in message
        or "invalid_parameter_error" in lowered
    )


def is_rate_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "429" in message or "limit_requests" in message or "request limit" in message


def chat(prompt, max_retry=LLM_MAX_RETRY, debug=False, temperature=0.2, top_p=0.9):
    def do_chat(p):
        completion = llm_client.chat.completions.create(
            model=os.environ["DOUBAO_MODEL_NAME"],
            messages=[
                {
                    "role": "system",
                    "content": "你是一个严谨的人工智能助手，擅长抽取信息检索领域的高质量专业关键词。"
                },
                {"role": "user", "content": p}
            ],
            top_p=top_p,
            temperature=temperature,
            extra_body={"enable_thinking": False},
        )
        return completion.choices[0].message.content

    remain = max_retry
    while remain > 0:
        try:
            return do_chat(prompt)
        except Exception as e:
            if is_fatal_llm_error(e):
                if debug:
                    print(f"[chat fatal] {e}")
                raise FatalLLMError(str(e)) from e

            remain -= 1
            if is_rate_limit_error(e):
                sleep_seconds = random.randint(8, 15)
            else:
                sleep_seconds = random.randint(1, 4)
            if debug:
                print(f"[chat error] {e}; remain_retry={remain}; sleep={sleep_seconds}s")
            if remain > 0:
                time.sleep(sleep_seconds)

    return None


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


# =========================
# Load Input
# =========================

def normalize_qa_item(item, source: str, index_hint):
    if not isinstance(item, dict):
        print(f"[skip] {source} item {index_hint}: not a JSON object")
        return None

    unique_id = normalize_text(item.get("unique_id", ""))
    question = normalize_text(item.get("question", ""))
    answer = normalize_text(item.get("answer", ""))

    if not unique_id:
        print(f"[skip] {source} item {index_hint}: missing unique_id")
        return None
    if not question or not answer:
        print(f"[skip] {source} item {index_hint}: missing question/answer")
        return None

    return {
        "unique_id": unique_id,
        "question": question,
        "answer": answer,
    }



def load_qa_jsonl(path: str):
    """
    兼容两种输入格式：
    1. JSON 数组：[ {...}, {...} ]
    2. JSONL：每行一个 {...}
    """
    rows = []
    if not os.path.exists(path):
        print(f"[warn] input not found: {path}")
        return rows

    with open(path, "r", encoding="utf-8") as f:
        raw_text = f.read().strip()

    if not raw_text:
        return rows

    try:
        parsed = json.loads(raw_text)
    except Exception:
        parsed = None

    if isinstance(parsed, list):
        for idx, item in enumerate(parsed, start=1):
            normalized = normalize_qa_item(item, path, idx)
            if normalized is not None:
                rows.append(normalized)
        return rows

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

            normalized = normalize_qa_item(item, path, line_no)
            if normalized is not None:
                rows.append(normalized)

    return rows


# =========================
# Prompt Builder
# =========================

def build_batch_qa_text(batch_items):
    parts = []
    for item in batch_items:
        uid = item["unique_id"]
        q = item["question"]
        a = item["answer"]

        parts.append(
            f'<item unique_id="{uid}">\n'
            f'<question>{q}</question>\n'
            f'<answer>{a}</answer>\n'
            f'</item>'
        )
    return "\n\n".join(parts)


def build_batch_prompt(batch_items):
    qa_text = build_batch_qa_text(batch_items)
    return KEYWORDS_BATCH_PROMPT_TPL.replace("{{qa_pairs}}", qa_text).strip()


# =========================
# Parse / Validate
# =========================

def clean_keyword(kw: str) -> str:
    kw = normalize_text(kw)
    kw = kw.strip(",，;； ")
    return kw


def validate_keywords_list(resp):
    """
    期望 resp 是 list[str]
    """
    if not isinstance(resp, list):
        return []

    cleaned = []
    seen = set()

    for x in resp:
        if not isinstance(x, str):
            continue

        kw = clean_keyword(x)
        if not kw:
            continue

        low = kw.lower()
        if low in seen:
            continue

        seen.add(low)
        cleaned.append(kw)

    return cleaned[:5]


def parse_batch_response(raw_resp, batch_items):
    parsed = extract_json_from_text(raw_resp)
    if not isinstance(parsed, dict):
        return {}

    result = {}
    expected_uids = [item["unique_id"] for item in batch_items]

    for uid in expected_uids:
        kw_list = parsed.get(uid, [])
        result[uid] = validate_keywords_list(kw_list)

    return result


# =========================
# Checkpoint / Resume
# =========================

def load_existing_results(result_path: str):
    """
    加载已经写入结果文件的内容：
    - processed_ids: 已完成 unique_id 集合
    - result_ckpt: unique_id -> item
    """
    processed_ids = set()
    result_ckpt = {}

    if not os.path.exists(result_path):
        return processed_ids, result_ckpt

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

            uid = item.get("unique_id")
            if not uid:
                continue

            processed_ids.add(uid)
            result_ckpt[uid] = item

    return processed_ids, result_ckpt


def save_progress(progress_path: str, processed_ids, total_count: int):
    """
    每写入一个样本结果，就更新一次进度文件
    """
    ensure_parent_dir(progress_path)

    progress = {
        "total_count": total_count,
        "processed_count": len(processed_ids),
        "remaining_count": max(total_count - len(processed_ids), 0),
        "processed_ids": sorted(processed_ids),
        "last_update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    tmp_path = progress_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

    os.replace(tmp_path, progress_path)


def append_result_line_atomic(output_path: str, item: dict):
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def rewrite_result_file_without_uids(output_path: str, remove_uids):
    remove_uids = set(remove_uids)
    if not remove_uids or not os.path.exists(output_path):
        return

    kept_lines = []
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except Exception:
                kept_lines.append(raw)
                continue

            if item.get("unique_id") in remove_uids:
                continue
            kept_lines.append(raw)

    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        if kept_lines:
            f.write("\n".join(kept_lines) + "\n")
    os.replace(tmp_path, output_path)


def rollback_failed_progress(output_path: str, progress_path: str, failed_uids, total_count: int):
    rewrite_result_file_without_uids(output_path, failed_uids)
    processed_ids, _ = load_existing_results(output_path)
    save_progress(progress_path, processed_ids, total_count)


# =========================
# Main Keywords Generation
# =========================

def gen_keywords_for_qas(qa_items, output_path, progress_path):
    ensure_parent_dir(output_path)
    ensure_parent_dir(progress_path)

    file_lock = threading.Lock()
    total_count = len(qa_items)

    processed_ids, result_ckpt = load_existing_results(output_path)
    print(f"已存在结果数：{len(processed_ids)} / {total_count}")

    remaining_items = [
        item for item in qa_items
        if item["unique_id"] not in processed_ids
    ]
    print(f"待处理剩余条数：{len(remaining_items)}")

    save_progress(progress_path, processed_ids, total_count)

    batches = [batch for batch in chunked(remaining_items, BATCH_SIZE) if batch]
    if not batches:
        print("没有待处理 batch，直接返回已有结果。")
        return result_ckpt

    consecutive_failed_uids = []
    future_to_batch = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for batch_items in batches:
            prompt = build_batch_prompt(batch_items)
            future = executor.submit(chat, prompt, LLM_MAX_RETRY, True)
            future_to_batch[future] = batch_items

        for future in tqdm(
            concurrent.futures.as_completed(future_to_batch),
            total=len(future_to_batch),
            desc="生成关键词(batch)"
        ):
            batch_items = future_to_batch[future]
            batch_uids = [item["unique_id"] for item in batch_items]

            try:
                raw_resp = future.result()
            except FatalLLMError as e:
                rollback_failed_progress(output_path, progress_path, consecutive_failed_uids, total_count)
                raise ConsecutiveLLMFailureError(
                    f"检测到不可恢复的配额/参数错误，已终止关键词抽取：{e}"
                )
            except Exception as e:
                print(f"[future result error] {e}")
                raw_resp = None

            if raw_resp is None:
                consecutive_failed_uids.extend(batch_uids)
                print(f"[keyword batch failed] batch_size={len(batch_uids)}; consecutive_failed={len(consecutive_failed_uids)}")
                if len(consecutive_failed_uids) >= BATCH_SIZE:
                    rollback_failed_progress(output_path, progress_path, consecutive_failed_uids, total_count)
                    raise ConsecutiveLLMFailureError("连续关键词抽取失败达到阈值，已终止并回滚失败阶段断点。")
                continue

            try:
                batch_parsed = parse_batch_response(raw_resp, batch_items)
            except Exception as e:
                print(f"[parse batch error] {e}")
                consecutive_failed_uids.extend(batch_uids)
                if len(consecutive_failed_uids) >= BATCH_SIZE:
                    rollback_failed_progress(output_path, progress_path, consecutive_failed_uids, total_count)
                    raise ConsecutiveLLMFailureError("连续关键词解析失败达到阈值，已终止并回滚失败阶段断点。")
                continue

            batch_successful = False
            for qa_item in batch_items:
                uid = qa_item["unique_id"]
                keywords = batch_parsed.get(uid, [])

                if not keywords:
                    continue

                item = {
                    "unique_id": uid,
                    "question": qa_item["question"],
                    "answer": qa_item["answer"],
                    "keywords": keywords
                }

                with file_lock:
                    if uid in processed_ids:
                        continue

                    append_result_line_atomic(output_path, item)
                    result_ckpt[uid] = item
                    processed_ids.add(uid)
                    save_progress(progress_path, processed_ids, total_count)
                batch_successful = True

            if batch_successful:
                consecutive_failed_uids.clear()

    return result_ckpt


def main():
    qa_items = load_qa_jsonl(QA_INPUT_PATH)

    print(f"total qa items: {len(qa_items)}")

    if qa_items:
        print("sample unique_id:", qa_items[0]["unique_id"])
        print("sample question:", qa_items[0]["question"][:120])
        print("sample answer:", qa_items[0]["answer"][:120])

    try:
        result_dict = gen_keywords_for_qas(
            qa_items,
            KEYWORDS_OUTPUT_PATH,
            PROGRESS_PATH
        )
    except ConsecutiveLLMFailureError as e:
        print(str(e))
        print("结果保存路径：", KEYWORDS_OUTPUT_PATH)
        print("进度保存路径：", PROGRESS_PATH)
        return

    total_keywords = sum(len(v.get("keywords", [])) for v in result_dict.values())

    print("生成完成")
    print("已保存条数：", len(result_dict))
    print("总关键词数：", total_keywords)
    print("结果保存路径：", KEYWORDS_OUTPUT_PATH)
    print("进度保存路径：", PROGRESS_PATH)


if __name__ == "__main__":
    main()