"""
利用前面过滤出来的文档块,生成QA问答对
规定core类型的文档生成4个QA对,extra文档生成1个
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

MAX_WORKERS = 20
CORE_BATCH_SIZE = 4
EXTRA_BATCH_SIZE = 8

CORE_INPUT_PATH = "/root/autodl-tmp/IR-RAG-System/data/processed_docs/qa_filter_outputs/core_docs.jsonl"
EXTRA_INPUT_PATH = "/root/autodl-tmp/IR-RAG-System/data/processed_docs/qa_filter_outputs/extra_docs.jsonl"
QA_OUTPUT_PATH = "/root/autodl-tmp/IR-RAG-System/data/qa_pairs/qa_pair.jsonl"
PROGRESS_PATH = "/root/autodl-tmp/IR-RAG-System/data/qa_pairs/progress/qa_progress.json"

llm_client = OpenAI(
    api_key=os.environ["DOUBAO_API_KEY"],
    base_url=os.environ["DOUBAO_BASE_URL"]
)

# =========================
# Batch Prompt Templates
# =========================

CORE_BATCH_QA_PROMPT_TPL = """
你是一名信息检索（Information Retrieval, IR）与搜索系统方向的专家助教。

我会给你多个核心文档块，这些文本来自 IR 教材正文，通常具有较强的独立语义，内容可能涉及：
- information retrieval 的定义、任务与应用范围
- structured / unstructured / semistructured data
- clustering / classification
- web search / enterprise search / personal information retrieval
- linear scan / grep / flexible matching
- ranked retrieval / indexing
- Boolean retrieval model
- collection / corpus / document / term 等基础概念

你的任务：
对每个文档分别生成 4 个高质量问答对（QA）。

【问题要求】
1. 问题必须直接基于对应文档，不能脱离文本凭空扩展；
2. 优先生成：
   - 定义型问题
   - 原理型问题
   - 解释型问题
   - 对比型问题
3. 每个文档的 4 个问题中，至少 1 个问题需要综合该文档中两句及以上的信息；
4. 不要生成以下低质量问题：
   - “这段讲了什么？”
   - “这一章讲了什么？”
   - 纯粹复述原句表面的问法
   - 依赖图像本身才能回答的问题
5. 若文本中出现图表引用，但文本本身已经给出了解释，应围绕解释后的概念提问，而不是问图像本身。

【答案要求】
1. 答案必须准确、完整、可以独立理解；
2. 不要出现“文中提到”“该段指出”“见上一段”“见某页”之类依赖上下文的话；
3. 答案简洁但必须覆盖关键点；
4. 回答语言与问题一致。

【输出要求】
1. 只输出一个 JSON 对象，不要输出任何额外说明；
2. JSON 的 key 必须是我给出的文档 unique_id；
3. value 必须是该文档对应的 QA 数组；
4. 每个核心文档必须返回 4 个 QA；如果确实无法生成，返回空数组 []；
5. 严格按照下面格式输出：

{
  "unique_id_1": [
    {"question": "...", "answer": "..."},
    {"question": "...", "answer": "..."},
    {"question": "...", "answer": "..."},
    {"question": "...", "answer": "..."}
  ],
  "unique_id_2": [
    {"question": "...", "answer": "..."},
    {"question": "...", "answer": "..."},
    {"question": "...", "answer": "..."},
    {"question": "...", "answer": "..."}
  ]
}

下面是多个文档：
{{documents}}

请输出结果：
"""

EXTRA_BATCH_QA_PROMPT_TPL = """
你是一名信息检索（Information Retrieval, IR）与搜索系统方向的专家助教。

我会给你多个扩展文档块，这些文本通常较短，往往只表达一个局部知识点，例如：
- 某个概念的补充定义
- 某个术语或对象的说明
- 某个检索任务、检索场景或操作方式的简短解释
- 某个例子的局部结论

你的任务：
对每个文档分别生成 1 个高质量问答对（QA）。

【问题要求】
1. 只围绕该文档中最明确、最稳定、最能独立成立的信息提问；
2. 优先生成：
   - 定义型问题
   - 单点事实型问题
   - 简短解释型问题
3. 不要生成需要大段上下文才能回答的问题；
4. 不要生成依赖图像、表格或章节上下文的问题；
5. 问题要自然，避免机械改写原句。

【答案要求】
1. 答案必须能够脱离上下文独立理解；
2. 表述准确、简洁；
3. 不要出现“文中提到”“本段指出”等措辞；
4. 回答语言与问题一致。

【输出要求】
1. 只输出一个 JSON 对象，不要输出任何额外说明；
2. JSON 的 key 必须是我给出的文档 unique_id；
3. value 必须是该文档对应的 QA 数组；
4. 每个扩展文档只返回 1 个 QA；如果确实不适合生成，则返回空数组 []；
5. 严格按照下面格式输出：

{
  "unique_id_1": [
    {"question": "...", "answer": "..."}
  ],
  "unique_id_2": [
    {"question": "...", "answer": "..."}
  ]
}

下面是多个文档：
{{documents}}

请输出结果：
"""

# =========================
# Utils
# =========================

#处理文本
def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()

#确保路径
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

    # 尝试直接整体解析
    try:
        return json.loads(text)
    except Exception:
        pass

    # 回退：从文本中抽取 JSON 片段
    candidates = re.findall(r"[\[{].*?[\]}]", text, flags=re.S)
    for c in candidates:
        try:
            return json.loads(c)
        except Exception:
            continue

    raise ValueError("未找到可解析的 JSON")


def chat(prompt, max_retry=3, debug=False, temperature=0.7, top_p=0.95):
    def do_chat(p):
        completion = llm_client.chat.completions.create(
            model=os.environ["DOUBAO_MODEL_NAME"],
            messages=[
                {
                    "role": "system",
                    "content": "你是一个严谨的人工智能助手，擅长生成高质量的信息检索领域问答数据。"
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


def load_jsonl_docs(path: str, expected_type: str):
    docs = []
    if not os.path.exists(path):
        print(f"[warn] input not found: {path}")
        return docs

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

            page_content = item.get("page_content", "")
            metadata = item.get("metadata", {}) or {}

            if not isinstance(page_content, str):
                page_content = str(page_content)

            page_content = page_content.strip()
            if not page_content:
                continue

            unique_id = metadata.get("unique_id")
            if not unique_id:
                print(f"[skip] {path} line {line_no}: missing unique_id")
                continue

            docs.append({
                "page_content": page_content,
                "metadata": metadata,
                "doc_type": expected_type
            })

    return docs


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def build_batch_documents_text(batch_docs):
    parts = []
    for doc in batch_docs:
        uid = doc["metadata"]["unique_id"]
        text = doc["page_content"]
        parts.append(
            f'<doc unique_id="{uid}">\n'
            f'{text}\n'
            f'</doc>'
        )
    return "\n\n".join(parts)


def build_batch_prompt(batch_docs, doc_type: str):
    docs_text = build_batch_documents_text(batch_docs)
    if doc_type == "core":
        return CORE_BATCH_QA_PROMPT_TPL.replace("{{documents}}", docs_text).strip()
    elif doc_type == "extra":
        return EXTRA_BATCH_QA_PROMPT_TPL.replace("{{documents}}", docs_text).strip()
    else:
        raise ValueError(f"未知 doc_type: {doc_type}")

#进一步提取大模型生成的QA,核心文档生成4对,其余文档生成1对
def validate_single_doc_qa_list(resp, doc_type: str):
    if not isinstance(resp, list):
        return []

    cleaned = []
    for item in resp:
        if not isinstance(item, dict):
            continue

        q = normalize_text(item.get("question", ""))
        a = normalize_text(item.get("answer", ""))

        if not q or not a:
            continue

        cleaned.append({
            "question": q,
            "answer": a
        })

    if doc_type == "core":
        return cleaned[:4]
    elif doc_type == "extra":
        return cleaned[:1]
    return []

#提取大模型的内容
def parse_batch_response(raw_resp, batch_docs, doc_type: str):
    parsed = extract_json_from_text(raw_resp)
    if not isinstance(parsed, dict):
        return {}

    result = {}
    expected_uids = [doc["metadata"]["unique_id"] for doc in batch_docs]

    for uid in expected_uids:
        qa_list = parsed.get(uid, [])
        result[uid] = validate_single_doc_qa_list(qa_list, doc_type)

    return result


# =========================
# Checkpoint / Resume
# =========================

def load_existing_results(result_path: str):
    """
    加载已经写入结果文件的内容：
    - processed_ids: 已完成 unique_id 集合
    - qa_ckpt: unique_id -> item
    """
    processed_ids = set()
    qa_ckpt = {}

    if not os.path.exists(result_path):
        return processed_ids, qa_ckpt

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
            qa_ckpt[uid] = item

    return processed_ids, qa_ckpt


def save_progress(progress_path: str, processed_ids, total_docs: int):
    """
    每写入一个文档结果，就更新一次进度文件
    """
    ensure_parent_dir(progress_path)

    progress = {
        "total_docs": total_docs,
        "processed_docs": len(processed_ids),
        "remaining_docs": max(total_docs - len(processed_ids), 0),
        "processed_ids": sorted(processed_ids),
        "last_update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    tmp_path = progress_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

    os.replace(tmp_path, progress_path)

#结果保存到文档
def append_result_line_atomic(output_path: str, item: dict):
    """
    以追加方式写入一行 JSONL
    """
    ensure_parent_dir(output_path)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


# =========================
# Main QA generation
# =========================

def gen_qa_for_docs(docs, qa_ckpt_filename, progress_path):
    ensure_parent_dir(qa_ckpt_filename)
    ensure_parent_dir(progress_path)

    file_lock = threading.Lock()
    total_docs = len(docs)

    # 读取已有结果，支持断点续跑
    processed_ids, qa_ckpt = load_existing_results(qa_ckpt_filename)
    print(f"已存在结果数：{len(processed_ids)} / {total_docs}")

    # 跳过已完成文档
    remaining_docs = [
        doc for doc in docs
        if doc["metadata"]["unique_id"] not in processed_ids
    ]
    print(f"待处理剩余文档数：{len(remaining_docs)}")

    # 初始化一次 progress，防止中断时没有进度文件
    save_progress(progress_path, processed_ids, total_docs)

    core_docs = [d for d in remaining_docs if d["doc_type"] == "core"]
    extra_docs = [d for d in remaining_docs if d["doc_type"] == "extra"]

    batches = []
    for batch in chunked(core_docs, CORE_BATCH_SIZE):
        if batch:
            batches.append(("core", batch))
    for batch in chunked(extra_docs, EXTRA_BATCH_SIZE):
        if batch:
            batches.append(("extra", batch))

    if not batches:
        print("没有待处理 batch，直接返回已有结果。")
        return qa_ckpt

    future_to_batch = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 线程池并发提交 batch 请求
        for doc_type, batch_docs in batches:
            prompt = build_batch_prompt(batch_docs, doc_type)
            future = executor.submit(chat, prompt, 3, True)
            future_to_batch[future] = {
                "doc_type": doc_type,
                "batch_docs": batch_docs,
            }

        # 谁先返回，就先处理谁
        for future in tqdm(
            concurrent.futures.as_completed(future_to_batch),
            total=len(future_to_batch),
            desc="生成QA(batch)"
        ):
            batch_info = future_to_batch[future]
            doc_type = batch_info["doc_type"]
            batch_docs = batch_info["batch_docs"]

            try:
                raw_resp = future.result()
            except Exception as e:
                print(f"[future result error] {e}")
                raw_resp = None

            if raw_resp is None:
                batch_parsed = {
                    doc["metadata"]["unique_id"]: []
                    for doc in batch_docs
                }
            else:
                try:
                    batch_parsed = parse_batch_response(raw_resp, batch_docs, doc_type)
                except Exception as e:
                    print(f"[parse batch error] {e}")
                    batch_parsed = {
                        doc["metadata"]["unique_id"]: []
                        for doc in batch_docs
                    }

            # 当前 batch 内，每个文档只保存自己的 QA
            for doc in batch_docs:
                uid = doc["metadata"]["unique_id"]
                qa_list = batch_parsed.get(uid, [])
                single_doc_raw_resp = json.dumps(qa_list, ensure_ascii=False)

                item = {
                    "unique_id": uid,
                    "raw_resp": single_doc_raw_resp,
                }

                with file_lock:
                    # 双重检查，防止极端情况下重复写
                    if uid in processed_ids:
                        continue

                    append_result_line_atomic(qa_ckpt_filename, item)

                    qa_ckpt[uid] = item
                    processed_ids.add(uid)

                    # 每写入一个文档，就立刻保存进度
                    save_progress(progress_path, processed_ids, total_docs)

    return qa_ckpt


def main():
    core_docs = load_jsonl_docs(CORE_INPUT_PATH, expected_type="core")
    extra_docs = load_jsonl_docs(EXTRA_INPUT_PATH, expected_type="extra")

    all_docs = core_docs + extra_docs

    print(f"core docs: {len(core_docs)}")
    print(f"extra docs: {len(extra_docs)}")
    print(f"total docs: {len(all_docs)}")

    if core_docs:
        print("core sample text:", core_docs[0]["page_content"][:200])
        print("core sample unique_id:", core_docs[0]["metadata"].get("unique_id"))

    if extra_docs:
        print("extra sample text:", extra_docs[0]["page_content"][:200])
        print("extra sample unique_id:", extra_docs[0]["metadata"].get("unique_id"))

    qa_dict = gen_qa_for_docs(all_docs, QA_OUTPUT_PATH, PROGRESS_PATH)

    core_doc_cnt = sum(1 for v in qa_dict.values() if v.get("doc_type") == "core")
    extra_doc_cnt = sum(1 for v in qa_dict.values() if v.get("doc_type") == "extra")
    total_qas = sum(len(v.get("qa_list", [])) for v in qa_dict.values())

    print("生成完成")
    print("已保存QA文档数：", len(qa_dict))
    print("core文档数：", core_doc_cnt)
    print("extra文档数：", extra_doc_cnt)
    print("总QA条数：", total_qas)
    print("结果保存路径：", QA_OUTPUT_PATH)
    print("进度保存路径：", PROGRESS_PATH)


if __name__ == "__main__":
    main()