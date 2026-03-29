"""
对已有 qa_pair.jsonl 里的 QA 进行质量打分。

输入：
- /root/autodl-tmp/IR-RAG-System/data/qa_pairs/qa_pair.jsonl

输出：
- /root/autodl-tmp/IR-RAG-System/data/qa_pairs/qa_score.jsonl
- /root/autodl-tmp/IR-RAG-System/data/qa_pairs/progress/qa_score_progress.json

输出格式：
{
  "unique_id": "...",
  "raw_resp": "...",
  "score": [5, 4]
}

说明：
1. score 与 raw_resp 中 QA 顺序一一对应；
2. 为保证“绝对分数”，默认不做 batch 评分；
3. 仍然使用线程池并发 + 断点续跑 + 原子写盘。
4. 单次调用失败不会写入结果；连续失败达到阈值后会终止，并回滚这段失败状态。
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
LLM_MAX_RETRY = 3
MAX_CONSECUTIVE_LLM_FAILURES = 5


class FatalLLMError(RuntimeError):
    pass

QA_INPUT_PATH = "/root/autodl-tmp/IR-RAG-System/data/qa_pairs/qa_pair.jsonl"
SCORE_OUTPUT_PATH = "/root/autodl-tmp/IR-RAG-System/data/qa_pairs/qa_score.jsonl"
PROGRESS_PATH = "/root/autodl-tmp/IR-RAG-System/data/qa_pairs/progress/qa_score_progress.json"

llm_client = OpenAI(
    api_key=os.environ["DOUBAO_API_KEY"],
    base_url=os.environ["DOUBAO_BASE_URL"]
)

# =========================
# Prompt Template
# =========================

QA_QUALITY_PROMPT_TPL = """
你是一名信息检索（Information Retrieval, IR）与搜索系统方向的严格评审员。

我会给你一组由 IR 教材文本生成的问题-回答对（QA）。
你的任务是：对每一个 QA 独立打分，评估它是否适合作为高质量训练数据。

请注意：
1. 必须逐条独立评分，不能把同一组里的 QA 相互比较；
2. 必须使用“绝对评分标准”，不要因为当前样本整体较好或较差而改变标准；
3. 问题和答案可能是中文，也可能是英文；
4. 你评估的是“这个 QA 本身的质量”，不是原教材内容本身的质量。

【评分标准（绝对分数）】

5分（高质量）
- 问题明确、自然、信息完整；
- 问题针对稳定知识点，如定义、原理、机制、对比、作用、条件、评价指标等；
- 答案准确、完整、可脱离原文独立理解；
- 问答强对应，无明显冗余，无上下文依赖；
- 没有“文中提到”“本段说明”“见第X章”等表述。

4分（较好）
- 问题和答案总体正确且可用；
- 可能存在轻微问题，例如：
  - 问法略机械；
  - 答案略冗长或略不够凝练；
  - 信息覆盖基本完整，但表达还不够理想；
- 仍然适合作为训练数据。

3分（一般）
- QA 基本相关，但质量普通；
- 常见问题包括：
  - 问题较直白、较机械，像原句改写；
  - 答案部分正确，但不够完整；
  - 问题价值一般，知识点不够集中；
- 勉强可用，但不算优质。

2分（较差）
- QA 存在明显缺陷；
- 常见问题包括：
  - 问题过于空泛，如“这段讲了什么”；
  - 问题依赖局部上下文、章节、图表；
  - 答案与问题对应不紧，或只回答了一部分；
  - 答案中出现“见第X章”“文中提到”“上文说过”等依赖上下文的话；
- 不适合作为高质量训练数据。

1分（很差）
- QA 明显不合格；
- 常见情况包括：
  - 问题本身无训练价值；
  - 问题与答案不匹配；
  - 答案明显错误、空泛、无关；
  - 问题是图表依赖型，离开图表无法回答；
  - 问题本质是在做“总结这段”“这段主要讲什么”这类低级任务。

【额外扣分信号】
以下情况应显著降分：
1. 问题是“这段讲了什么/文本描述了什么/本章讲了什么”这类摘要型问题；
2. 问题依赖图、表、章节、页码、上下文；
3. 答案出现“文中提到”“本段指出”“见第3章”“如上所述”等；
4. 问题与答案语言风格不协调或答非所问；
5. 问题过于琐碎，不像一个稳定知识点；
6. 答案只是重复问题，或信息增量很低。

【输入格式】
我会给你一个 QA 列表，每个元素都包含：
- question
- answer

【输出要求】
1. 只输出 JSON，不要输出任何额外说明；
2. 必须按输入顺序逐条返回评分结果；
3. 返回结果数量必须与输入 QA 数量一致；
4. 使用如下格式：

{
  "results": [5, 3]
}

下面是待评分的 QA：
{{qa_list}}

请只返回 JSON：
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
    return (
        "AllocationQuota.FreeTierOnly" in message
        or "free tier" in message.lower()
        or "quota" in message.lower()
    )


def chat(prompt: str,
         max_retry: int = LLM_MAX_RETRY,
         debug: bool = False,
         temperature: float = 0.0,
         top_p: float = 1.0):
    """
    为了得到更稳定的绝对分数，这里固定 temperature=0.0。
    对于明确的配额类 fatal 错误，不再做无意义重试。
    """
    remain = max_retry
    while remain > 0:
        try:
            resp = llm_client.chat.completions.create(
                model=os.environ["DOUBAO_MODEL_NAME"],
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个严谨的人工智能助手，擅长对信息检索领域问答数据做稳定、一致、绝对标准的质量评分。"
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                top_p=top_p,
                extra_body={"enable_thinking": False},
            )
            return resp.choices[0].message.content
        except Exception as e:
            if is_fatal_llm_error(e):
                if debug:
                    print(f"[chat fatal] {e}")
                raise FatalLLMError(str(e)) from e

            remain -= 1
            sleep_seconds = random.randint(1, 4)
            if debug:
                print(f"[chat error] {e}; remain_retry={remain}; sleep={sleep_seconds}s")
            if remain > 0:
                time.sleep(sleep_seconds)

    return None


def append_result_line_atomic(output_path: str, item: dict):
    """
    以追加方式写入一行 JSONL
    """
    ensure_parent_dir(output_path)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())


def rewrite_result_file_without_uids(output_path: str, remove_uids):
    """
    从结果文件中删除指定 unique_id 对应的记录。
    """
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


def rollback_failed_progress(output_path: str, progress_path: str, failed_uids, total_docs: int):
    """
    删除失败阶段对应结果，并按当前结果文件重建断点。
    """
    rewrite_result_file_without_uids(output_path, failed_uids)
    processed_ids, _ = load_existing_results(output_path)
    save_progress(progress_path, processed_ids, total_docs)


# =========================
# Input Loader
# =========================

def load_qa_docs(path: str):
    """
    读取 qa_pair.jsonl
    每一行保留：
    - unique_id
    - raw_resp 原始字符串
    - qa_list 解析后的列表
    """
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

            uid = item.get("unique_id")
            raw_resp = item.get("raw_resp", "")

            if not uid:
                print(f"[skip] {path} line {line_no}: missing unique_id")
                continue

            try:
                qa_list = json.loads(raw_resp)
            except Exception as e:
                print(f"[skip] {path} line {line_no}: raw_resp parse failed: {e}")
                continue

            if not isinstance(qa_list, list):
                continue

            docs.append({
                "unique_id": uid,
                "raw_resp": raw_resp,
                "qa_list": qa_list,
            })

    return docs

# =========================
# Prompt Builder
# =========================

def build_score_prompt(qa_list):
    qa_text = json.dumps(qa_list, ensure_ascii=False, separators=(",", ":"))
    return QA_QUALITY_PROMPT_TPL.replace("{{qa_list}}", qa_text).strip()

# =========================
# Validation
# =========================

def validate_score_item(item):
    if isinstance(item, dict):
        item = item.get("score", 1)

    try:
        score = int(item)
    except Exception:
        score = 1

    if score < 1:
        score = 1
    if score > 5:
        score = 5

    return score


def validate_score_list(parsed, qa_list):
    """
    结果必须与 qa_list 顺序和长度对齐。
    若模型返回数量不足，则缺项补 1 分。
    """
    results = parsed.get("results", [])
    if not isinstance(results, list):
        results = []

    cleaned = [validate_score_item(x) for x in results]

    if len(cleaned) < len(qa_list):
        cleaned.extend([1] * (len(qa_list) - len(cleaned)))

    return cleaned[:len(qa_list)]


def parse_score_response(raw_resp, qa_list):
    parsed = extract_json_from_text(raw_resp)
    if not isinstance(parsed, dict):
        return [1] * len(qa_list)

    return validate_score_list(parsed, qa_list)

# =========================
# Checkpoint / Resume
# =========================

def load_existing_results(result_path: str):
    """
    加载已经写入结果文件的内容：
    - processed_ids: 已完成 unique_id 集合
    - score_ckpt: unique_id -> item
    """
    processed_ids = set()
    score_ckpt = {}

    if not os.path.exists(result_path):
        return processed_ids, score_ckpt

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
            score_ckpt[uid] = item

    return processed_ids, score_ckpt


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

# =========================
# Scoring
# =========================

class ConsecutiveLLMFailureError(RuntimeError):
    pass


def score_one_doc(doc):
    """
    单个 unique_id 对应的一组 QA 单独评分。
    这样更利于保持绝对评分标准，避免 batch 内部相互参照。
    """
    uid = doc["unique_id"]
    qa_list = doc["qa_list"]

    # 空 QA 直接给空分数，不调用大模型
    if not qa_list:
        return uid, "success", []

    prompt = build_score_prompt(qa_list)
    try:
        raw_resp = chat(prompt, max_retry=LLM_MAX_RETRY, debug=True, temperature=0.0, top_p=1.0)
    except FatalLLMError as e:
        print(f"[fatal llm error] uid={uid}; error={e}")
        return uid, "fatal", None

    if raw_resp is None:
        return uid, "failed", None

    try:
        score_list = parse_score_response(raw_resp, qa_list)
    except Exception as e:
        print(f"[parse score error] uid={uid}, error={e}")
        return uid, "failed", None

    return uid, "success", score_list


def gen_scores_for_docs(docs, output_path, progress_path):
    ensure_parent_dir(output_path)
    ensure_parent_dir(progress_path)

    file_lock = threading.Lock()
    total_docs = len(docs)

    processed_ids, score_ckpt = load_existing_results(output_path)
    print(f"已存在评分结果数：{len(processed_ids)} / {total_docs}")

    remaining_docs = [
        doc for doc in docs
        if doc["unique_id"] not in processed_ids
    ]
    print(f"待处理剩余文档数：{len(remaining_docs)}")

    save_progress(progress_path, processed_ids, total_docs)

    if not remaining_docs:
        print("没有待处理文档，直接返回已有结果。")
        return score_ckpt

    consecutive_failed_uids = []
    abort_now = False
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS)
    pbar = tqdm(total=len(remaining_docs), desc="QA评分", dynamic_ncols=True)
    try:
        for doc, result in zip(remaining_docs, executor.map(score_one_doc, remaining_docs)):
            pbar.update(1)
            uid = doc["unique_id"]

            try:
                result_uid, status, score_list = result
            except Exception as e:
                print(f"[future result error] uid={uid}, error={e}")
                result_uid, status, score_list = uid, "failed", None

            if status == "fatal":
                abort_now = True
                rollback_failed_progress(
                    output_path=output_path,
                    progress_path=progress_path,
                    failed_uids=consecutive_failed_uids,
                    total_docs=total_docs,
                )
                executor.shutdown(wait=False, cancel_futures=True)
                raise ConsecutiveLLMFailureError(
                    "检测到不可恢复的配额/权限错误，已立即终止本次评分，并回滚未完成阶段的断点。"
                )

            if status != "success":
                consecutive_failed_uids.append(result_uid)
                print(
                    f"[llm failure] uid={result_uid}; consecutive_failed="
                    f"{len(consecutive_failed_uids)}/{MAX_CONSECUTIVE_LLM_FAILURES}"
                )
                if len(consecutive_failed_uids) >= MAX_CONSECUTIVE_LLM_FAILURES:
                    abort_now = True
                    rollback_failed_progress(
                        output_path=output_path,
                        progress_path=progress_path,
                        failed_uids=consecutive_failed_uids,
                        total_docs=total_docs,
                    )
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise ConsecutiveLLMFailureError(
                        "连续 LLM 调用失败达到阈值，已终止本次评分，并回滚这段失败数据与断点。"
                    )

                pbar.set_postfix({
                    "done": len(processed_ids),
                    "remain": max(total_docs - len(processed_ids), 0),
                    "fail_streak": len(consecutive_failed_uids),
                })
                continue

            consecutive_failed_uids.clear()
            item = {
                "unique_id": result_uid,
                "raw_resp": doc["raw_resp"],
                "score": score_list,
            }

            with file_lock:
                if result_uid in processed_ids:
                    continue

                append_result_line_atomic(output_path, item)
                score_ckpt[result_uid] = item
                processed_ids.add(result_uid)
                save_progress(progress_path, processed_ids, total_docs)

            pbar.set_postfix({
                "done": len(processed_ids),
                "remain": max(total_docs - len(processed_ids), 0),
                "fail_streak": 0,
            })
    finally:
        pbar.close()
        if not abort_now:
            executor.shutdown(wait=True, cancel_futures=False)

    return score_ckpt

# =========================
# Main
# =========================

def main():
    docs = load_qa_docs(QA_INPUT_PATH)

    print(f"total docs: {len(docs)}")
    if docs:
        print("sample unique_id:", docs[0]["unique_id"])
        print("sample raw_resp:", docs[0]["raw_resp"][:200])

    try:
        score_dict = gen_scores_for_docs(
            docs=docs,
            output_path=SCORE_OUTPUT_PATH,
            progress_path=PROGRESS_PATH
        )
    except ConsecutiveLLMFailureError as e:
        print(str(e))
        print("结果保存路径：", SCORE_OUTPUT_PATH)
        print("进度保存路径：", PROGRESS_PATH)
        return

    total_scored_qa = 0
    score_counter = {k: 0 for k in range(1, 6)}
    for v in score_dict.values():
        score_list = v.get("score", [])
        if isinstance(score_list, list):
            total_scored_qa += len(score_list)
            for score in score_list:
                try:
                    score = int(score)
                except Exception:
                    score = 1
                if score < 1:
                    score = 1
                if score > 5:
                    score = 5
                score_counter[score] += 1

    print("评分完成")
    print("已保存文档数：", len(score_dict))
    print("已评分QA总数：", total_scored_qa)
    for score in range(1, 6):
        print(f"{score}分QA数：", score_counter[score])
    print("结果保存路径：", SCORE_OUTPUT_PATH)
    print("进度保存路径：", PROGRESS_PATH)


if __name__ == "__main__":
    main()