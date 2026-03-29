import argparse
import hashlib
import json
import os
import random
import re
import time
from typing import Any, Dict, Iterable, List, Tuple

from openai import OpenAI
from tqdm.auto import tqdm

from langchain_core.documents import Document
from src.path import final_split_docs
from src.retriever.bm25_retriever import BM25
from src.retriever.milvus_retriever import MilvusRetriever
from src.utils import merge_docs

random.seed(42)

DEFAULT_TRAIN_MASTER_PATH = "/root/autodl-tmp/IR-RAG-System/data/qa_pairs/train_augmented_master.jsonl"
DEFAULT_TEST_MASTER_PATH = "/root/autodl-tmp/IR-RAG-System/data/qa_pairs/test_augmented_master.jsonl"
DEFAULT_TRAIN_OUTPUT_PATH = "/root/autodl-tmp/IR-RAG-System/data/qa_pairs/reranker/train_rank_labels.jsonl"
DEFAULT_VAL_OUTPUT_PATH = "/root/autodl-tmp/IR-RAG-System/data/qa_pairs/reranker/val_rank_labels.jsonl"
DEFAULT_TEST_OUTPUT_PATH = "/root/autodl-tmp/IR-RAG-System/data/qa_pairs/reranker/test_rank_labels.jsonl"
DEFAULT_TRAIN_PROGRESS_PATH = "/root/autodl-tmp/IR-RAG-System/data/qa_pairs/progress/train_rank_progress.json"
DEFAULT_VAL_PROGRESS_PATH = "/root/autodl-tmp/IR-RAG-System/data/qa_pairs/progress/val_rank_progress.json"
DEFAULT_TEST_PROGRESS_PATH = "/root/autodl-tmp/IR-RAG-System/data/qa_pairs/progress/test_rank_progress.json"
DEFAULT_FINAL_DOCS_PATH = final_split_docs
DEFAULT_TRAIN_QA_PATH = "/root/autodl-tmp/IR-RAG-System/data/qa_pairs/train_qa_pair.json"
DEFAULT_TEST_QA_PATH = "/root/autodl-tmp/IR-RAG-System/data/qa_pairs/test_qa_pair.json"
DEFAULT_QA_GENERALIZED_PATH = "/root/autodl-tmp/IR-RAG-System/data/qa_pairs/qa_generalized.jsonl"
DEFAULT_BM25_TOPK = 10
DEFAULT_MILVUS_TOPK = 10
DEFAULT_MAX_LLM_DOC_CHARS = 1600
DEFAULT_MAX_RETRY = 3
DEFAULT_VAL_RATIO = 0.1


SYSTEM_PROMPT = (
    "你是一名严格的信息检索相关性评估专家。"
    "给定一个查询以及若干检索得到的文档块，"
    "请为每个候选文档分配一个标签：1（强相关）、0（相关）、-1（不相关）。"
    "请对标签1保持非常保守的使用：只有当某个文档块本身就构成“金标准证据”，"
    "能够直接且充分地回答问题时，才可以标为1，而不是仅仅因为它在主题上相关或有一定帮助。"
    "你的输出必须只包含标签结果，不要输出解释。"
)


LLM_PROMPT_TPL = """
你需要判断每个被检索到的文档块，对于回答一个信息检索问题是否有用。

标签说明：
- 1 = 强相关
  只有当该文档块本身就几乎等同于“金标准证据块”时，才能标为 1。
  也就是说，该块单独就直接、充分、明确地回答问题，几乎不需要依赖其他块补充。
  如果该块只是主题相近、部分覆盖、提供背景、提供辅助论据，哪怕明显相关，也不要标为 1，而应优先标为 0。
  对 1 的判定必须非常保守。
- 0 = 相关
  该文档块在主题上相关，并且对回答问题有帮助，但只提供部分、间接、辅助或不够完整的信息。
  只要你对是否达到“金标准证据块”有任何犹豫，就应标为 0 而不是 1。
- -1 = 不相关
  该文档块无法帮助回答问题，或仅仅是表面相关。

判断原则：
1. 以 Query（问题）为主要判断依据。
2. 只根据 Query 与候选文档文本本身判断，不要假设任何额外上下文。
3. 标签 1 必须极其严格：只有当文档块单独即可直接、充分地回答问题时，才允许标为 1。
4. 不要因为文档块和问题属于同一主题、同一章节、同一页，或包含部分关键词，就标为 1。
5. 如果一个文档块只是背景说明、同义改写、局部补充、例子、延伸描述，通常应标为 0，而不是 1。
6. 不要强行把所有候选都标为正样本。
7. 必须按顺序为每个候选返回标签。

问题（Query）：
{query}

候选文档（Candidates）：
{candidates}

请仅返回如下 JSON 格式：
{{
  "labels": [0, -1]
}}
""".strip()


class FatalLLMError(RuntimeError):
    pass


def normalize_text(text: str) -> str:
    """统一整理空白字符，便于后续文本比较、去重和构造提示词。"""
    return re.sub(r"\s+", " ", str(text or "")).strip()


def ensure_parent_dir(path: str):
    """在目标文件的父目录不存在时，先自动创建该目录。"""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def md5_text(text: str) -> str:
    """为兜底 ID 或去重键生成稳定的 md5 哈希值。"""
    return hashlib.md5((text or "").encode("utf-8")).hexdigest()


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """从大模型原始输出中提取第一个可解析的 JSON 对象或数组。"""
    text = (text or "").strip()
    if not text:
        raise ValueError("empty response")

    fenced = re.search(r"```(?:json)?\s*([\[{].*?[\]}])\s*```", text, flags=re.S | re.I)
    if fenced:
        return json.loads(fenced.group(1))

    try:
        return json.loads(text)
    except Exception:
        pass

    candidates = re.findall(r"[\[{].*?[\]}]", text, flags=re.S)
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            continue

    raise ValueError("failed to parse json")

#提取异常信息
def is_fatal_llm_error(exc: Exception) -> bool:
    """识别配额类大模型异常，这类错误通常应直接中止整次生成任务。"""
    message = str(exc).lower()
    return "quota" in message or "allocationquota" in message or "free tier" in message

#根据问题获得父块uid的函数
def build_item_group_key(
    item: Dict[str, Any],
    question_uid_to_question: Dict[str, str],
    question_to_source_doc_uid: Dict[str, str],
    uid_to_doc: Dict[str, Document],
) -> str:
    """
    将样本回溯到来源的检索块,并返回该检索块的父块的uid
    输入:
    item 表示train_augmented_master.jsonl(或者test版)的一个样本
    question_uid_to_question 表示问题文本到原始文档块映射
    question_to_source_doc_uid 表示文档块到对应document映射
    """
    source_uid = normalize_text(item.get("source_uid", ""))
    if source_uid:
        try:
            source_doc_uid = resolve_source_doc_uid(
                source_uid=source_uid,
                question_uid_to_question=question_uid_to_question,
                question_to_source_doc_uid=question_to_source_doc_uid,
            )
            source_parent_uid, _ = resolve_parent_doc(source_doc_uid=source_doc_uid, uid_to_doc=uid_to_doc)
            return normalize_text(source_parent_uid) or normalize_text(source_doc_uid) or source_uid
        except Exception:
            pass

    return normalize_text(item.get("id", "")) or md5_text(json.dumps(item, ensure_ascii=False, sort_keys=True))


#切分训练集和验证集
def split_rows_by_group_balance(
    rows: List[Dict[str, Any]],
    val_ratio: float,
    seed: int,
    question_uid_to_question: Dict[str, str],
    question_to_source_doc_uid: Dict[str, str],
    uid_to_doc: Dict[str, Document],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """按原始父块分组切分样本，并用多次随机采样寻找更公平的验证集划分。
    
    增强点：
    - 增加“小组扎堆惩罚”
    - 若候选验证集中的小组占比明显高于全局小组占比，则提高 score
    """

    SMALL_GROUP_THRESHOLD = 12
    SMALL_GROUP_PENALTY_WEIGHT = 2.0

    # 按来源父块 uid 分组，保证同一父块衍生的问题不会同时落入训练集和验证集。
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in rows:
        group_key = build_item_group_key(
            item=item,
            question_uid_to_question=question_uid_to_question,
            question_to_source_doc_uid=question_to_source_doc_uid,
            uid_to_doc=uid_to_doc,
        )
        grouped.setdefault(group_key, []).append(item)

    # 计算目标验证集样本数
    total_rows = len(rows)
    target_val_rows = int(round(total_rows * max(0.0, min(1.0, val_ratio))))
    if total_rows > 1 and val_ratio > 0 and target_val_rows == 0:
        target_val_rows = 1
    if target_val_rows >= total_rows:
        target_val_rows = max(1, total_rows - 1)

    # 处理空分组情况
    group_items = list(grouped.items())
    if not group_items:
        return [], [], {
            "group_count": 0,
            "train_group_count": 0,
            "val_group_count": 0,
            "target_val_rows": target_val_rows,
            "actual_train_rows": 0,
            "actual_val_rows": 0,
            "actual_val_ratio": 0.0,
            "best_score": 0.0,
            "search_rounds": 0,
            "small_group_threshold": SMALL_GROUP_THRESHOLD,
            "global_small_group_ratio": 0.0,
            "actual_val_small_group_ratio": 0.0,
        }

    # 计算目标验证组数
    group_count = len(group_items)
    target_val_groups = max(1, round(group_count * val_ratio)) if val_ratio > 0 else 0
    if target_val_groups >= group_count and group_count > 1:
        target_val_groups = group_count - 1

    # 预先统计每个 group 的大小
    group_sizes: Dict[str, int] = {
        group_key: len(items) for group_key, items in group_items
    }

    # 全局小组统计
    total_small_group_count = sum(
        1 for size in group_sizes.values() if size <= SMALL_GROUP_THRESHOLD
    )
    global_small_group_ratio = (
        total_small_group_count / group_count if group_count > 0 else 0.0
    )

    # 决定搜索次数
    search_rounds = min(3000, max(300, group_count * 30))
    best_val_group_keys = set()
    best_score = None
    best_ratio_penalty = 0.0
    best_candidate_small_group_ratio = 0.0

    # 多次随机打乱分组顺序，每轮都构造一个候选验证集，最后选最接近目标比例的方案。
    for round_idx in range(search_rounds):
        rng = random.Random(seed + round_idx)
        shuffled_group_items = list(group_items)
        rng.shuffle(shuffled_group_items)

        # 初始化候选验证集
        candidate_keys = set()
        candidate_rows = 0
        candidate_group_count = 0

        for group_key, items in shuffled_group_items:
            group_size = len(items)
            if candidate_rows >= target_val_rows and candidate_group_count >= target_val_groups:
                break

            # 计算“取”和“不取”的样本数差和组数差
            diff_if_take = abs(target_val_rows - (candidate_rows + group_size))
            diff_if_skip = abs(target_val_rows - candidate_rows)
            group_diff_if_take = abs(target_val_groups - (candidate_group_count + 1))
            group_diff_if_skip = abs(target_val_groups - candidate_group_count)

            should_take = False
            if candidate_rows == 0 and target_val_rows > 0:
                should_take = True
            elif diff_if_take < diff_if_skip:
                should_take = True
            elif diff_if_take == diff_if_skip and group_diff_if_take <= group_diff_if_skip:
                should_take = True
            elif candidate_group_count < target_val_groups and rng.random() < 0.35:
                should_take = True

            if should_take:
                candidate_keys.add(group_key)
                candidate_rows += group_size
                candidate_group_count += 1

        # 基础偏差
        row_diff = abs(candidate_rows - target_val_rows)
        group_diff = abs(candidate_group_count - target_val_groups)

        # 候选验证集中的小组占比
        candidate_small_group_count = sum(
            1 for key in candidate_keys if group_sizes[key] <= SMALL_GROUP_THRESHOLD
        )
        candidate_small_group_ratio = (
            candidate_small_group_count / candidate_group_count
            if candidate_group_count > 0 else 0.0
        )

        # 小组扎堆惩罚：只在候选验证集小组占比高于全局占比时惩罚
        ratio_penalty = max(
            0.0,
            candidate_small_group_ratio - global_small_group_ratio
        )

        # 最终 score
        score = (
            row_diff
            + group_diff * 0.5
            + ratio_penalty * SMALL_GROUP_PENALTY_WEIGHT
        )

        if best_score is None or score < best_score:
            best_score = score
            best_val_group_keys = candidate_keys
            best_ratio_penalty = ratio_penalty
            best_candidate_small_group_ratio = candidate_small_group_ratio

            # 只有在比例和结构都很好时才提前结束
            if row_diff == 0 and group_diff == 0 and ratio_penalty == 0:
                break

    train_rows: List[Dict[str, Any]] = []
    val_rows: List[Dict[str, Any]] = []
    for group_key, items in grouped.items():
        if group_key in best_val_group_keys:
            val_rows.extend(items)
        else:
            train_rows.extend(items)

    actual_val_group_count = len(best_val_group_keys)
    actual_val_small_group_count = sum(
        1 for key in best_val_group_keys if group_sizes[key] <= SMALL_GROUP_THRESHOLD
    )
    actual_val_small_group_ratio = (
        actual_val_small_group_count / actual_val_group_count
        if actual_val_group_count > 0 else 0.0
    )

    stats = {
        "group_count": len(grouped),
        "train_group_count": len(grouped) - len(best_val_group_keys),
        "val_group_count": len(best_val_group_keys),
        "target_val_rows": target_val_rows,
        "target_val_groups": target_val_groups,
        "actual_train_rows": len(train_rows),
        "actual_val_rows": len(val_rows),
        "actual_val_ratio": (len(val_rows) / total_rows) if total_rows else 0.0,
        "best_score": best_score if best_score is not None else 0.0,
        "search_rounds": search_rounds,
        "small_group_threshold": SMALL_GROUP_THRESHOLD,
        "small_group_penalty_weight": SMALL_GROUP_PENALTY_WEIGHT,
        "global_small_group_ratio": global_small_group_ratio,
        "actual_val_small_group_ratio": actual_val_small_group_ratio,
        "best_ratio_penalty": best_ratio_penalty,
        "best_candidate_small_group_ratio": best_candidate_small_group_ratio,
    }
    return train_rows, val_rows, stats

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """读取 jsonl 文件到内存，并在遇到坏行时跳过并打印日志。"""
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except Exception as exc:
                print(f"[load error] {path} line={line_no}: {exc}")
    return rows


def append_jsonl(path: str, rows: Iterable[Dict[str, Any]]):
    """将结果追加写入 jsonl 文件，并及时刷盘以支持断点续跑。"""
    ensure_parent_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())


def save_progress(path: str, processed_query_ids: List[str], stats: Dict[str, Any]):
    """保存当前 split 的处理进度，便于中断后安全恢复。"""
    ensure_parent_dir(path)
    payload = {
        "processed_query_ids": processed_query_ids,
        "stats": stats,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
    }
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def load_processed_query_ids(progress_path: str) -> set:
    """从进度文件中读取已经处理过的 query ID。"""
    if not os.path.exists(progress_path):
        return set()

    try:
        with open(progress_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return set()

    processed_query_ids = payload.get("processed_query_ids", [])
    return {str(query_id) for query_id in processed_query_ids if str(query_id).strip()}


def load_processed_query_ids_from_output(output_path: str, rows: List[Dict[str, Any]]) -> set:
    """
    从已有输出文件中恢复已经成功落盘的 query ID。

    优先读取新格式中的 query_id；若是旧格式输出没有 query_id，
    则退化为按 query 文本匹配当前 split 中等价的样本 ID。
    这样无论输出来自旧逻辑还是新逻辑，都能避免续跑时重复生成。
    """
    if not os.path.exists(output_path):
        return set()

    query_to_ids: Dict[str, set] = {}
    for item in rows:
        query_id = str(item.get("id", "")).strip()
        if not query_id:
            continue
        query_text = parse_retrieval_query(item)
        if not query_text:
            continue
        query_to_ids.setdefault(query_text, set()).add(query_id)

    processed_query_ids = set()
    with open(output_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except Exception as exc:
                print(f"[load error] {output_path} line={line_no}: {exc}")
                continue

            query_ids = row.get("query_ids")
            if isinstance(query_ids, list):
                normalized_query_ids = {
                    str(query_id).strip() for query_id in query_ids if str(query_id).strip()
                }
                if normalized_query_ids:
                    processed_query_ids.update(normalized_query_ids)
                    continue

            query_id = str(row.get("query_id", "")).strip()
            if query_id:
                processed_query_ids.add(query_id)
                continue

            query_text = normalize_text(row.get("query", ""))
            if query_text:
                processed_query_ids.update(query_to_ids.get(query_text, set()))

    return processed_query_ids


def load_final_docs_by_uid(path: str) -> Dict[str, Document]:
    
    uid_to_doc: Dict[str, Document] = {}
    for item in load_jsonl(path):
        metadata = item.get("metadata") or {}
        uid = metadata.get("unique_id")
        if not uid:
            continue
        uid_to_doc[str(uid)] = Document(
            page_content=item.get("page_content", ""),
            metadata=metadata,
        )
    return uid_to_doc


def load_qa_question_map(paths: List[str]) -> Dict[str, str]:
    """利用train_qa的unque_id和master的source_id一致的方式,关联master的qa和train的qa"""
    question_map: Dict[str, str] = {}
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            question_uid = normalize_text(item.get("unique_id", ""))
            question = normalize_text(item.get("question", ""))
            if question_uid and question:
                question_map[question_uid] = question
    return question_map


def parse_numbered_questions(raw_resp: str) -> List[str]:
    """从 qa_generalized 结果中解析出按编号保存的泛化问题。"""
    questions = []
    for line in (raw_resp or "").splitlines():
        cleaned = normalize_text(re.sub(r"^\s*\d+\.\s*", "", line))
        if cleaned:
            questions.append(cleaned)
    return questions


def load_question_to_source_doc_map(path: str) -> Dict[str, str]:
    """
    train_pair的q就是qa_generalized的q,然后train_pair的q就是qa_generalized可以得到原始文档#生成的第几个q
    """
    question_to_doc_uid: Dict[str, str] = {}

    for item in load_jsonl(path):
        source_uid = normalize_text(item.get("source_uid", ""))
        if not source_uid or "#" not in source_uid:
            continue

        source_doc_uid = source_uid.split("#", 1)[0]
        original_question = normalize_text(item.get("unique_id", ""))
        if original_question:
            question_to_doc_uid[original_question] = source_doc_uid

        for generalized_question in parse_numbered_questions(item.get("raw_resp", "")):
            question_to_doc_uid[generalized_question] = source_doc_uid

    return question_to_doc_uid


def resolve_source_doc_uid(
    source_uid: str,
    question_uid_to_question: Dict[str, str],
    question_to_source_doc_uid: Dict[str, str],
) -> str:
    """
    这里具体思路就是利用source_uid得到最初泛化后的问题文本
    然后利用泛化后问题文本得到原始文档uid
    """
    #利用train_augmented_master.jsonl的source_uid,去匹配train_qa_pair的question
    question = question_uid_to_question.get(source_uid)
    if not question:
        raise KeyError(f"source_uid not found in train_qa_pair.json: {source_uid}")
    #利用question去匹配原始文档的uid
    source_doc_uid = question_to_source_doc_uid.get(question)
    if not source_doc_uid:
        raise KeyError(f"question not found in qa_generalized.jsonl: {question}")

    return source_doc_uid


def resolve_parent_doc(source_doc_uid: str, uid_to_doc: Dict[str, Document]) -> Tuple[str, Document]:
    """
    将来源子块提升为父块，保证监督信号与当前检索粒度一致。
    输入:
        source_doc_uid 表示原始文档的unique_id
        uid_to_doc 是原始文档uid 到 原始文档的映射

    """
    source_doc = uid_to_doc.get(source_doc_uid)
    if source_doc is None:
        raise KeyError(f"source_doc_uid not found in final_split_docs: {source_doc_uid}")

    source_meta = source_doc.metadata or {}
    #获得对应的父uid
    parent_uid = source_meta.get("parent_id") or source_meta.get("unique_id") or source_doc_uid
    parent_doc = uid_to_doc.get(parent_uid, source_doc)
    return parent_uid, parent_doc


def parse_retrieval_query(item: Dict[str, Any]) -> str:
    """
    从增强样本中抽取真正用于检索的查询文本。
    这里额外处理是因为构造的问题中,存在一种问题是例如：
    {Question: xxx\nKeywords: xxx}
    对于这种问题,如果加上关键字，检索器会被关键字作弊，污染语义
    """
    raw_input = normalize_text(item.get("input", ""))
    task_type = item.get("task_type", "")

    if task_type == "qa_with_keywords":
        patterns = [
            r"^Question:\s*(.*?)\s*Keywords:\s*.*$",
            r"^问题：\s*(.*?)\s*关键词：\s*.*$",
        ]
        for pattern in patterns:
            match = re.match(pattern, raw_input, flags=re.S)
            if match:
                return normalize_text(match.group(1))

    return raw_input


def get_doc_uid(doc: Document) -> str:
    """获取稳定的文档标识；若缺少显式 ID，则退化为基于内容的哈希。"""
    metadata = doc.metadata or {}
    uid = metadata.get("unique_id")
    if uid:
        return str(uid)

    fallback = "||".join([
        str(metadata.get("source", "")),
        str(metadata.get("page_no", "")),
        str(metadata.get("orig_page_no", "")),
        normalize_text(doc.page_content)[:1000],
    ])
    return md5_text(fallback)

def format_candidates_for_prompt(candidates: List[Document], max_chars: int) -> str:
    """将候选文档整理成紧凑的提示词格式，供判标模型使用。"""
    blocks = []
    for idx, doc in enumerate(candidates, start=1):
        text = normalize_text(doc.page_content)[:max_chars]
        blocks.append(
            "\n".join([
                f"[{idx}]",
                f"text: {text}",
            ])
        )
    return "\n\n".join(blocks)

def make_llm_client() -> Tuple[OpenAI, str]:
    """创建兼容 OpenAI 的客户端，并解析排序标注模型配置。"""
    api_key = os.environ.get("RANK_API_KEY") or os.environ.get("DOUBAO_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("RANK_BASE_URL") or os.environ.get("DOUBAO_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    model_name = os.environ.get("RANK_MODEL_NAME") or os.environ.get("DOUBAO_MODEL_NAME") or os.environ.get("OPENAI_MODEL_NAME")

    if not api_key:
        raise RuntimeError("missing API key: set RANK_API_KEY / DOUBAO_API_KEY / OPENAI_API_KEY")
    if not base_url:
        raise RuntimeError("missing base_url: set RANK_BASE_URL / DOUBAO_BASE_URL / OPENAI_BASE_URL")
    if not model_name:
        raise RuntimeError("missing model name: set RANK_MODEL_NAME / DOUBAO_MODEL_NAME / OPENAI_MODEL_NAME")

    return OpenAI(api_key=api_key, base_url=base_url), model_name

def judge_candidates_with_llm(
    llm_client: OpenAI,
    model_name: str,
    item: Dict[str, Any],
    query_text: str,
    candidates: List[Document],
    max_doc_chars: int,
    max_retry: int,
) -> List[int]:
    """调用判标模型，为候选文档打相关性标签。"""
    prompt = LLM_PROMPT_TPL.format(
        query=query_text,
        candidates=format_candidates_for_prompt(candidates, max_chars=max_doc_chars),
    )

    remain = max_retry
    while remain > 0:
        try:
            resp = llm_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                top_p=1.0,
                # extra_body={"enable_thinking": False},
            )
            content = resp.choices[0].message.content
            payload = extract_json_from_text(content)
            labels = payload.get("labels")
            if not isinstance(labels, list):
                raise ValueError("labels is not a list")
            if len(labels) != len(candidates):
                raise ValueError(f"label length mismatch: {len(labels)} vs {len(candidates)}")

            normalized_labels = []
            for label in labels:
                if label not in (-1, 0, 1):
                    raise ValueError(f"invalid label: {label}")
                normalized_labels.append(int(label))
            return normalized_labels
        except Exception as exc:
            if is_fatal_llm_error(exc):
                raise FatalLLMError(str(exc)) from exc
            remain -= 1
            if remain <= 0:
                raise
            time.sleep(random.randint(1, 3))

    raise RuntimeError("unreachable llm retry state")


def build_output_row(
    query_id: str,
    query_text: str,
    doc: Document,
    label: int,
) -> Dict[str, Any]:
    """将单条已标注的 query-document 对转换为最终排序器训练格式。"""
    return {
        "query_id": query_id,
        "query": query_text,
        "document": doc.page_content,
        "relevance": int(label),
    }

#这个是对一个问题的完整处理流程
def process_one_query(
    item: Dict[str, Any],
    uid_to_doc: Dict[str, Document],
    question_uid_to_question: Dict[str, str],
    question_to_source_doc_uid: Dict[str, str],
    bm25: BM25,
    milvus: MilvusRetriever,
    llm_client: OpenAI,
    model_name: str,
    bm25_topk: int,
    milvus_topk: int,
    max_doc_chars: int,
    max_retry: int,
) -> List[Dict[str, Any]]:
    """为单个问题生成完整的标注结果，包括硬正样本父块和其余候选文档。"""
    #提取问题
    query_text = parse_retrieval_query(item)
    if not query_text:
        raise ValueError(f"empty query text for item={item.get('id')}")
    
    #获得原始文档uid
    source_uid = str(item.get("source_uid", ""))
    source_doc_uid = resolve_source_doc_uid(
        source_uid=source_uid,
        question_uid_to_question=question_uid_to_question,
        question_to_source_doc_uid=question_to_source_doc_uid,
    )
    #获得父块
    source_parent_uid, strong_doc = resolve_parent_doc(
        source_doc_uid=source_doc_uid,
        uid_to_doc=uid_to_doc,
    )

    #问题来源文本，标签一定是1
    output_rows = [
        build_output_row(
            query_id=str(item.get("id", "")),
            query_text=query_text,
            doc=strong_doc,
            label=1,
        )
    ]
    
    #检索器检索合并
    bm25_docs = bm25.retrieve_topk(query_text, topk=bm25_topk)
    milvus_docs = milvus.retrieve_topk(query_text, topk=milvus_topk)
    merged_candidates = merge_docs(bm25_docs, milvus_docs)

    
    strong_uid = str((strong_doc.metadata or {}).get("unique_id", ""))
    remaining_candidates = [
        candidate for candidate in merged_candidates
        if str((candidate.metadata or {}).get("unique_id", "")) != strong_uid
    ]

    if not remaining_candidates:
        return output_rows

    judgments = judge_candidates_with_llm(
        llm_client=llm_client,
        model_name=model_name,
        item=item,
        query_text=query_text,
        candidates=remaining_candidates,
        max_doc_chars=max_doc_chars,
        max_retry=max_retry,
    )

    for candidate, label in zip(remaining_candidates, judgments):
        output_rows.append(
            build_output_row(
                query_id=str(item.get("id", "")),
                query_text=query_text,
                doc=candidate,
                label=label,
            )
        )

    return output_rows


def run_split(
    split_name: str,
    rows: List[Dict[str, Any]],
    output_path: str,
    progress_path: str,
    overwrite: bool,
    uid_to_doc: Dict[str, Document],
    question_uid_to_question: Dict[str, str],
    question_to_source_doc_uid: Dict[str, str],
    bm25: BM25,
    milvus: MilvusRetriever,
    llm_client: OpenAI,
    model_name: str,
    bm25_topk: int,
    milvus_topk: int,
    max_doc_chars: int,
    max_retry: int,
) -> Dict[str, Any]:
    """完整处理一个数据划分，写出结果，并维护可恢复的进度信息。"""
    if overwrite and os.path.exists(output_path):
        os.remove(output_path)

    processed_query_ids = set()
    if not overwrite:
        processed_query_ids = load_processed_query_ids(progress_path)
        processed_query_ids.update(load_processed_query_ids_from_output(output_path, rows))
    stats = {
        "split": split_name,
        "written_rows": 0,
        "processed_queries": 0,
        "skipped_queries": 0,
        "failed_queries": 0,
        "total_rows": len(rows),
    }
    failed_query_ids: List[str] = []

    print(f"[info] split={split_name} rows={len(rows)} output={output_path}")
    print(f"[info] split={split_name} progress={progress_path}")

    for item in tqdm(rows, desc=f"generate_rank:{split_name}"):
        query_id = str(item.get("id", ""))
        if not query_id:
            stats["skipped_queries"] += 1
            continue
        if query_id in processed_query_ids:
            stats["skipped_queries"] += 1
            continue

        try:
            output_rows = process_one_query(
                item=item,
                uid_to_doc=uid_to_doc,
                question_uid_to_question=question_uid_to_question,
                question_to_source_doc_uid=question_to_source_doc_uid,
                bm25=bm25,
                milvus=milvus,
                llm_client=llm_client,
                model_name=model_name,
                bm25_topk=bm25_topk,
                milvus_topk=milvus_topk,
                max_doc_chars=max_doc_chars,
                max_retry=max_retry,
            )
            append_jsonl(output_path, output_rows)
            processed_query_ids.add(query_id)
            stats["processed_queries"] += 1
            stats["written_rows"] += len(output_rows)
        except FatalLLMError:
            raise
        except Exception as exc:
            failed_query_ids.append(query_id)
            stats["failed_queries"] += 1
            print(f"[query failed] split={split_name} id={query_id} err={exc}")

        save_progress(
            progress_path,
            processed_query_ids=sorted(processed_query_ids),
            stats={**stats, "failed_query_ids": failed_query_ids},
        )

    return {**stats, "failed_query_ids": failed_query_ids}


def build_argparser() -> argparse.ArgumentParser:
    """定义命令行参数，用于配置输入来源、切分方式和输出路径。"""
    parser = argparse.ArgumentParser(description="Generate train/val/test reranker labels from augmented QA data.")
    parser.add_argument("--train-master-path", default=DEFAULT_TRAIN_MASTER_PATH)
    parser.add_argument("--test-master-path", default=DEFAULT_TEST_MASTER_PATH)
    parser.add_argument("--final-docs-path", default=DEFAULT_FINAL_DOCS_PATH)
    parser.add_argument("--train-qa-path", default=DEFAULT_TRAIN_QA_PATH)
    parser.add_argument("--test-qa-path", default=DEFAULT_TEST_QA_PATH)
    parser.add_argument("--qa-generalized-path", default=DEFAULT_QA_GENERALIZED_PATH)
    parser.add_argument("--train-output-path", default=DEFAULT_TRAIN_OUTPUT_PATH)
    parser.add_argument("--val-output-path", default=DEFAULT_VAL_OUTPUT_PATH)
    parser.add_argument("--test-output-path", default=DEFAULT_TEST_OUTPUT_PATH)
    parser.add_argument("--train-progress-path", default=DEFAULT_TRAIN_PROGRESS_PATH)
    parser.add_argument("--val-progress-path", default=DEFAULT_VAL_PROGRESS_PATH)
    parser.add_argument("--test-progress-path", default=DEFAULT_TEST_PROGRESS_PATH)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--bm25-topk", type=int, default=DEFAULT_BM25_TOPK)
    parser.add_argument("--milvus-topk", type=int, default=DEFAULT_MILVUS_TOPK)
    parser.add_argument("--max-doc-chars", type=int, default=DEFAULT_MAX_LLM_DOC_CHARS)
    parser.add_argument("--max-retry", type=int, default=DEFAULT_MAX_RETRY)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main():
    """执行排序器训练数据生成主流程，产出 train、val、test 三个划分。"""
    args = build_argparser().parse_args()

    train_master_rows = load_jsonl(args.train_master_path)
    test_master_rows = load_jsonl(args.test_master_path)

    uid_to_doc = load_final_docs_by_uid(args.final_docs_path)
    question_uid_to_question = load_qa_question_map([args.train_qa_path, args.test_qa_path])
    question_to_source_doc_uid = load_question_to_source_doc_map(args.qa_generalized_path)
    train_rows, val_rows, split_stats = split_rows_by_group_balance(
        rows=train_master_rows,
        val_ratio=args.val_ratio,
        seed=args.split_seed,
        question_uid_to_question=question_uid_to_question,
        question_to_source_doc_uid=question_to_source_doc_uid,
        uid_to_doc=uid_to_doc,
    )
    llm_client, model_name = make_llm_client()

    print(f"[info] train_master_rows={len(train_master_rows)}")
    print(f"[info] train_rows={len(train_rows)}")
    print(f"[info] val_rows={len(val_rows)}")
    print(f"[info] test_rows={len(test_master_rows)}")
    print(f"[info] split_groups={split_stats['group_count']}")
    print(f"[info] train_groups={split_stats['train_group_count']}")
    print(f"[info] val_groups={split_stats['val_group_count']}")
    print(f"[info] target_val_rows={split_stats['target_val_rows']}")
    print(f"[info] actual_val_ratio={split_stats['actual_val_ratio']:.4f}")
    print(f"[info] final_docs={len(uid_to_doc)}")
    print(f"[info] train_qa_questions={len(question_uid_to_question)}")
    print(f"[info] question_to_source_doc={len(question_to_source_doc_uid)}")
    print(f"[info] llm_model={model_name}")

    bm25 = BM25(docs=None)
    milvus = MilvusRetriever(docs=None)

    split_summaries = []
    split_summaries.append(
        run_split(
            split_name="train",
            rows=train_rows,
            output_path=args.train_output_path,
            progress_path=args.train_progress_path,
            overwrite=args.overwrite,
            uid_to_doc=uid_to_doc,
            question_uid_to_question=question_uid_to_question,
            question_to_source_doc_uid=question_to_source_doc_uid,
            bm25=bm25,
            milvus=milvus,
            llm_client=llm_client,
            model_name=model_name,
            bm25_topk=args.bm25_topk,
            milvus_topk=args.milvus_topk,
            max_doc_chars=args.max_doc_chars,
            max_retry=args.max_retry,
        )
    )
    split_summaries.append(
        run_split(
            split_name="val",
            rows=val_rows,
            output_path=args.val_output_path,
            progress_path=args.val_progress_path,
            overwrite=args.overwrite,
            uid_to_doc=uid_to_doc,
            question_uid_to_question=question_uid_to_question,
            question_to_source_doc_uid=question_to_source_doc_uid,
            bm25=bm25,
            milvus=milvus,
            llm_client=llm_client,
            model_name=model_name,
            bm25_topk=args.bm25_topk,
            milvus_topk=args.milvus_topk,
            max_doc_chars=args.max_doc_chars,
            max_retry=args.max_retry,
        )
    )
    split_summaries.append(
        run_split(
            split_name="test",
            rows=test_master_rows,
            output_path=args.test_output_path,
            progress_path=args.test_progress_path,
            overwrite=args.overwrite,
            uid_to_doc=uid_to_doc,
            question_uid_to_question=question_uid_to_question,
            question_to_source_doc_uid=question_to_source_doc_uid,
            bm25=bm25,
            milvus=milvus,
            llm_client=llm_client,
            model_name=model_name,
            bm25_topk=args.bm25_topk,
            milvus_topk=args.milvus_topk,
            max_doc_chars=args.max_doc_chars,
            max_retry=args.max_retry,
        )
    )

    for summary in split_summaries:
        print(
            f"[done] split={summary['split']} processed={summary['processed_queries']} "
            f"written_rows={summary['written_rows']} failed={summary['failed_queries']} skipped={summary['skipped_queries']}"
        )


if __name__ == "__main__":
    main()
