"""
升级版过滤脚本：
1. 保留规则粗筛 + LLM 复判的总体框架。
2. LLM 阶段使用：小 batch + 线程池 + as_completed。
3. 对可疑样本自动单文档复判，降低 batch 的相对比较效应。
4. 输出文件格式保持不变，断点续跑保持不变。
"""
import os
import re
import json
import time
import random
import hashlib
import threading
import concurrent.futures
from typing import Tuple, List, Dict, Set

from tqdm import tqdm
from langchain_core.documents import Document
from openai import OpenAI

random.seed(42)

# ===== 可调参数 =====
MAX_WORKERS = 6
LLM_BATCH_SIZE = 4
LLM_MAX_RETRY = 3
CHECKPOINT_EVERY = 20
SINGLE_RECHECK_ENABLED = True

llm_client = OpenAI(
    api_key=os.environ["DOUBAO_API_KEY"],
    base_url=os.environ["DOUBAO_BASE_URL"]
)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())

def count_words(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))

#简单分割文本
def split_sentences_simple(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

#利用开头判断是否是不完整的开头
def starts_like_incomplete(text: str) -> bool:
    text = normalize_text(text)
    if not text:
        return True
    if text[0] in ".,;:)":
        return True
    bad_starts = [
        "however", "but", "thus", "therefore", "moreover", "furthermore",
        "for example", "for instance", "in this case", "in particular",
        "they", "it", "this", "these", "such", "he", "she", "we"
    ]
    first_words = text[:80].lower()
    return any(first_words.startswith(x) for x in bad_starts)

#利用结尾判断是否是未说完的
def ends_like_incomplete(text: str) -> bool:
    text = normalize_text(text)
    if not text:
        return True
    if text[-1] in ".!?":
        return False
    if text[-1] in ",:;-(":
        return True
    last_part = text[-30:].lower()
    return bool(re.search(r"\b(and|or|of|to|for|with|by|in|on|that)$", last_part))

#噪声字符占比
def noise_ratio(text: str) -> float:
    """
    除去字母、数字、常见标点符号、空格以外的均是噪声字符
    """
    text = text or ""
    if not text:
        return 1.0
    noise_chars = re.findall(r"[^\w\s\.,;:!?()\-\[\]/%]+", text)
    return len(noise_chars) / max(len(text), 1)

#利用字符判断是否是低质量文本
def low_information_text(text: str) -> bool:
    s = normalize_text(text)
    if not s:
        return True
    patterns = [
        r"^chapter\s+\d+",
        r"^exercise\s+\d+",
        r"^references?$",
        r"^bibliography$",
        r"^index$",
        r"^summary$",
        r"^in this chapter",
        r"^we will see",
        r"^this chapter",
    ]
    return any(re.search(p, s, re.I) for p in patterns)

#利用字符判断是否是高质量文本
def is_ir_high_value_text(text: str) -> bool:
    s = normalize_text(text).lower()
    ir_terms = [
        "information retrieval", "inverted index", "postings list",
        "boolean retrieval", "vector space model", "tf-idf", "bm25",
        "query", "document", "term frequency", "tokenization",
        "stemming", "ranking", "indexing", "relevance feedback",
        "cosine similarity", "precision", "recall", "dictionary",
        "retrieval model", "collection", "vocabulary"
    ]
    if any(term in s for term in ir_terms):
        return True
    if re.search(r"\b(is|are|refers to|defined as|means|consists of|can be)\b", s) and count_words(s) >= 8:
        return True
    return False

#判断是否是图表caption
def looks_like_caption(text: str) -> bool:
    s = normalize_text(text)
    return bool(re.match(r"^[◮]?\s*(Figure|Fig\.?|Table|Algorithm)\s+\d+(?:\.\d+)*\b", s, flags=re.I))

#判断是否是习题
def looks_like_exercise(text: str) -> bool:
    s = normalize_text(text)
    return bool(re.search(r"\bExercise\s+\d+(?:\.\d+)*\b", s, flags=re.I))

#判断是否是步骤、选项文本
def looks_like_steps_or_options(text: str) -> bool:
    s = normalize_text(text)
    if re.match(r"^[a-z]\.", s, flags=re.I):
        return True
    if re.match(r"^\d+\.\s", s):
        return True
    if len(re.findall(r"\b\d+\.\b", s)) >= 2:
        return True
    return False

#判断一个文本块是否可以脱离图表信息
def has_strong_figure_dependency(text: str, metadata: Dict) -> bool:
    """
    判断逻辑是非图表标题且引用图表,但是又没有图表的内容时认为是强依赖
    """
    s = normalize_text(text)    

    if looks_like_caption(s):
        return False
    
    has_figure_mention = bool(
        re.search(r"\b(Figure|Fig\.?|Table|Algorithm)\b", s, flags=re.I)
    )
    has_reference_phrase = bool(
        re.search(r"\b(as shown in|see|shown in)\b", s, flags=re.I)
    )
    has_meta_figure = bool(metadata.get("figure_refs"))
    if (has_figure_mention or has_reference_phrase) and not has_meta_figure:
        return True

    return False

#判断是否像是在讲知识
def has_core_knowledge_patterns(text: str) -> bool:
    s = normalize_text(text).lower()
    patterns = [
        r"\b(is|are|refers to|defined as|means)\b",
        r"\b(unlike|in contrast|distinguished|compared with|compared to)\b",
        r"\b(because|therefore|thus|so that|allows|used to|by)\b",
        r"\b(algorithm|process|method|procedure|step)\b",
    ]
    return any(re.search(p, s) for p in patterns)


def md5_text(s: str) -> str:
    return hashlib.md5((s or "").encode("utf-8")).hexdigest()

#生成uid
def get_doc_uid(doc: Document) -> str:
    metadata = doc.metadata or {}
    uid = metadata.get("unique_id")
    if uid:
        return str(uid)
    base = "||".join([
        str(metadata.get("source", "")),
        str(metadata.get("orig_page_no", "")),
        str(metadata.get("page_no", "")),
        str(metadata.get("chunk_index", "")),
        normalize_text(doc.page_content)[:500],
    ])
    return md5_text(base)


def load_docs(path: str):
    try:
        docs = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                docs.append(Document(page_content=item["page_content"], metadata=item.get("metadata", {})))
        if docs:
            print("检测为 JSONL 格式")
            return docs
    except Exception:
        pass

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    docs = [Document(page_content=item["page_content"], metadata=item.get("metadata", {})) for item in data]
    print("检测为 JSON(list) 格式")
    return docs


def append_doc_jsonl(doc: Document, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"page_content": doc.page_content, "metadata": doc.metadata}, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


DOC_FILTER_PROMPT_TPL = """
你是一名信息检索（Information Retrieval）与搜索系统领域的专家。

现在给你一段教材或技术文档中的文本，请判断它是否适合用于生成“高质量问答对（QA）”微调数据。

分类标准：
1. core
仅当该文本本身能够稳定支持多个（通常≥4）不重复的高质量问答时，才判为 core。
通常要求：
- 语义完整，能独立理解
- 含有明确知识点、定义、原理、方法、机制、步骤解释或对比关系
- 不依赖图表、练习题语境、上一节/下一节上下文
- 不是单纯一条短定义，且不仅仅只能问出一个问题

2. extra
可保留，但更适合生成 1 个左右的问答，而不是多个高质量问答。
通常包括：
- 单句但完整的定义、解释或局部知识点
- 有价值，但信息密度略低于 core
- 语义基本完整，但不足以稳定支撑多个不重复问答

3. drop
不适合用于生成 QA。
通常包括：
- 空内容、噪声、乱码
- 练习题、题目选项、作图要求、编号步骤残片
- 图题、表题、算法标题，或强依赖图表/章节上下文
- 过于残缺、过渡句、引导句、总结句

请严格以“该文本单独用于后续 QA 生成时的价值”为准。
不要因为一句话能看懂就轻易判为 core。

请严格返回如下 JSON，不要输出其他内容：
{"label": "core/extra/drop", "reason": "简要说明理由"}

下面是待判断文本：
<document>
{{document}}
</document>

请输出结果：
"""

BATCH_DOC_FILTER_PROMPT_TPL = """
你是一名信息检索（Information Retrieval）与搜索系统领域的专家。

现在给你多段教材或技术文档中的文本，请分别判断它们是否适合用于生成“高质量问答对（QA）”微调数据。

重要要求：
- 每个文档必须独立判断。
- 不要在不同文档之间做相对比较。
- 不要因为某个文档比同批其他文档更短，就自动降低或提升它的标签。
- 你的判断标准必须与“单文档单独判断”尽可能一致。

分类标准：
1. core：只有当该文本本身能够稳定支持多个（通常≥3）不重复高质量问答时，才判为 core。
2. extra：文本有价值，但通常更适合生成 1 个左右问答，或信息密度不足以支撑多个高质量问答。
3. drop：不适合生成 QA，如练习题、图题、表题、算法标题、强上下文依赖、残缺句、噪声等。

返回一个 JSON 对象，key 是文档 unique_id，value 是：
{"label": "core/extra/drop", "reason": "简要说明理由"}

不要输出任何额外说明。

下面是待判断文本：
{{documents}}

请输出结果：
"""

#从大模型生成的内容中提取需要的文本
def extract_json_from_text(text: str):
    text = (text or "").strip()
    if not text:
        raise ValueError("未找到可解析的 JSON")
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
    m = re.search(r"[\[{].*[\]}]", text, flags=re.S)
    if m:
        return json.loads(m.group(0))
    raise ValueError("未找到可解析的 JSON")

#给文本增加辅助信息
def build_doc_filter_text(doc: Document) -> str:
    text = normalize_text(doc.page_content)
    metadata = doc.metadata or {}
    has_figure = bool(metadata.get("figure_refs"))
    has_footnote = bool(metadata.get("related_footnotes"))
    orig_page_no = metadata.get("orig_page_no", "")
    chunk_level = metadata.get("chunk_level", "")
    parts = [
        "正文：",
        text,
        "",
        "辅助信息：",
        f"- 是否涉及图表引用: {'是' if has_figure else '否'}",
        f"- 是否有关联脚注: {'是' if has_footnote else '否'}",
        f"- 是否像图题/表题: {'是' if looks_like_caption(text) else '否'}",
        f"- 是否像练习题/题目: {'是' if looks_like_exercise(text) else '否'}",
        f"- 是否像编号步骤/选项: {'是' if looks_like_steps_or_options(text) else '否'}",
        f"- 是否强依赖图表: {'是' if has_strong_figure_dependency(text, metadata) else '否'}",
    ]
    if orig_page_no != "":
        parts.append(f"- 原始页码: {orig_page_no}")
    if chunk_level != "":
        parts.append(f"- 文档层级: {chunk_level}")
    return "\n".join(parts).strip()

#调用大模型，失败时会重新调用
def call_chat(prompt: str, model_name: str, temperature: float = 0.0, max_retry: int = LLM_MAX_RETRY):
    remain = max_retry
    while remain > 0:
        try:
            resp = llm_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return resp.choices[0].message.content
        except Exception:
            remain -= 1
            if remain > 0:
                time.sleep(random.randint(1, 3))
    return None

#传入文本，拼接辅助信息，调用大模型，返回label信息.用于处理批次调用失败的情况
def classify_doc_with_llm(doc: Document, model_name: str, prompt_tpl: str = DOC_FILTER_PROMPT_TPL) -> Tuple[str, str]:
    input_text = build_doc_filter_text(doc)
    prompt = prompt_tpl.replace("{{document}}", input_text)
    raw_text = call_chat(prompt, model_name=model_name, temperature=0.0)
    if raw_text is None:
        return "extra", "llm_error_fallback_extra:RequestFailed"
    try:
        result = extract_json_from_text(raw_text)
        label = result.get("label", "drop")
        reason = result.get("reason", "unknown")
        if label not in {"core", "extra", "drop"}:
            return "extra", f"llm_error_fallback_extra:invalid_label:{label}"
        return label, reason
    except Exception as e:
        return "extra", f"llm_error_fallback_extra:{type(e).__name__}"

#建立batch，用于减少大模型请求次数
def build_batch_filter_documents_text(batch_docs: List[Document]) -> str:
    parts = []
    for doc in batch_docs:
        metadata = doc.metadata or {}
        uid = metadata.get("resume_uid") or metadata.get("unique_id") or md5_text(normalize_text(doc.page_content)[:500])
        doc_text = build_doc_filter_text(doc)
        parts.append(f'<doc unique_id="{uid}">\n{doc_text}\n</doc>')
    return "\n\n".join(parts)

#批次调用大模型并返回结果
def classify_docs_with_llm_batch(batch_docs: List[Document], model_name: str) -> Dict[str, Tuple[str, str]]:
    """
    批次处理文本,调用大模型.当批次调用失败时,再尝试一个个调用大模型
    """
    docs_text = build_batch_filter_documents_text(batch_docs)
    prompt = BATCH_DOC_FILTER_PROMPT_TPL.replace("{{documents}}", docs_text)
    raw_text = call_chat(prompt, model_name=model_name, temperature=0.0)
    if raw_text is None:
        #处理批次调用失败的情况
        return {doc.metadata["resume_uid"]: classify_doc_with_llm(doc, model_name=model_name) for doc in batch_docs}
    try:
        result = extract_json_from_text(raw_text)
        if not isinstance(result, dict):
            raise ValueError("batch result is not dict")
        outputs: Dict[str, Tuple[str, str]] = {}
        for doc in batch_docs:
            uid = doc.metadata["resume_uid"]
            item = result.get(uid, {})
            label = item.get("label", "extra") if isinstance(item, dict) else "extra"
            reason = item.get("reason", "unknown") if isinstance(item, dict) else "unknown"
            if label not in {"core", "extra", "drop"}:
                label, reason = "extra", f"llm_error_fallback_extra:invalid_label:{label}"
            outputs[uid] = (label, reason)
        return outputs
    except Exception:
        return {doc.metadata["resume_uid"]: classify_doc_with_llm(doc, model_name=model_name) for doc in batch_docs}

#判断是否要二次判断label
def should_single_recheck(doc: Document, label: str, route_reason: str, reason: str) -> bool:
    """
    返回False表示不用二次处理.
    下列是要再检测的情况：
    如果文本是习题、图标题、强依赖图,如果当前不是drop则可能误判
    如果是core但是太短
    如果是drop但是是高质量文本
    如果是core但是边界样本且没有核心知识
    调用大模型报错,为了避免重复报错
    """
    text = normalize_text(doc.page_content)
    metadata = doc.metadata or {}
    word_num = count_words(text)
    sent_num = len(split_sentences_simple(text))
    if looks_like_exercise(text) or has_strong_figure_dependency(text, metadata):
        return label != "drop"
    if label == "core" and (word_num < 35 or sent_num <= 1):
        return True
    if label == "drop" and is_ir_high_value_text(text) and word_num >= 18:
        return True
    #看起来有价值 + 被LLM判成core，但不像在讲知识 → 高风险误判 → 需要复查
    if label == "core" and route_reason == "borderline_high_value" and not has_core_knowledge_patterns(text):
        return True
    if "llm_error_fallback_extra" in reason:
        return False
    return False

#在大模型产生标签和初步判断产生标签中保守选择
def conservative_merge_label(batch_label: str, single_label: str) -> str:
    order = {"drop": 0, "extra": 1, "core": 2}
    return batch_label if order[batch_label] <= order[single_label] else single_label


#初步生成label
def pre_filter_doc(
    doc: Document,
    core_min_words: int = 50,
    core_min_sentences: int = 2,
    extra_min_words: int = 8,
    max_noise_ratio: float = 0.08,
) -> Tuple[str, str]:
    """
    硬过滤:空文本、噪声文本、低信息文本、不完整文本直接丢
    练习块、问题块以及依赖图表但是没有图表信息直接丢
    """

    #硬过滤
    text = normalize_text(doc.page_content)
    metadata = doc.metadata or {}
    if not text:
        return "drop", "empty"
    if noise_ratio(text) > max_noise_ratio:
        return "drop", "too_noisy"
    if low_information_text(text):
        return "drop", "low_information"
    if starts_like_incomplete(text) and ends_like_incomplete(text):
        return "drop", "incomplete_both_ends"

    word_num = count_words(text)
    sent_num = len(split_sentences_simple(text))
    high_value = is_ir_high_value_text(text)
    caption_like = looks_like_caption(text)
    exercise_like = looks_like_exercise(text)
    step_like = looks_like_steps_or_options(text)
    strong_figure_dep = has_strong_figure_dependency(text, metadata)
    context_dependent = starts_like_incomplete(text)  #大概率对前文有依赖

    doc.metadata["qa_caption_like"] = caption_like
    doc.metadata["qa_exercise_like"] = exercise_like
    doc.metadata["qa_step_like"] = step_like
    doc.metadata["qa_context_dependent"] = context_dependent
    doc.metadata["qa_strong_figure_dependent"] = strong_figure_dep

    if exercise_like:
        return "drop", "exercise_or_question_block"
    if strong_figure_dep:
        return "drop", "caption_or_strong_figure_dependency"

    if (
        word_num >= core_min_words
        and sent_num >= core_min_sentences
        and not step_like
        and not context_dependent
        and has_core_knowledge_patterns(text)
    ):
        return "core_rule", "rule_high_quality"

    if (
        high_value
        and extra_min_words <= word_num < core_min_words
        and sent_num <= 2
        and not context_dependent
        and not step_like
    ):
        return "extra_rule", "rule_short_but_complete"
    
    #对于看起来有价值但是达不到core、extra标准的，交给LLM
    if high_value and word_num >= extra_min_words:
        return "need_llm", "borderline_high_value"
    #规则判不稳，交给LLM
    return "need_llm", "borderline"


def load_processed_ids_from_jsonl(path: str) -> Set[str]:
    processed_ids = set()
    if not os.path.exists(path):
        return processed_ids
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                metadata = item.get("metadata", {}) or {}
                uid = metadata.get("resume_uid")
                if uid:
                    processed_ids.add(uid)
            except Exception:
                continue
    return processed_ids


def load_checkpoint(path: str) -> Dict:
    if not os.path.exists(path):
        return {"route_stats": {}, "final_stats": {}, "processed_count": 0}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {
                "route_stats": data.get("route_stats", {}),
                "final_stats": data.get("final_stats", {}),
                "processed_count": data.get("processed_count", 0),
            }
    except Exception:
        return {"route_stats": {}, "final_stats": {}, "processed_count": 0}


def save_checkpoint(path: str, route_stats: Dict, final_stats: Dict, processed_count: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {"route_stats": route_stats, "final_stats": final_stats, "processed_count": processed_count}
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def count_jsonl_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def finalize_one_doc(
    doc: Document,
    uid: str,
    label: str,
    reason: str,
    source: str,
    route: str,
    rule_reason: str,
    core_output_path: str,
    extra_output_path: str,
    drop_output_path: str,
    processed_ids: Set[str],
    route_stats: Dict,
    final_stats: Dict,
    counts: Dict[str, int],
    checkpoint_every: int,
    checkpoint_path: str,
    file_lock: threading.Lock,
    progress_state: Dict[str, int],
):
    text = normalize_text(doc.page_content)
    doc.metadata["qa_doc_type"] = label
    doc.metadata["qa_filter_reason"] = reason
    doc.metadata["qa_filter_source"] = source
    doc.metadata["qa_prefilter_route"] = route
    doc.metadata["qa_prefilter_reason"] = rule_reason
    doc.metadata["qa_word_count"] = count_words(text)
    doc.metadata["qa_sentence_count"] = len(split_sentences_simple(text))
    doc.metadata["qa_has_figure"] = bool(doc.metadata.get("figure_refs"))
    doc.metadata["qa_has_footnote"] = bool(doc.metadata.get("related_footnotes"))

    with file_lock:
        if uid in processed_ids:
            return
        final_stats[f"{label}:{reason}"] = final_stats.get(f"{label}:{reason}", 0) + 1
        if label == "core":
            append_doc_jsonl(doc, core_output_path)
            counts["core"] += 1
        elif label == "extra":
            append_doc_jsonl(doc, extra_output_path)
            counts["extra"] += 1
        else:
            append_doc_jsonl(doc, drop_output_path)
            counts["drop"] += 1
        processed_ids.add(uid)
        progress_state["newly_processed"] += 1
        if progress_state["newly_processed"] % checkpoint_every == 0:
            save_checkpoint(checkpoint_path, route_stats=route_stats, final_stats=final_stats, processed_count=len(processed_ids))


def main():
    input_path = "/root/autodl-tmp/IR-RAG-System/data/processed_docs/final_split_docs"
    output_dir = "/root/autodl-tmp/IR-RAG-System/data/processed_docs/qa_filter_outputs"
    core_output_path = os.path.join(output_dir, "core_docs.jsonl")
    extra_output_path = os.path.join(output_dir, "extra_docs.jsonl")
    drop_output_path = os.path.join(output_dir, "drop_docs.jsonl")
    checkpoint_path = os.path.join(output_dir, "checkpoint.json")

    model_name = os.environ["DOUBAO_MODEL_NAME"]
    os.makedirs(output_dir, exist_ok=True)
    docs = load_docs(input_path)
    print(f"原始文档数: {len(docs)}")

    processed_ids = set()
    processed_ids |= load_processed_ids_from_jsonl(core_output_path)
    processed_ids |= load_processed_ids_from_jsonl(extra_output_path)
    processed_ids |= load_processed_ids_from_jsonl(drop_output_path)

    ckpt = load_checkpoint(checkpoint_path)
    route_stats = ckpt["route_stats"]
    final_stats = ckpt["final_stats"]

    counts = {
        "core": count_jsonl_lines(core_output_path),
        "extra": count_jsonl_lines(extra_output_path),
        "drop": count_jsonl_lines(drop_output_path),
    }

    print(f"已恢复已处理文档数: {len(processed_ids)}")
    print(f"已存在核心文档数: {counts['core']}")
    print(f"已存在扩展文档数: {counts['extra']}")
    print(f"已存在丢弃文档数: {counts['drop']}")

    file_lock = threading.Lock()
    progress_state = {"newly_processed": 0}
    llm_candidates: List[Tuple[Document, str, str]] = []

    pbar = tqdm(docs, desc="Filtering docs", dynamic_ncols=True)
    try:
        for doc in pbar:
            doc.metadata = doc.metadata or {}
            uid = get_doc_uid(doc)
            doc.metadata["resume_uid"] = uid
            if uid in processed_ids:
                pbar.set_postfix({"skip": len(processed_ids), "core": counts["core"], "extra": counts["extra"], "drop": counts["drop"]})
                continue
            
            #规则过滤
            route, rule_reason = pre_filter_doc(doc)
            route_stats[route] = route_stats.get(route, 0) + 1

            if route == "drop":
                finalize_one_doc(doc, uid, "drop", rule_reason, "rule", route, rule_reason,
                                 core_output_path, extra_output_path, drop_output_path,
                                 processed_ids, route_stats, final_stats, counts,
                                 CHECKPOINT_EVERY, checkpoint_path, file_lock, progress_state)
            elif route == "core_rule":
                finalize_one_doc(doc, uid, "core", rule_reason, "rule", route, rule_reason,
                                 core_output_path, extra_output_path, drop_output_path,
                                 processed_ids, route_stats, final_stats, counts,
                                 CHECKPOINT_EVERY, checkpoint_path, file_lock, progress_state)
            elif route == "extra_rule":
                finalize_one_doc(doc, uid, "extra", rule_reason, "rule", route, rule_reason,
                                 core_output_path, extra_output_path, drop_output_path,
                                 processed_ids, route_stats, final_stats, counts,
                                 CHECKPOINT_EVERY, checkpoint_path, file_lock, progress_state)
            else:
                llm_candidates.append((doc, uid, rule_reason))

            #更新规则阶段的进度条信息
            processed_total = counts["core"] + counts["extra"] + counts["drop"]
            kept_total = counts["core"] + counts["extra"]
            pbar.set_postfix({
                "core": counts["core"],
                "extra": counts["extra"],
                "drop": counts["drop"],
                "keep_rate": f"{kept_total / max(1, processed_total):.2%}",
                "done": len(processed_ids),
                "llm_wait": len(llm_candidates),
            })

        #对于第一次处理认为需要进一步判断的，利用大模型来进行判断
        if llm_candidates:
            batches = [llm_candidates[i:i + LLM_BATCH_SIZE] for i in range(0, len(llm_candidates), LLM_BATCH_SIZE)]
            future_to_batch = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                for batch in batches:
                    batch_docs = [x[0] for x in batch]
                    future = executor.submit(classify_docs_with_llm_batch, batch_docs, model_name)
                    future_to_batch[future] = batch

                llm_pbar = tqdm(concurrent.futures.as_completed(future_to_batch), total=len(future_to_batch),
                                desc="Filtering docs (LLM batch)", dynamic_ncols=True)

                for future in llm_pbar:
                    batch = future_to_batch[future]
                    uid_to_result = future.result()

                    for doc, uid, rule_reason in batch:
                        label, reason = uid_to_result.get(uid, ("extra", "llm_error_fallback_extra:missing_uid_result"))
                        source = "llm_batch"

                        #如果启用了单条复查，而且这条值得复查，就再单独调用一次 LLM
                        if SINGLE_RECHECK_ENABLED and should_single_recheck(doc, label, rule_reason, reason):
                            single_label, single_reason = classify_doc_with_llm(doc, model_name=model_name)
                            label = conservative_merge_label(label, single_label)
                            reason = f"batch={reason} | single={single_reason}"
                            source = "llm_batch_plus_single"

                        finalize_one_doc(doc, uid, label, reason, source, "need_llm", rule_reason,
                                         core_output_path, extra_output_path, drop_output_path,
                                         processed_ids, route_stats, final_stats, counts,
                                         CHECKPOINT_EVERY, checkpoint_path, file_lock, progress_state)

                    processed_total = counts["core"] + counts["extra"] + counts["drop"]
                    kept_total = counts["core"] + counts["extra"]
                    llm_pbar.set_postfix({
                        "core": counts["core"],
                        "extra": counts["extra"],
                        "drop": counts["drop"],
                        "keep_rate": f"{kept_total / max(1, processed_total):.2%}",
                        "done": len(processed_ids),
                    })

    except KeyboardInterrupt:
        print("\n检测到手动中断，正在保存 checkpoint...")
    finally:
        save_checkpoint(checkpoint_path, route_stats=route_stats, final_stats=final_stats, processed_count=len(processed_ids))

    print("\n===== 预筛路由统计 =====")
    for k, v in sorted(route_stats.items(), key=lambda x: -x[1]):
        print(f"{k}: {v}")

    print("\n===== 最终分类统计 =====")
    for k, v in sorted(final_stats.items(), key=lambda x: -x[1]):
        print(f"{k}: {v}")

    print(f"\n核心文档数: {counts['core']}")
    print(f"扩展文档数: {counts['extra']}")
    print(f"丢弃文档数: {counts['drop']}")
    print(f"总保留比例: {(counts['core'] + counts['extra']) / max(1, (counts['core'] + counts['extra'] + counts['drop'])):.2%}")

    print(f"\n核心文档已保存到: {core_output_path}")
    print(f"扩展文档已保存到: {extra_output_path}")
    print(f"丢弃文档已保存到: {drop_output_path}")
    print(f"Checkpoint 已保存到: {checkpoint_path}")


if __name__ == "__main__":
    main()
