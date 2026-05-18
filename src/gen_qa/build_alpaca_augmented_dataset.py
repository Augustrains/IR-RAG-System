import os
import re
import json
import random
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from src.path import bge_m3_model_path


bge_m3_model_path = "/root/autodl-tmp/IR-RAG-System/models/bge-m3"

embedding_handler = BGEM3EmbeddingFunction(
    model_name=bge_m3_model_path,
    device="cuda"
)
random.seed(42)

# =========================
# Path Config
# =========================

INPUT_PATH = "/root/autodl-tmp/IR-RAG-System/data/qa_pairs/test_qa_keywords.jsonl"

# 中间主表：保留 task_type / question_type / source_uid 等元信息
MASTER_OUTPUT_PATH = "/root/autodl-tmp/IR-RAG-System/data/qa_pairs/test_augmented_master.jsonl"

# 最终给 LLaMA-Factory 的 Alpaca 格式
ALPACA_OUTPUT_PATH = "/root/autodl-tmp/IR-RAG-System/data/qa_pairs/test_augmented_alpaca.jsonl"

# INPUT_PATH = "/root/autodl-tmp/IR-RAG-System/data/qa_pairs/train_qa_keywords.jsonl"

# # 中间主表：保留 task_type / question_type / source_uid 等元信息
# MASTER_OUTPUT_PATH = "/root/autodl-tmp/IR-RAG-System/data/qa_pairs/train_augmented_master.jsonl"

# # 最终给 LLaMA-Factory 的 Alpaca 格式
# ALPACA_OUTPUT_PATH = "/root/autodl-tmp/IR-RAG-System/data/qa_pairs/train_augmented_alpaca.jsonl"

# 可选：如果你还想导出一个 test 版，就再跑一次，换输入路径和输出路径即可
SYSTEM_PROMPT = "你是信息检索领域的专业助手。"


# =========================
# Utils
# =========================

def ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def contains_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def safe_lower(text: str) -> str:
    return (text or "").strip().lower()

#清洗出来关键字
def dedup_keywords(keywords: Any, max_keywords: int = 5) -> List[str]:
    if not isinstance(keywords, list):
        return []

    cleaned = []
    seen = set()

    for kw in keywords:
        if not isinstance(kw, str):
            continue
        kw = normalize_text(kw)
        if not kw:
            continue

        low = kw.lower()
        if low in seen:
            continue
        seen.add(low)
        cleaned.append(kw)

    return cleaned[:max_keywords]


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    if not os.path.exists(path):
        print(f"[warn] input not found: {path}")
        return rows

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except Exception as e:
                print(f"[load error] line={line_no}, err={e}")
                continue

            unique_id = normalize_text(item.get("unique_id", ""))
            question = normalize_text(item.get("question", ""))
            answer = normalize_text(item.get("answer", ""))
            keywords = dedup_keywords(item.get("keywords", []), max_keywords=5)

            if not unique_id or not question or not answer:
                continue

            rows.append({
                "unique_id": unique_id,
                "question": question,
                "answer": answer,
                "keywords": keywords,
            })

    return rows


def save_jsonl(path: str, rows: List[Dict[str, Any]]):
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

#向量单位化
def l2_normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vecs / norms


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def bgem3_encode(texts: List[str]) -> np.ndarray:
    """
    直接调用现有的 BGEM3EmbeddingFunction
    返回 dense 向量并做 L2 normalize
    """
    emb = embedding_handler(texts)
    dense = np.asarray(emb["dense"], dtype=np.float32)
    dense = l2_normalize(dense)
    return dense

#给问句分类
def classify_question_type_rule(question: str) -> str:
    """
    返回：
    - definition
    - representation
    - mechanism
    - comparison
    - reason
    - factoid
    - other
    """
    q = normalize_text(question)
    ql = safe_lower(q)

    # ---- 中文规则 ----
    if contains_chinese(q):
        # definition
        if re.match(r"^什么是", q):
            return "definition"

        # representation
        if re.search(r"(代表什么|如何表示|怎么表示|是如何表示的|含义是什么)", q):
            return "representation"

        # comparison
        if re.search(r"(区别|不同|相比|相比之下|优于|劣于|哪个更|哪一个更)", q):
            return "comparison"

        # reason
        if re.match(r"^为什么", q) or re.search(r"(为何|原因是什么)", q):
            return "reason"

        # mechanism
        if re.match(r"^如何", q) or re.search(r"(怎样|怎么|如何处理|如何工作|如何实现|如何构建)", q):
            return "mechanism"

        # factoid
        if re.match(r"^(谁|何时|什么时候|哪里|哪种|哪些|多少|多长时间|哪个)", q):
            return "factoid"

        return "other"

    # ---- 英文规则 ----

    # definition
    if re.match(r"^what is\b", ql) or re.match(r"^what are\b", ql):
        return "definition"

    # representation
    if re.match(r"^what does\b", ql) and "represent" in ql:
        return "representation"
    if re.match(r"^how are\b", ql) and "represent" in ql:
        return "representation"
    if "represented" in ql:
        return "representation"

    # comparison
    if re.search(r"\b(difference|compare|compared|versus|vs\.?|better|worse|more commonly)\b", ql):
        return "comparison"

    # reason
    if re.match(r"^why\b", ql):
        return "reason"

    # mechanism
    if re.match(r"^how\b", ql):
        return "mechanism"

    # factoid
    if re.match(r"^(when|where|who|which|how many|how long)\b", ql):
        return "factoid"

    return "other"



QUESTION_TYPE_PROTOTYPES: Dict[str, List[str]] = {
    "definition": [
        "What is BM25?",
        "What is an inverted index?",
        "What is tf-idf?",
        "Define Boolean retrieval.",
        "什么是倒排索引？",
        "什么是 BM25？",
        "什么是自由文本查询？"
    ],
    "representation": [
        "What does n01r represent?",
        "How are documents represented in the vector space model?",
        "What is the meaning of this notation?",
        "这个符号代表什么？",
        "文档是如何表示的？",
        "这个记号的含义是什么？"
    ],
    "mechanism": [
        "How does BM25 work?",
        "How does Boolean retrieval work?",
        "How is an inverted index constructed?",
        "How does the ranking function work?",
        "如何构建倒排索引？",
        "检索系统如何工作？"
    ],
    "comparison": [
        "What is the difference between precision and recall?",
        "How does BM25 compare with tf-idf?",
        "Which is better, stemming or lemmatization?",
        "precision 和 recall 有什么区别？",
        "BM25 和 tf-idf 有什么不同？"
    ],
    "reason": [
        "Why is accuracy misleading in information retrieval?",
        "Why do we need stemming?",
        "Why is query expansion useful?",
        "为什么 accuracy 不是好的指标？",
        "为什么需要词干提取？"
    ],
    "factoid": [
        "When was Boolean search the default method?",
        "Who proposed the vector space model?",
        "How many documents are in the collection?",
        "什么时候 Boolean search 是默认方法？",
        "谁提出了向量空间模型？"
    ],
}


class QuestionTypeEmbedClassifier:
    """
    只在规则失败时，对 question_type 做 prototype 语义匹配
    """
    def __init__(
        self,
        prototypes: Optional[Dict[str, List[str]]] = None,
        threshold: float = 0.50, #相似度阈值
    ):
        self.prototypes = prototypes or QUESTION_TYPE_PROTOTYPES
        self.threshold = threshold

        self.prototype_centroids: Dict[str, np.ndarray] = {}
        self._build_prototype_centroids()
    
    #构建每种类型的向量中心（centroid），用于后续的相似度计算
    def _build_prototype_centroids(self):
        for qtype, texts in self.prototypes.items():
            vecs = bgem3_encode(texts)
            if len(vecs) == 0:
                continue

            centroid = vecs.mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm

            self.prototype_centroids[qtype] = centroid
   
   #计算问题与每个类型中心的余弦相似度，返回得分最高的类型和对应的相似度分数
    def predict_with_score(self, question: str) -> Tuple[str, float]:
        q_vec = bgem3_encode([question])[0]

        best_type = "other"
        best_score = -1.0

        for qtype, centroid in self.prototype_centroids.items():
            score = cosine_sim(q_vec, centroid)
            if score > best_score:
                best_score = score
                best_type = qtype

        if best_score < self.threshold:
            return "other", best_score

        return best_type, best_score

    def predict(self, question: str) -> str:
        return self.predict_with_score(question)[0]


_embed_classifier = None

#创建一个分类器
def get_embed_classifier() -> QuestionTypeEmbedClassifier:
    global _embed_classifier
    if _embed_classifier is None:
        _embed_classifier = QuestionTypeEmbedClassifier(
            threshold=0.50
        )
    return _embed_classifier


def classify_question_type(
    question: str,
    use_embed_fallback: bool = True,
) -> str:
    """
    推荐最终入口：
    1. 先规则分类
    2. 如果规则返回 other，再用 BGEM3 fallback
    """
    qtype = classify_question_type_rule(question)

    if qtype != "other":
        return qtype

    if use_embed_fallback:
        embed_classifier = get_embed_classifier()
        return embed_classifier.predict(question)

    return "other"

#构建单纯QA问题
def build_original_sample(item: Dict[str, Any], question_type: str) -> Dict[str, Any]:
    q = item["question"]
    a = item["answer"]

    if contains_chinese(q) or contains_chinese(a):
        instruction = "请回答信息检索问题。"
    else:
        instruction = "Please answer the information retrieval question."

    return {
        "id": f'{item["unique_id"]}_orig',
        "source_uid": item["unique_id"],
        "task_type": "qa",
        "question_type": question_type,
        "instruction": instruction,
        "input": q,
        "output": a,
        "system": SYSTEM_PROMPT,
    }

#构建含关键词的问题
def build_question_keywords_sample(item: Dict[str, Any], question_type: str) -> Dict[str, Any]:
    q = item["question"]
    a = item["answer"]
    kws = item["keywords"]

    kw_text = ", ".join(kws) if kws else "无"

    if contains_chinese(q) or contains_chinese(a):
        instruction = "请根据问题和关键词给出准确回答。"
        input_text = f"问题：{q}\n关键词：{kw_text}"
    else:
        instruction = "Please answer accurately using the question and keywords."
        input_text = f"Question: {q}\nKeywords: {kw_text}"

    return {
        "id": f'{item["unique_id"]}_qkw',
        "source_uid": item["unique_id"],
        "task_type": "qa_with_keywords",
        "question_type": question_type,
        "instruction": instruction,
        "input": input_text,
        "output": a,
        "system": SYSTEM_PROMPT,
    }

#构建含解释类的问题
def build_term_explanation_sample(item: Dict[str, Any], question_type: str) -> Optional[Dict[str, Any]]:
    """
    只对 definition 做，因为这有本类的任务，才能顺利提取出一个可以解释的术语
    从 question 或 keywords 中抽一个最核心术语。
    """
    if question_type != "definition":
        return None

    q = item["question"]
    a = item["answer"]
    kws = item["keywords"]

    term = ""

    # 中文：什么是XXX
    if contains_chinese(q):
        m = re.match(r"^什么是(.+?)[？?]?$", q)
        if m:
            term = normalize_text(m.group(1))
        elif kws:
            term = kws[0]

        if not term:
            return None

        return {
            "id": f'{item["unique_id"]}_term',
            "source_uid": item["unique_id"],
            "task_type": "term_explanation",
            "question_type": question_type,
            "instruction": "请解释下列信息检索术语。",
            "input": term,
            "output": a,
            "system": SYSTEM_PROMPT,
        }

    # 英文：What is / What are
    ql = safe_lower(q)
    if re.match(r"^what is\b", ql):
        term = re.sub(r"^what is\s+", "", q, flags=re.I).rstrip(" ?")
    elif re.match(r"^what are\b", ql):
        term = re.sub(r"^what are\s+", "", q, flags=re.I).rstrip(" ?")
    elif kws:
        term = kws[0]

    term = normalize_text(term)
    if not term:
        return None

    return {
        "id": f'{item["unique_id"]}_term',
        "source_uid": item["unique_id"],
        "task_type": "term_explanation",
        "question_type": question_type,
        "instruction": "Please explain the following information retrieval term.",
        "input": term,
        "output": a,
        "system": SYSTEM_PROMPT,
    }

#构建错误问题，即对于一个原本的QA，构建一个Q变成否定版本的QA，让模型学会判断对错
def make_wrong_statement(item: Dict[str, Any], question_type: str) -> str:
    """
    根据 QA，自动生成一个“看起来合理但其实是错的句子”
    只给 definition / representation 用。
    因为只有定义和表示类任务，因为这两类任务十分容易构建反转的样本
    """
    q = item["question"]
    kws = item["keywords"]
    zh = contains_chinese(q)

    kw1 = kws[0] if len(kws) >= 1 else ""

    if question_type not in {"definition", "representation"}:
        return ""

    if zh:
        # definition
        if question_type == "definition":
            m = re.match(r"^什么是(.+?)[？?]?$", q)
            if m:
                subj = normalize_text(m.group(1))
                if subj:
                    return f"{subj}不是信息检索中的具体概念，只是普通表达。"
            if kw1:
                return f"{kw1}在信息检索中没有明确含义。"
            return ""

        # representation
        if question_type == "representation":
            if kw1:
                return f"{kw1}不能被表示为任何结构化形式。"
            return "该对象通常无法进行任何形式的表示。"

    else:
        # definition
        if question_type == "definition":
            ql = safe_lower(q)
            if re.match(r"^what is\b", ql):
                subj = re.sub(r"^what is\s+", "", q, flags=re.I).rstrip(" ?")
                subj = normalize_text(subj)
                if subj:
                    return f"{subj} is not an information retrieval concept, but merely a general expression."
            if re.match(r"^what are\b", ql):
                subj = re.sub(r"^what are\s+", "", q, flags=re.I).rstrip(" ?")
                subj = normalize_text(subj)
                if subj:
                    return f"{subj} are not used in information retrieval systems."
            if kw1:
                return f"{kw1} has no specific meaning in information retrieval."
            return ""

        # representation
        if question_type == "representation":
            if kw1:
                return f"{kw1} does not represent any specific concept in this context."
            return "This object does not represent any specific concept in this context."

    return ""

#构建判断纠错类的问题
def build_error_correction_sample(item: Dict[str, Any], question_type: str) -> Optional[Dict[str, Any]]:
    """
    只对 definition / representation 做。
    """
    if question_type not in {"definition", "representation"}:
        return None
    
    wrong_stmt = make_wrong_statement(item, question_type)
    if not wrong_stmt:
        return None

    a = item["answer"]
    zh = contains_chinese(item["question"]) or contains_chinese(a) or contains_chinese(wrong_stmt)

    if zh:
        instruction = "判断下面说法是否正确；如果错误，请给出正确表述。"
        input_text = wrong_stmt
        output_text = f"该说法不准确。正确表述是：{a}"
    else:
        instruction = "Determine whether the following statement is correct. If it is incorrect, provide the correct statement."
        input_text = wrong_stmt
        output_text = f"This statement is inaccurate. The correct statement is: {a}"

    return {
        "id": f'{item["unique_id"]}_fix',
        "source_uid": item["unique_id"],
        "task_type": "error_correction",
        "question_type": question_type,
        "instruction": instruction,
        "input": input_text,
        "output": output_text,
        "system": SYSTEM_PROMPT,
    }

#对一个原始 QA 样本，构建多个增强样本
def augment_one_item(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []

    qtype = classify_question_type(item["question"])

    # 1) 原始 QA：全部保留
    out.append(build_original_sample(item, qtype))

    # 2) 问题 + 关键词：全部保留
    out.append(build_question_keywords_sample(item, qtype))

    # 3) 术语解释：只对 definition 做
    term_item = build_term_explanation_sample(item, qtype)
    if term_item is not None:
        out.append(term_item)

    # 4) 判断纠错：只对 definition / representation 做
    fix_item = build_error_correction_sample(item, qtype)
    if fix_item is not None:
        out.append(fix_item)

    return out


def build_master_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    master_rows = []
    seen_ids = set()

    for item in rows:
        augmented = augment_one_item(item)
        for x in augmented:
            sid = x["id"]
            if sid in seen_ids:
                continue
            seen_ids.add(sid)
            master_rows.append(x)

    return master_rows


#转换成训练所需要的格式
def to_alpaca_row(master_item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "instruction": master_item["instruction"],
        "input": master_item["input"],
        "output": master_item["output"],
        "system": master_item["system"],
    }


def build_alpaca_rows(master_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [to_alpaca_row(x) for x in master_rows]

#统计结果
def print_stats(rows: List[Dict[str, Any]], master_rows: List[Dict[str, Any]]):
    print("原始样本数：", len(rows))
    print("增强后主表样本数：", len(master_rows))

    qtype_counter = {}
    task_counter = {}

    # 统计原始 question_type
    for item in rows:
        qtype = classify_question_type(item["question"])
        qtype_counter[qtype] = qtype_counter.get(qtype, 0) + 1

    # 统计增强 task_type
    for item in master_rows:
        task = item["task_type"]
        task_counter[task] = task_counter.get(task, 0) + 1

    print("question_type 分布：")
    for k, v in sorted(qtype_counter.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {k}: {v}")

    print("task_type 分布：")
    for k, v in sorted(task_counter.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {k}: {v}")


def main():
    rows = load_jsonl(INPUT_PATH)
    if not rows:
        print("没有读取到有效输入，程序结束。")
        return

    # 构造中间主表
    master_rows = build_master_rows(rows)

    # 导出主表
    save_jsonl(MASTER_OUTPUT_PATH, master_rows)

    # 导出 Alpaca
    alpaca_rows = build_alpaca_rows(master_rows)
    save_jsonl(ALPACA_OUTPUT_PATH, alpaca_rows)

    # 打印统计
    print_stats(rows, master_rows)

    print("主表输出路径：", MASTER_OUTPUT_PATH)
    print("Alpaca 输出路径：", ALPACA_OUTPUT_PATH)

    # 预览几条
    print("\n===== Alpaca Sample 1 =====")
    print(json.dumps(alpaca_rows[0], ensure_ascii=False, indent=2))

    if len(alpaca_rows) > 1:
        print("\n===== Alpaca Sample 2 =====")
        print(json.dumps(alpaca_rows[1], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()