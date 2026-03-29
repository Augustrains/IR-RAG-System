import re
import json
import hashlib
from typing import List, Dict, Any, Tuple

import numpy as np
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from src.path import bge_m3_model_path, merged_docs


embedding_handler = BGEM3EmbeddingFunction(
    model_name=bge_m3_model_path,
    device="cuda"
)

test_docs = merged_docs


ENUMERATION_ITEM_RE = re.compile(r'^(?:\(?\d+\)|\d+\.|[A-Za-z]\)|[•\-*])\s+')
SECTION_HEADING_RE = re.compile(r'^(?:\d+(?:\.\d+)*)\s+[A-Z]')
INLINE_TERM_HEADING_RE = re.compile(r'^(?:[A-Z][A-Z\- ]{2,}|[A-Z][A-Z\- ]+?:)$')

# =========================
# 0. 结构先验
# =========================
# 分数 > 0: 更倾向切分
# 分数 < 0: 更倾向合并
STRUCTURE_SCORE_MAP: Dict[Tuple[str, str], float] = {
    ("section_heading", "section_heading"): 0.90,
    ("section_heading", "sentence"): -0.22,
    ("section_heading", "list_item"): -0.18,
    ("section_heading", "caption"): 0.95,

    ("inline_term_heading", "sentence"): -0.10,
    ("inline_term_heading", "list_item"): -0.08,
    ("inline_term_heading", "caption"): 0.70,
    ("inline_term_heading", "inline_term_heading"): 0.55,

    ("sentence", "section_heading"): 1.20,
    ("sentence", "inline_term_heading"): 0.10,
    ("sentence", "sentence"): 0.0,
    ("sentence", "list_item"): 0.48,
    ("sentence", "caption"): 0.75,

    ("list_item", "list_item"): 0.72,
    ("list_item", "sentence"): 0.08,
    ("list_item", "section_heading"): 1.10,
    ("list_item", "inline_term_heading"): 0.08,
    ("list_item", "caption"): 0.62,

    ("caption", "section_heading"): 1.10,
    ("caption", "inline_term_heading"): 0.50,
    ("caption", "sentence"): 0.70,
    ("caption", "list_item"): 0.60,
    ("caption", "caption"): 0.70,
}

HARD_CUT_STRUCTURE_PAIRS = {
    ("sentence", "section_heading"),
    ("list_item", "section_heading"),
    ("caption", "section_heading"),
    ("caption", "sentence"),
    ("caption", "list_item"),
    ("list_item", "caption"),
}


# =========================
# 1. 基础工具
# =========================
def md5_text(text: str) -> str:
    return hashlib.md5((text or "").encode("utf-8")).hexdigest()

#去除文本空格
def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()

#记录文本的单词数目
def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))

#对一批向量做L2 归一化
def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return vectors / norms

#计算向量乘积
def cosine_sim(vec1: np.ndarray, vec2: np.ndarray) -> float:
    denom = float(np.linalg.norm(vec1) * np.linalg.norm(vec2))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(vec1, vec2) / denom)

##一组向量聚和成一个块向量
def safe_mean_embedding(vectors: List[np.ndarray], dim: int = 1024) -> np.ndarray:
    if not vectors:
        return np.zeros((dim,), dtype=np.float32)
    mat = np.stack(vectors, axis=0)
    centroid = mat.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm < 1e-12:
        return centroid.astype(np.float32)
    return (centroid / norm).astype(np.float32)

# 鲁棒标准化
def robust_scale(values: List[float]) -> List[float]:
    if not values:
        return []
    arr = np.array(values, dtype=np.float32)
    med = float(np.median(arr))
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    iqr = max(q3 - q1, 1e-6)
    scaled = (arr - med) / iqr
    return scaled.tolist()

#文本-->去重后的词组
def tokenize_for_overlap(text: str) -> set:
    low = (text or "").lower()
    return set(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", low))

#计算文本的词相似度，越相似分数越低
def lexical_shift_score(left_text: str, right_text: str) -> float:
    left_tokens = tokenize_for_overlap(left_text)
    right_tokens = tokenize_for_overlap(right_text)
    if not left_tokens or not right_tokens:
        return 0.0
    inter = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    jaccard = inter / max(union, 1)
    return 1.0 - jaccard


# =========================
# 2. 句子切分
# =========================

#处理包含编号的句子
def split_enumeration_segments(text: str) -> List[str]:
    """
    支持匹配
    """
    s = normalize_space(text)
    if not s:
        return []

    matches = list(re.finditer(r'(?<!\w)(?=(?:\(?\d+\)|\d+\.|[A-Za-z]\))\s+[A-Z])', s))
    if len(matches) <= 1:
        return [s]

    parts: List[str] = []
    first_start = matches[0].start()
    prefix = s[:first_start].strip()
    if prefix:
        parts.append(prefix)

    starts = [m.start() for m in matches] + [len(s)]
    for a, b in zip(starts, starts[1:]):
        seg = s[a:b].strip()
        if seg:
            parts.append(seg)
    return parts

#合并单独编号和后续句子
def merge_marker_only_segments(parts: List[str]) -> List[str]:
    """
    如果当前片段只是一个枚举标记（比如 1. / (1) / a)），就把它和后面的内容合并
    """
    merged: List[str] = []
    i = 0
    while i < len(parts):
        curr = (parts[i] or "").strip()
        if re.fullmatch(r'(?:\(?\d+\)|\d+\.|[A-Za-z]\))', curr) and i + 1 < len(parts):
            nxt = (parts[i + 1] or "").strip()
            merged.append(f"{curr} {nxt}".strip())
            i += 2
            continue
        merged.append(curr)
        i += 1
    return [x for x in merged if x]

#分割文本
def split_into_sentences(text: str) -> List[str]:
    """
    首先按照句末标点+空白+数字/大写字母/"/'来切分句子
    然后,额外处理句子中存在标号的情况，如 1.xxx 2.xxx 


    """
    text = normalize_space(text)
    if not text:
        return []

    protected = {
        "e.g.": "EG_PLACEHOLDER",
        "i.e.": "IE_PLACEHOLDER",
        "etc.": "ETC_PLACEHOLDER",
        "Fig.": "FIG_PLACEHOLDER",
        "Figs.": "FIGS_PLACEHOLDER",
        "Eq.": "EQ_PLACEHOLDER",
        "Eqs.": "EQS_PLACEHOLDER",
        "Dr.": "DR_PLACEHOLDER",
        "Mr.": "MR_PLACEHOLDER",
        "Mrs.": "MRS_PLACEHOLDER",
        "vs.": "VS_PLACEHOLDER",
        "No.": "NO_PLACEHOLDER",
        "Sec.": "SEC_PLACEHOLDER",
        "Ch.": "CH_PLACEHOLDER",
    }
    for k, v in protected.items():
        text = text.replace(k, v)

    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])', text)

    stage2: List[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue

        sub_parts = split_enumeration_segments(p)
        sub_parts = merge_marker_only_segments(sub_parts)
        for sp in sub_parts:
            sp = sp.strip()
            if not sp:
                continue
            stage2.append(sp)

    sentences: List[str] = []
    for p in stage2:
        for k, v in protected.items():
            p = p.replace(v, k)
        p = p.strip()
        if len(p) >= 2:
            sentences.append(p)

    return sentences

# =========================
# 3. 元数据
# =========================

#从多个 unit 的 metadata 中，提取并合并出一个页面级的统一元信息”
def extract_page_metadata(page_item: Dict[str, Any]) -> Dict[str, Any]:
    units = page_item.get("units", [])
    if not units:
        return {
            "page_no": None,
            "orig_page_no": None,
            "source": None,
            "source_pages": [],
            "unit_count": 0,
        }

    first_meta = None
    for unit in units:
        meta = unit.get("metadata", {})
        if meta:
            first_meta = meta
            break
    first_meta = first_meta or {}
   
    #找到所有unit的页信息
    all_source_pages = []
    seen = set()
    for unit in units:
        meta = unit.get("metadata", {}) or {}
        sp_list = meta.get("source_pages") or [{
            "page_no": meta.get("page_no"),
            "orig_page_no": meta.get("orig_page_no"),
            "source": meta.get("source"),
        }]
        for sp in sp_list:
            key = (sp.get("page_no"), sp.get("orig_page_no"), sp.get("source"))
            if key in seen:
                continue
            seen.add(key)
            all_source_pages.append({
                "page_no": sp.get("page_no"),
                "orig_page_no": sp.get("orig_page_no"),
                "source": sp.get("source"),
            })

    return {
        "page_no": first_meta.get("page_no"),
        "orig_page_no": first_meta.get("orig_page_no"),
        "source": first_meta.get("source"),
        "source_pages": all_source_pages,
        "unit_count": len(units),
    }


# =========================
# 4. 轻量角色识别
# =========================

#判断文本语义角色
def detect_body_role(sentence: str) -> str:
    """
    判断文本的语义角色，包括转折|举例|定义|枚举|陈述句
    """
    s = (sentence or "").strip()
    low = s.lower()

    transition_patterns = [
        r"^however\b",
        r"^in contrast\b",
        r"^on the other hand\b",
        r"^at the other extreme\b",
        r"^in between\b",
        r"^by contrast\b",
        r"^nevertheless\b",
        r"^thus\b",
        r"^meanwhile\b",
        r"^instead\b",
    ]
    if any(re.search(p, low) for p in transition_patterns):
        return "transition"

    example_patterns = [
        r"^for example\b",
        r"^for instance\b",
        r"^as an example\b",
        r"^consider\b",
        r"\bsuch as\b",
    ]
    if any(re.search(p, low) for p in example_patterns):
        return "example"

    definition_patterns = [
        r"\bis defined as\b",
        r"\brefers to\b",
        r"\bis the task of\b",
        r"\bis finding\b",
        r"\bthe term\b",
        r"\bmeans\b",
        r"\bis a model for\b",
    ]
    if any(re.search(p, low) for p in definition_patterns):
        return "definition"

    enumeration_patterns = [
        r"^(?:\(?\d+\)|\d+\.)\s+",
        r"^first\b",
        r"^second\b",
        r"^third\b",
    ]
    if any(re.search(p, low) for p in enumeration_patterns):
        return "enumeration"

    return "normal"



#判断是否以数字点、括号数字、字母编号、符号列表开头
def is_enumeration_item_text(text: str) -> bool:
    s = (text or "").strip()
    return bool(ENUMERATION_ITEM_RE.match(s))

#判断一段文本是“章节标题”还是“行内术语标题”
def classify_heading_node_type(text: str) -> str:
    """
    判断是章节标题还是术语标题，术语标题的特点是由大写字母开头的短句子
    """
    s = normalize_space(text)
    if not s:
        return "section_heading"

    if SECTION_HEADING_RE.match(s):
        return "section_heading"

    low = s.lower()
    if re.match(r'^(chapter|appendix|part|section)\b', low):
        return "section_heading"

    if INLINE_TERM_HEADING_RE.match(s) and count_words(s) <= 8:
        return "inline_term_heading"

    return "section_heading"


# =========================
# 5. 校验 / 节点构造
# =========================

#检测unit的信息是否完整
def validate_page_item(page_item: Dict[str, Any]) -> None:
    if not isinstance(page_item, dict):
        raise ValueError("每一页必须是 dict")
    if "units" not in page_item:
        raise ValueError("缺少 units")
    if not isinstance(page_item["units"], list):
        raise ValueError("units 必须是 list")
    for i, unit in enumerate(page_item["units"]):
        if not isinstance(unit, dict):
            raise ValueError(f"第 {i} 个 unit 不是 dict")
        if "page_content" not in unit:
            raise ValueError(f"第 {i} 个 unit 缺少 page_content")
        if "metadata" not in unit:
            raise ValueError(f"第 {i} 个 unit 缺少 metadata")
        if "unit_type" not in unit["metadata"]:
            raise ValueError(f"第 {i} 个 unit.metadata 缺少 unit_type")
        if "order_in_page" not in unit["metadata"]:
            raise ValueError(f"第 {i} 个 unit.metadata 缺少 order_in_page")

#创建节点
def build_page_nodes(page_item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    这里创建节点,规则是脚注忽略,类型是body则分割文本一个句子对应一个节点.
    其余的直接作为一个节点
    对于类型是heading、以及body的,会额外进一步得到详细类型
    """
    validate_page_item(page_item)

    units = sorted(
        page_item["units"],
        key=lambda u: u["metadata"].get("order_in_page", 10 ** 9)
    )

    nodes: List[Dict[str, Any]] = []
    global_node_idx = 0

    for unit_idx, unit in enumerate(units):
        unit_type = unit["metadata"]["unit_type"]
        #这里忽略脚注信息，即脚注不会被直接放入文档块的文本中
        if unit_type == "footnote":
            continue

        text = (unit.get("page_content") or "").strip()
        if not text:
            continue

        if unit_type == "body":
            sentences = split_into_sentences(text) or [text]
            for sent_idx, sent in enumerate(sentences):
                node_type = "list_item" if is_enumeration_item_text(sent) else "sentence"
                nodes.append({
                    "node_index": global_node_idx,
                    "node_type": node_type,
                    "unit_type": "body",
                    "text": sent,
                    "unit_index": unit_idx,
                    "order_in_page": unit["metadata"]["order_in_page"],
                    "sentence_index_in_unit": sent_idx,
                    "role": detect_body_role(sent),
                    "unit_metadata": unit["metadata"],
                })
                global_node_idx += 1
        else:
            node_type = unit_type
            if unit_type == "heading":
                node_type = classify_heading_node_type(text)
            nodes.append({
                "node_index": global_node_idx,
                "node_type": node_type,
                "unit_type": unit_type,
                "text": text,
                "unit_index": unit_idx,
                "order_in_page": unit["metadata"]["order_in_page"],
                "sentence_index_in_unit": 0,
                "role": unit_type,
                "unit_metadata": unit["metadata"],
            })
            global_node_idx += 1

    return nodes


# =========================
# 6. Embedding
# =========================

#将一组文本转化成归一化向量
def get_dense_embeddings(texts: List[str], embedding_handler) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1024), dtype=np.float32)
    emb = embedding_handler(texts)
    dense = np.array(emb["dense"], dtype=np.float32)
    return l2_normalize(dense)

#给每个节点增加对应文本向量信息
def attach_node_embeddings(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    texts = [n["text"] for n in nodes]
    vecs = get_dense_embeddings(texts, embedding_handler)
    for i, n in enumerate(nodes):
        n["embedding"] = vecs[i]
    return nodes

#计算某一连续节点平均语义向量
def get_window_centroid(nodes: List[Dict[str, Any]], start: int, end: int) -> np.ndarray:
    valid = []
    for i in range(start, end):
        if 0 <= i < len(nodes):
            valid.append(nodes[i]["embedding"])
    dim = len(nodes[0]["embedding"]) if nodes else 1024
    return safe_mean_embedding(valid, dim=dim)

#利用句子类型打分
def structure_score(left: Dict[str, Any], right: Dict[str, Any]) -> float:
    return STRUCTURE_SCORE_MAP.get((left["node_type"], right["node_type"]), 0.0)

#利用语义角色打分
def role_score(left: Dict[str, Any], right: Dict[str, Any]) -> float:
    allowed = {"sentence", "list_item"}
    if left["node_type"] not in allowed or right["node_type"] not in allowed:
        return 0.0

    rr = right.get("role", "normal")
    if rr == "transition":
        return 0.38
    if rr == "definition":
        return 0.18
    if rr == "example":
        return -0.10
    if rr == "enumeration":
        return 0.16
    return 0.0

#额外处理同属于一个unit的文本
def unit_continuity_score(left: Dict[str, Any], right: Dict[str, Any]) -> float:
    if (
        left["unit_type"] == "body"
        and right["unit_type"] == "body"
        and left["unit_index"] == right["unit_index"]
    ):
        if left["node_type"] == "list_item" and right["node_type"] == "list_item":
            return 0.06
        return -0.03
    return 0.0

#对其中一个类型是标题的额外处理
def should_force_cut(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    """
    某些边界直接强制切开，不交给阈值博弈。
    """
    if left["node_type"] == "caption" and right["node_type"] in {
        "sentence", "list_item", "section_heading", "inline_term_heading"
    }:
        return True

    if left["node_type"] in {"sentence", "list_item"} and right["node_type"] == "section_heading":
        return True

    if left["node_type"] == "list_item" and right["node_type"] == "caption":
        return True

    return False

#利用上面的规则进行打分
def compute_initial_boundary_scores(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    利用打分规则对相邻节点的边界打分,获得所有边界的分数
    """
    if len(nodes) < 2:
        return []

    raw_rows = []
    for i in range(len(nodes) - 1):
        left = nodes[i]
        right = nodes[i + 1]

        sim_adj = cosine_sim(left["embedding"], right["embedding"])
        semantic_adj = 1.0 - sim_adj

        left_centroid = get_window_centroid(nodes, max(0, i - 1), i + 1)
        right_centroid = get_window_centroid(nodes, i + 1, min(len(nodes), i + 3))
        semantic_window = 1.0 - cosine_sim(left_centroid, right_centroid)

        left_text = " ".join(n["text"] for n in nodes[max(0, i - 1): i + 1])
        right_text = " ".join(n["text"] for n in nodes[i + 1: min(len(nodes), i + 3)])
        lexical_shift = lexical_shift_score(left_text, right_text)

        raw_rows.append({
            "boundary_index": i,
            "left_node_index": i,
            "right_node_index": i + 1,
            "semantic_adj": semantic_adj,
            "semantic_window": semantic_window,
            "lexical_shift": lexical_shift,
            "structure_score": structure_score(left, right),
            "role_score": role_score(left, right),
            "continuity_score": unit_continuity_score(left, right),
            "hard_cut": (left["node_type"], right["node_type"]) in HARD_CUT_STRUCTURE_PAIRS,
            "force_cut": should_force_cut(left, right),
        })

    adj_scaled = robust_scale([r["semantic_adj"] for r in raw_rows])
    window_scaled = robust_scale([r["semantic_window"] for r in raw_rows])
    lexical_scaled = robust_scale([r["lexical_shift"] for r in raw_rows])

    boundaries = []
    for idx, r in enumerate(raw_rows):
        semantic_score = 0.50 * adj_scaled[idx] + 0.35 * window_scaled[idx] + 0.15 * lexical_scaled[idx]
        final_score = (
            0.72 * semantic_score
            + 0.18 * r["structure_score"]
            + 0.07 * r["role_score"]
            + 0.03 * r["continuity_score"]
        )

        if r["hard_cut"]:
            final_score += 0.45
        if r["force_cut"]:
            final_score += 1.20

        boundaries.append({
            **r,
            "semantic_score": float(semantic_score),
            "initial_score": float(final_score),
            "propagated_score": float(final_score),
        })

    return boundaries


# =========================
# 7. 轻传播 + 峰值增强
# =========================

#对边界分数做平滑处理，保留真峰，压掉假峰
def propagate_boundary_scores(
    boundaries: List[Dict[str, Any]],
    num_steps: int = 1,
    lambda_prop: float = 0.18,
    peak_bonus: float = 0.12,
) -> List[Dict[str, Any]]:
    if not boundaries:
        return boundaries

    scores = np.array([b["initial_score"] for b in boundaries], dtype=np.float32)
    
    #平滑噪声
    for _ in range(num_steps):
        new_scores = scores.copy()
        for i in range(len(scores)):
            neighbors = []
            weights = []
            if i - 1 >= 0:
                neighbors.append(scores[i - 1])
                weights.append(0.75)
            if i + 1 < len(scores):
                neighbors.append(scores[i + 1])
                weights.append(0.75)
            if neighbors:
                neighbor_mean = float(np.average(neighbors, weights=weights))
                new_scores[i] = (1.0 - lambda_prop) * scores[i] + lambda_prop * neighbor_mean
        scores = new_scores

    #强化高峰
    for i in range(len(scores)):
        left = scores[i - 1] if i - 1 >= 0 else scores[i] - 0.1
        right = scores[i + 1] if i + 1 < len(scores) else scores[i] - 0.1
        local_peak_strength = scores[i] - max(left, right)
        if local_peak_strength > 0:
            scores[i] += peak_bonus * local_peak_strength

    for i, b in enumerate(boundaries):
        b["propagated_score"] = float(scores[i])

    return boundaries

# =========================
# 8. 选边界
# =========================

#判断是否比相邻边界分数高
def is_local_peak(boundaries: List[Dict[str, Any]], idx: int) -> bool:
    curr = boundaries[idx]["propagated_score"]
    left = boundaries[idx - 1]["propagated_score"] if idx - 1 >= 0 else -1e9
    right = boundaries[idx + 1]["propagated_score"] if idx + 1 < len(boundaries) else -1e9
    return curr >= left and curr >= right

#真正选择边界的函数，利用边界分数确定真实边界
def choose_cut_boundaries(
    nodes: List[Dict[str, Any]],
    boundaries: List[Dict[str, Any]],
    base_threshold: float = 0.10,
    percentile_threshold: float = 60.0,
    min_gap: int = 1,
) -> List[int]:
    """
    返回List[i],表示在node[i]和node[i+1]之间分
    """
    if not boundaries:
        return []

    scores = np.array([b["propagated_score"] for b in boundaries], dtype=np.float32)
    #将边界分数的的60％位置的分数，和base_threshold取得一个真实的边界分数
    dynamic_threshold = max(base_threshold, float(np.percentile(scores, percentile_threshold)))
    
    #获取候选
    candidate_idxs = []
    for i, b in enumerate(boundaries):
        score = b["propagated_score"]
        hard_cut = b.get("hard_cut", False)
        force_cut = b.get("force_cut", False)

        left = nodes[i]
        right = nodes[i + 1]

        # heading 默认附着后文
        if left["node_type"] == "section_heading" and right["unit_type"] == "body":
            continue

        if left["node_type"] == "inline_term_heading" and right["unit_type"] == "body":
            continue

        # caption 后接正文：一定切
        if left["node_type"] == "caption" and right["unit_type"] == "body":
            candidate_idxs.append(i)
            continue

        candidate = force_cut or hard_cut or (score >= dynamic_threshold and is_local_peak(boundaries, i))
        if candidate:
            candidate_idxs.append(i)
    
    #消除候选冲突
    selected = []
    for idx in candidate_idxs:
        if not selected:
            selected.append(idx)
            continue
        #当两个候选很近的时候，处理候选
        if idx - selected[-1] <= min_gap:
            prev = selected[-1]

            prev_force = boundaries[prev].get("force_cut", False)
            curr_force = boundaries[idx].get("force_cut", False)
            if curr_force and not prev_force:
                selected[-1] = idx
                continue
            if prev_force and not curr_force:
                continue

            if boundaries[idx]["propagated_score"] > boundaries[prev]["propagated_score"]:
                selected[-1] = idx
        else:
            selected.append(idx)

    return selected


# =========================
# 9. 重建 chunk
# =========================

#处理图表caption和body合并的情况
def split_mixed_caption_chunk(chunk_nodes: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """
    如果一个 chunk 里以 caption 开头，后面跟了很多 body 节点，
    则把 caption 单独拆出来，避免图注吞正文。
    """
    if not chunk_nodes:
        return []

    if chunk_nodes[0]["node_type"] != "caption":
        return [chunk_nodes]

    if len(chunk_nodes) == 1:
        return [chunk_nodes]

    rest_has_body = any(n["unit_type"] == "body" for n in chunk_nodes[1:])
    if not rest_has_body:
        return [chunk_nodes]

    return [[chunk_nodes[0]], chunk_nodes[1:]]


def rebuild_chunks_from_cuts(
    page_item: Dict[str, Any],
    nodes: List[Dict[str, Any]],
    cuts: List[int],
) -> List[Dict[str, Any]]:
    """
    利用真实边界,进行节点分割合并,返回最终chunk
    """
    page_meta = extract_page_metadata(page_item)

    if not nodes:
        return []

    cut_set = set(cuts)
    spans = []
    start = 0
    for i in range(len(nodes) - 1):
        if i in cut_set:
            spans.append((start, i))
            start = i + 1
    spans.append((start, len(nodes) - 1))

    results = []
    for _, (s, e) in enumerate(spans):
        raw_chunk_nodes = nodes[s:e + 1]
        sub_chunks = split_mixed_caption_chunk(raw_chunk_nodes)
        
        #合并节点
        for chunk_nodes in sub_chunks:
            chunk_text = "\n".join(n["text"] for n in chunk_nodes if n["text"]).strip()
            if not chunk_text:
                continue

            unit_types = [n["unit_type"] for n in chunk_nodes]
            node_types = [n["node_type"] for n in chunk_nodes]

            parent_heading = None
            for n in chunk_nodes:
                if n["node_type"] in {"section_heading", "inline_term_heading"}:
                    parent_heading = n["text"]
                    break

            metadata = {
                "chunk_index": len(results),
                "chunk_type": "semantic_chunk",
                "page_no": page_meta.get("page_no"),
                "orig_page_no": page_meta.get("orig_page_no"),
                "source": page_meta.get("source"),
                "source_pages": page_meta.get("source_pages", []),
                "start_node_index": chunk_nodes[0]["node_index"],
                "end_node_index": chunk_nodes[-1]["node_index"],
                "node_count": len(chunk_nodes),
                "unit_types": unit_types,
                "node_types": node_types,
                "parent_heading": parent_heading,
                "contains_heading": any(t in {"section_heading", "inline_term_heading"} for t in node_types),
                "heading_types": [t for t in node_types if t in {"section_heading", "inline_term_heading"}],
                "contains_caption": "caption" in node_types,
                "word_count": count_words(chunk_text),
            }

            caption_node = next((n for n in chunk_nodes if n["node_type"] == "caption"), None)
            if caption_node:
                um = caption_node.get("unit_metadata", {}) or {}
                metadata["region_id"] = um.get("region_id")
                metadata["region_type"] = um.get("region_type")
                metadata["caption_label"] = um.get("caption_label")
                metadata["image_path"] = um.get("image_path")

            results.append({
                "page_content": chunk_text,
                "metadata": metadata,
            })

    return results


# =========================
# 10. 主流程
# =========================
def semantic_split_page_with_boundary_propagation(
    page_item: Dict[str, Any],
    propagation_steps: int = 1,
    propagation_lambda: float = 0.18,
    cut_percentile: float = 60.0,
    base_threshold: float = 0.10,
    return_debug: bool = False,
):
    nodes = build_page_nodes(page_item)
    nodes = attach_node_embeddings(nodes)

    boundaries = compute_initial_boundary_scores(nodes)
    boundaries = propagate_boundary_scores(
        boundaries,
        num_steps=propagation_steps,
        lambda_prop=propagation_lambda,
    )

    cuts = choose_cut_boundaries(
        nodes=nodes,
        boundaries=boundaries,
        base_threshold=base_threshold,
        percentile_threshold=cut_percentile,
        min_gap=1,
    )

    chunks = rebuild_chunks_from_cuts(page_item, nodes, cuts)

    if not return_debug:
        return chunks

    debug_nodes = []
    for n in nodes:
        debug_nodes.append({
            "node_index": n["node_index"],
            "node_type": n["node_type"],
            "unit_type": n["unit_type"],
            "role": n["role"],
            "text": n["text"],
        })

    debug_boundaries = []
    cut_set = set(cuts)
    for b in boundaries:
        debug_boundaries.append({
            "boundary_index": b["boundary_index"],
            "left_node_index": b["left_node_index"],
            "right_node_index": b["right_node_index"],
            "semantic_adj": b["semantic_adj"],
            "semantic_window": b["semantic_window"],
            "lexical_shift": b["lexical_shift"],
            "semantic_score": b["semantic_score"],
            "structure_score": b["structure_score"],
            "role_score": b["role_score"],
            "continuity_score": b["continuity_score"],
            "initial_score": b["initial_score"],
            "propagated_score": b["propagated_score"],
            "hard_cut": b.get("hard_cut", False),
            "force_cut": b.get("force_cut", False),
            "is_cut": b["boundary_index"] in cut_set,
        })

    return {
        "chunks": chunks,
        "debug": {
            "nodes": debug_nodes,
            "boundaries": debug_boundaries,
            "cuts": cuts,
        }
    }


def semantic_split_pages_to_chunks(
    pages: List[Dict[str, Any]],
    propagation_steps: int = 1,
    propagation_lambda: float = 0.18,
    cut_percentile: float = 60.0,
    base_threshold: float = 0.10,
) -> List[Dict[str, Any]]:
    all_chunks: List[Dict[str, Any]] = []

    for page_idx, page_item in enumerate(pages):
        page_chunks = semantic_split_page_with_boundary_propagation(
            page_item=page_item,
            propagation_steps=propagation_steps,
            propagation_lambda=propagation_lambda,
            cut_percentile=cut_percentile,
            base_threshold=base_threshold,
            return_debug=False,
        )

        for local_idx, chunk in enumerate(page_chunks):
            meta = chunk.setdefault("metadata", {})
            meta["page_index_in_input"] = page_idx
            meta["chunk_index_in_page"] = local_idx
            meta["global_chunk_index"] = len(all_chunks)

            unique_base = (
                f"{meta.get('source', '')}|"
                f"{meta.get('orig_page_no', '')}|"
                f"{meta.get('page_no', '')}|"
                f"{local_idx}|"
                f"{chunk.get('page_content', '')[:200]}"
            )
            meta["unique_id"] = md5_text(unique_base)

            all_chunks.append(chunk)

    return all_chunks


# =========================
# 11. 数据读取 / 调试
# =========================
def load_pages_from_json_file(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    if not raw:
        return []

    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except Exception:
        pass

    pages = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pages.append(json.loads(line))
    return pages


def test_boundary_propagation_first_two_pages(file_path: str = merged_docs) -> List[Dict[str, Any]]:
    pages = load_pages_from_json_file(file_path)
    first_two_pages = pages[:2]

    all_chunks = semantic_split_pages_to_chunks(
        pages=first_two_pages,
        propagation_steps=1,
        propagation_lambda=0.18,
        cut_percentile=60.0,
        base_threshold=0.10,
    )

    print(f"总 chunk 数: {len(all_chunks)}")
    for i, chunk in enumerate(all_chunks[:20]):
        meta = chunk["metadata"]
        print(f"===== Chunk {i} =====")
        print(f"source={meta.get('source')}")
        print(f"orig_page_no={meta.get('orig_page_no')}")
        print(f"page_no={meta.get('page_no')}")
        print(f"source_pages={meta.get('source_pages')}")
        print(f"word_count={meta.get('word_count')}")
        print(chunk["page_content"])
        print()

    return all_chunks


if __name__ == "__main__":
    test_boundary_propagation_first_two_pages(test_docs)