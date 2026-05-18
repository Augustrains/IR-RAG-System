import re
import copy
import hashlib
import os
from typing import List, Dict, Any, Tuple, Optional
import json

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.chunking.semantic_chunker import semantic_split_pages_to_chunks
from src.path import merged_docs


INPUT_PATH = merged_docs
OUTPUT_DIR = "/root/autodl-tmp/IR-RAG-System/data/processed_docs/split"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "first10_pages_split.jsonl")
SEMANTIC_CHUNKS_FILE = os.path.join(OUTPUT_DIR, "first10_pages_semantic_chunks.jsonl")
save_Figure=os.path.join(OUTPUT_DIR, "figure_first10_pages_semantic_chunks.jsonl")

# =========================================================
# 基础配置
# =========================================================
_max_parent_size = 512
_child_chunk_size = 256
_child_chunk_overlap = 50


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def md5_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def split_first_sentence(text: str) -> str:
    text = normalize_text(text)
    if not text:
        return ""
    parts = re.split(r'(?<=[\.\!\?。！？])\s+', text)
    parts = [p.strip() for p in parts if p.strip()]
    if parts:
        return parts[0]
    return text


def shorten_text(text: str, max_len: int = 150) -> str:
    text = normalize_text(text)
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "..."


#统一图表写法的格式
def normalize_caption_label(label: str) -> str:
    """
    统一格式为 Figure x.x , Table x
    """
    s = normalize_text(label)
    if not s:
        return ""

    m = re.match(r'^(Figure|Fig\.?|Table)\s+(\d+(?:\.\d+)*)$', s, flags=re.I)
    if not m:
        return ""

    kind = m.group(1).lower()
    num = m.group(2)

    if kind.startswith("fig"):
        return f"Figure {num}"
    if kind == "table":
        return f"Table {num}"
    return ""

#去除caption中最开头的部分，提取纯语义标题
def remove_caption_label_prefix(caption_text: str, caption_label: str) -> str:
    """
    对于如 ◮Figure 1.5 xxxxxxx,返回 xxxxxxxx,去除前面◮Figure这部分
    """
    text = normalize_text(caption_text)
    label = normalize_caption_label(caption_label)
    if not text:
        return ""

    text = re.sub(r'^[◮•\-\s]+', '', text).strip()

    if label:
        num = label.split()[-1]
        pattern = rf'^(?:Figure|Fig\.?|Table)\s+{re.escape(num)}\s*[:.\-]?\s*'
        text = re.sub(pattern, '', text, flags=re.I).strip()

    return text

#得到caption的第一句文本,用于给引用该图表的正文增加一个解释
def make_short_caption(caption_text: str, caption_label: str, max_len: int = 150) -> str:
    cleaned = remove_caption_label_prefix(caption_text, caption_label)
    first_sent = split_first_sentence(cleaned)
    return shorten_text(first_sent, max_len=max_len)

#建立Figure索引
def build_figure_index_from_merge_docs(page_items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    从 merge_docs / merged pages 中构建图表索引。
    要求图表 caption 信息已经存在于每页的 units 中。
    """
    figure_index: Dict[str, Dict[str, Any]] = {}

    for page_item in page_items:
        units = page_item.get("units", []) or []

        page_meta = page_item.get("page_metadata", {}) or page_item.get("metadata", {}) or {}
        page_source = page_meta.get("source", page_item.get("source"))
        page_no = page_meta.get("page_no", page_item.get("page_no"))
        orig_page_no = page_meta.get("orig_page_no", page_item.get("orig_page_no"))

        for unit in units:
            meta = unit.get("metadata", {}) or {}
            if meta.get("unit_type") != "caption":
                continue

            caption_label = normalize_caption_label(meta.get("caption_label", ""))
            caption_text = normalize_text(
                unit.get("page_content", "") or meta.get("caption_text", "")
            )
            image_path = normalize_text(meta.get("image_path", ""))

            if not caption_label or not caption_text:
                continue

            if caption_label not in figure_index:
                figure_index[caption_label] = {
                    "caption_label": caption_label,
                    "caption_text": caption_text,
                    "short_caption": make_short_caption(caption_text, caption_label),
                    "image_path": image_path,
                    "page_no": meta.get("page_no", page_no),
                    "orig_page_no": meta.get("orig_page_no", orig_page_no),
                    "source": meta.get("source", page_source),
                }
            else:
                # 已存在时，补全缺失字段
                old = figure_index[caption_label]
                if not old.get("image_path") and image_path:
                    old["image_path"] = image_path
                if not old.get("caption_text") and caption_text:
                    old["caption_text"] = caption_text
                if not old.get("short_caption") and caption_text:
                    old["short_caption"] = make_short_caption(caption_text, caption_label)

    return figure_index


STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "by",
    "with", "as", "at", "from", "is", "are", "was", "were", "be",
}

#获取文本的单词
def tokenize_for_overlap(text: str) -> List[str]:
    text = normalize_text(text).lower()
    if not text:
        return []

    tokens = re.findall(r'[a-zA-Z]+(?:-[a-zA-Z]+)?|\d+(?:\.\d+)*', text)

    # 去掉停用词 & 太短的词（减少误判）
    return [
        tok for tok in tokens
        if tok not in STOPWORDS and (len(tok) > 1 or tok.isdigit())
    ]

#最长连续匹配
def longest_contiguous_match_len(a: List[str], b: List[str]) -> int:
    """
    计算两个 token 序列的最长连续公共子串长度（token 级别）
    """
    if not a or not b:
        return 0

    dp = [0] * (len(b) + 1)
    best = 0

    for i in range(1, len(a) + 1):
        new_dp = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                new_dp[j] = dp[j - 1] + 1
                if new_dp[j] > best:
                    best = new_dp[j]
        dp = new_dp

    return best

#判断caption和文本连续单词重叠程度,用于确定是引用caption还是就是caption本身
def chunk_contains_more_than_half_caption(
    chunk_text: str,
    caption_text: str,
    caption_label: str,
    threshold: float = 0.5,
) -> bool:
    """
    只基于“最长连续片段覆盖率”的判断
    """

    chunk_norm = normalize_text(chunk_text).lower()
    caption_body = remove_caption_label_prefix(caption_text, caption_label)
    caption_norm = normalize_text(caption_body).lower()

    if not chunk_norm or not caption_norm:
        return False

    # 直接子串匹配
    if caption_norm in chunk_norm:
        return True

    caption_tokens = tokenize_for_overlap(caption_norm)
    chunk_tokens = tokenize_for_overlap(chunk_norm)

    if not caption_tokens or not chunk_tokens:
        return False
    
    #最长连续匹配
    contig_len = longest_contiguous_match_len(caption_tokens, chunk_tokens)
    ratio = contig_len / len(caption_tokens)

    return ratio >= threshold

#结构化图表信息
def build_figure_ref_item(info: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "caption_label": info.get("caption_label", ""),
        "caption_text": info.get("caption_text", ""),
        "image_path": info.get("image_path", ""),
        "page_no": info.get("page_no"),
        "orig_page_no": info.get("orig_page_no"),
        "source": info.get("source"),
    }

REF_PATTERN = re.compile(
    r'\b(Figure|Fig\.?|Table)\s+(\d+(?:\.\d+)*)\b',
    flags=re.I
)

#将标准的图表索引扩展成多种匹配格式
def build_patterns_for_ref(ref_label: str) -> List[str]:
    m = re.match(r'^(Figure|Table)\s+(\d+(?:\.\d+)*)$', ref_label, flags=re.I)
    if not m:
        return []

    kind = m.group(1).lower()
    num = m.group(2)

    if kind == "figure":
        return [
            rf'\bFigure\s+{re.escape(num)}\b',
            rf'\bFig\.\s*{re.escape(num)}\b',
            rf'\bFig\s+{re.escape(num)}\b',
        ]
    return [rf'\bTable\s+{re.escape(num)}\b']

#从正文中找到Figure/Table 引用，并且插入对应一个图表说明
def inject_caption_inline_for_ref(
    text: str,
    ref_label: str,
    short_caption: str,
    max_replacements_per_ref: int = 2,
) -> str:
    if not text or not short_caption:
        return text

    new_text = text
    replace_count = 0
    patterns = build_patterns_for_ref(ref_label)

    for pattern in patterns:
        def repl(match):
            nonlocal replace_count
            if replace_count >= max_replacements_per_ref:
                return match.group(0)

            matched_ref = match.group(0)
            tail = match.string[match.end(): match.end() + 160]
            if re.match(r'^\s*\([^)]{1,160}\)', tail):
                return matched_ref

            replace_count += 1
            return f"{matched_ref} ({short_caption})"
        
        #匹配+替换
        new_text = re.sub(pattern, repl, new_text, flags=re.I)
        if replace_count >= max_replacements_per_ref:
            break

    return new_text

#从正文中提取图表引用,用于后续给正文增加对应图表信息
def extract_candidate_refs(text: str) -> List[str]:
    """
    提取正文中例如 Figure 1.2 , Tabel 1这样的字段
    """
    text = normalize_text(text)
    if not text:
        return []

    refs = []
    seen = set()

    for m in REF_PATTERN.finditer(text):
        kind = m.group(1).lower()
        num = m.group(2)

        if kind.startswith("fig"):
            ref = f"Figure {num}"
        else:
            ref = f"Table {num}"

        if ref not in seen:
            seen.add(ref)
            refs.append(ref)

    return refs

#处理文本，增加对应图表信息
def enrich_chunk_with_figures(
    chunk_text: str,
    chunk_meta: Dict[str, Any],
    figure_index: Dict[str, Dict[str, Any]],
    max_refs_per_chunk: int = 8,
) -> Tuple[str, Dict[str, Any]]:
    """
    针对chunk_text,首先得到caption的引用,利用figure_index获取详细信息,判断重叠次数,返回增强后的文本
    """
    refs = extract_candidate_refs(chunk_text)
    if not refs:
        return chunk_text, {"figure_refs": []}

    refs = refs[:max_refs_per_chunk]
    new_text = chunk_text
    figure_refs = []
    seen_labels = set()

    for ref in refs:
        info = figure_index.get(ref)
        if not info or ref in seen_labels:
            continue
        seen_labels.add(ref)

        caption_text = normalize_text(info.get("caption_text", ""))
        short_caption = normalize_text(info.get("short_caption", ""))

        # 判断当前 chunk 是否已经覆盖了 caption 超过一半内容
        has_more_than_half_caption = chunk_contains_more_than_half_caption(
            chunk_text=new_text,
            caption_text=caption_text,
            caption_label=ref,
            threshold=0.5,
        )

        # 不足一半，才做注入
        if short_caption and not has_more_than_half_caption:
            new_text = inject_caption_inline_for_ref(
                new_text,
                ref_label=ref,
                short_caption=short_caption,
                max_replacements_per_ref=2,
            )

        figure_refs.append(build_figure_ref_item(info))

    return normalize_text(new_text), {"figure_refs": figure_refs}

#从chunk中提取页信息
def get_chunk_pages(chunk_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    pages = chunk_meta.get("source_pages")
    if isinstance(pages, list) and pages:
        results = []
        for p in pages:
            if isinstance(p, dict):
                results.append({
                    "source": p.get("source", chunk_meta.get("source")),
                    "page_no": p.get("page_no"),
                    "orig_page_no": p.get("orig_page_no"),
                })
        if results:
            return results

    return [{
        "source": chunk_meta.get("source"),
        "page_no": chunk_meta.get("page_no"),
        "orig_page_no": chunk_meta.get("orig_page_no"),
    }]



#建立脚注检索
def build_footnote_index(page_items: List[Dict[str, Any]]) -> Dict[Tuple[Any, Any, Any], List[Dict[str, Any]]]:
    footnote_index: Dict[Tuple[Any, Any, Any], List[Dict[str, Any]]] = {}

    for page_item in page_items:
        units = page_item.get("units", []) or []
        for unit in units:
            meta = unit.get("metadata", {}) or {}
            if meta.get("unit_type") != "footnote":
                continue

            footnote_text = normalize_text(unit.get("page_content", ""))
            if not footnote_text:
                continue

            key = (meta.get("source"), meta.get("page_no"), meta.get("orig_page_no"))
            footnote_index.setdefault(key, []).append({
                "footnote_text": footnote_text,
                "page_no": meta.get("page_no"),
                "orig_page_no": meta.get("orig_page_no"),
                "source": meta.get("source"),
            })

    return footnote_index

#对脚注和正文进行分词
def tokenize_for_footnote_match(text: str) -> List[str]:
    """
    为脚注-句子匹配准备的轻量分词：
    - 只保留英文/数字
    - 小写
    - 去掉常见停用词
    """
    text = normalize_text(text).lower()
    if not text:
        return []

    tokens = re.findall(r"[a-zA-Z0-9]+", text)

    stopwords = {
        "the", "a", "an", "and", "or", "but", "if", "then", "else",
        "of", "to", "in", "on", "for", "with", "by", "as", "at", "from",
        "is", "are", "was", "were", "be", "been", "being",
        "this", "that", "these", "those",
        "it", "its", "he", "she", "they", "them", "their",
        "we", "our", "you", "your", "i", "my",
        "do", "does", "did", "done",
        "have", "has", "had",
        "not", "no", "yes",
        "can", "could", "may", "might", "must", "should", "would",
        "than", "such", "into", "about", "over", "under",
        "also", "however", "there", "here", "thus"
    }

    return [tok for tok in tokens if tok not in stopwords and len(tok) >= 2]

#计算脚注和文本相似度
def compute_footnote_sentence_match_score(sentence: str, footnote_text: str) -> float:
    """
    计算脚注和句子的匹配分数。
    当前采用：
    - 归一化后 token overlap
    - 稀有词命中加权
    - 子串弱奖励
    """
    sentence = normalize_text(sentence)
    footnote_text = normalize_text(footnote_text)

    if not sentence or not footnote_text:
        return 0.0

    sent_tokens = tokenize_for_footnote_match(sentence)
    foot_tokens = tokenize_for_footnote_match(footnote_text)

    if not sent_tokens or not foot_tokens:
        return 0.0

    sent_set = set(sent_tokens)
    foot_set = set(foot_tokens)

    overlap = sent_set & foot_set
    if not overlap:
        return 0.0

    # 基础重合率：脚注中的词有多少能在句子中找到
    recall_like = len(overlap) / max(len(foot_set), 1)

    # 精确率倾向：句子中命中了多少脚注词
    precision_like = len(overlap) / max(len(sent_set), 1)

    # 稀有词加权：长词通常更像术语，更有区分度
    rare_bonus = 0.0
    for tok in overlap:
        if len(tok) >= 6:
            rare_bonus += 0.08
        elif len(tok) >= 4:
            rare_bonus += 0.04

    # 子串弱奖励
    substring_bonus = 0.0
    for tok in overlap:
        if len(tok) >= 5:
            substring_bonus = 0.08
            break

    score = 0.65 * recall_like + 0.25 * precision_like + rare_bonus + substring_bonus
    return min(score, 1.0)

#分割文本，用于判断脚注相似度
def split_into_sentences(text: str) -> List[str]:
    """
    将文本切分为句子（适用于英文学术 PDF，如 IR book）

    特点：
    - 避免在 e.g. / i.e. / Fig. / Eq. 等缩写处错误断句
    - 支持引号、括号后的句号
    - 适配数字开头句子（如 1. Formally, ...）
    """

    if not text:
        return []

    # ========= 1. 预清洗 =========
    text = re.sub(r'\s+', ' ', text.strip())

    # ========= 2. 保护常见缩写 =========
    abbreviations = [
        "e.g.", "i.e.", "etc.", "Fig.", "Figs.", "Eq.", "Eqs.",
        "Dr.", "Mr.", "Mrs.", "Ms.",
        "U.S.", "U.K.",
        "vs.", "cf.",
        "No.", "al."
    ]

    abbr_map = {}
    for i, abbr in enumerate(abbreviations):
        placeholder = f"__ABBR_{i}__"
        abbr_map[placeholder] = abbr
        text = text.replace(abbr, placeholder)

    # ========= 3. 核心断句规则 =========
    # 在 . ! ? 后面断句
    # 条件：
    #   后面是空格 + 大写字母 / 数字 / 引号
    sentence_split_pattern = re.compile(
        r'(?<=[\.\!\?])\s+(?=[A-Z0-9"\'])'
    )

    sentences = sentence_split_pattern.split(text)

    # ========= 4. 还原缩写 =========
    restored_sentences = []
    for sent in sentences:
        for placeholder, abbr in abbr_map.items():
            sent = sent.replace(placeholder, abbr)
        sent = sent.strip()
        if sent:
            restored_sentences.append(sent)

    return restored_sentences

#给文档添加脚注信息
def enrich_chunk_with_footnotes(
    chunk_text: str,
    chunk_meta: Dict[str, Any],
    footnote_index: Dict[Tuple[Any, Any, Any], List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """
    对当前 chunk 进行脚注补充：
    - 只处理 chunk 所属页上的脚注
    - 对每条脚注，在当前 chunk 内做句子级匹配
    - 命中明显才加入
    - 每条脚注只加入一次
    """
    chunk_text = normalize_text(chunk_text)
    if not chunk_text:
        return {"related_footnotes": []}
    
    chunk_pages = get_chunk_pages(chunk_meta)
    if not chunk_pages:
        return {"related_footnotes": []}
  
    sentences = split_into_sentences(chunk_text)
    if not sentences:
        sentences = [chunk_text]

    related_footnotes = []
    seen_texts = set()

    # 可按需要调节
    min_match_score = 0.35

    for p in chunk_pages:
        #从 footnote_index 中拿到“该页的所有脚注”
        key = (p.get("source"), p.get("page_no"), p.get("orig_page_no"))
        page_footnotes = footnote_index.get(key, [])

        for footnote_item in page_footnotes:
            footnote_text = normalize_text(footnote_item.get("footnote_text", ""))
            if not footnote_text:
                continue

            # 同一条脚注只放一次
            if footnote_text in seen_texts:
                continue

            best_score = 0.0
            for sent in sentences:
                score = compute_footnote_sentence_match_score(sent, footnote_text)
                if score > best_score:
                    best_score = score

            if best_score >= min_match_score:
                related_footnotes.append({
                    "footnote_text": footnote_text,
                })
                seen_texts.add(footnote_text)

    return {"related_footnotes": related_footnotes}

#构建图表索引+脚注索引+语义分块+插入图表、脚注内容
def semantic_split_pages_to_enriched_chunks(
    pages: List[Dict[str, Any]],
    propagation_steps: int = 1,
    propagation_lambda: float = 0.18,
    cut_percentile: float = 60.0,
    base_threshold: float = 0.10,
) -> List[Dict[str, Any]]:
    """
    语义分割部分
    """
    figure_index = build_figure_index_from_merge_docs(pages)

    footnote_index = build_footnote_index(pages)

    raw_chunks = semantic_split_pages_to_chunks(
        pages=pages,
        propagation_steps=propagation_steps,
        propagation_lambda=propagation_lambda,
        cut_percentile=cut_percentile,
        base_threshold=base_threshold,
    )
    
    #保存语义分块结果
    # save_dict_list_to_jsonl(raw_chunks, SEMANTIC_CHUNKS_FILE)
    # print(f"语义分块的结果已保存到:{SEMANTIC_CHUNKS_FILE}")

    enhanced_chunks: List[Dict[str, Any]] = []

    for chunk in raw_chunks:
        chunk = copy.deepcopy(chunk)
        chunk_text = normalize_text(chunk.get("page_content", ""))
        chunk_meta = chunk.setdefault("metadata", {})

        if not chunk_text:
            continue

        new_text, figure_meta = enrich_chunk_with_figures(
            chunk_text=chunk_text,
            chunk_meta=chunk_meta,
            figure_index=figure_index,
            max_refs_per_chunk=8,
        )

        footnote_meta = enrich_chunk_with_footnotes(
            chunk_text=new_text,
            chunk_meta=chunk_meta,
            footnote_index=footnote_index,
        )

        chunk["page_content"] = new_text
        chunk_meta["figure_refs"] = figure_meta.get("figure_refs", [])
        chunk_meta["related_footnotes"] = footnote_meta.get("related_footnotes", [])

        enhanced_chunks.append(chunk)

    return enhanced_chunks


#选择所有页信息中最靠前的那个
def get_min_page_info(chunk_meta: Dict[str, Any]) -> Tuple[Any, Any]:
    chunk_pages = get_chunk_pages(chunk_meta)
    valid = []
    for p in chunk_pages:
        page_no = p.get("page_no")
        orig_page_no = p.get("orig_page_no")
        valid.append((page_no if page_no is not None else 10**9,
                      orig_page_no if orig_page_no is not None else 10**9))
    if not valid:
        return chunk_meta.get("page_no"), chunk_meta.get("orig_page_no")
    valid.sort()
    page_no, orig_page_no = valid[0]
    page_no = None if page_no == 10**9 else page_no
    orig_page_no = None if orig_page_no == 10**9 else orig_page_no
    return page_no, orig_page_no

#精简图表信息
def prune_figure_refs(figure_refs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pruned = []
    seen = set()
    for item in figure_refs or []:
        image_path = normalize_text((item or {}).get("image_path", ""))
        page_no = (item or {}).get("page_no")
        key = (image_path, page_no)
        if not image_path or key in seen:
            continue
        seen.add(key)
        pruned.append({
            "image_path": image_path,
            "page_no": page_no,
        })
    return pruned

#精简脚注信息
def prune_footnotes(footnotes: List[Any]) -> List[str]:
    pruned = []
    seen = set()
    for item in footnotes or []:
        if isinstance(item, dict):
            text = normalize_text(item.get("footnote_text", ""))
        else:
            text = normalize_text(str(item))
        if not text or text in seen:
            continue
        seen.add(text)
        pruned.append(text)
    return pruned

#获取metadata信息
def build_parent_metadata(chunk_meta: Dict[str, Any], parent_id: str) -> Dict[str, Any]:
    page_no, orig_page_no = get_min_page_info(chunk_meta)
    return {
        "unique_id": parent_id,
        "parent_id": parent_id,
        "chunk_level": "parent",
        "source": chunk_meta.get("source"),
        "page_no": page_no,
        "orig_page_no": orig_page_no,
        "figure_refs": prune_figure_refs(chunk_meta.get("figure_refs", [])),
        "related_footnotes": prune_footnotes(chunk_meta.get("related_footnotes", [])),
    }

def build_child_metadata(parent_metadata: Dict[str, Any], child_id: str) -> Dict[str, Any]:
    return {
        "unique_id": child_id,
        "parent_id": parent_metadata.get("parent_id"),
        "child_id": child_id,
        "chunk_level": "child",
        "source": parent_metadata.get("source"),
        "page_no": parent_metadata.get("page_no"),
        "orig_page_no": parent_metadata.get("orig_page_no"),
    }

#将语义分割后的文档块进一步分割成父块+子块
def build_parent_child_docs(
    enriched_chunks: List[Dict[str, Any]],
    max_parent_size: int = _max_parent_size,
    child_chunk_size: int = _child_chunk_size,
    child_chunk_overlap: int = _child_chunk_overlap,
) -> tuple[List[Document], List[Document]]:
    """
    返回:
    all_mongo_split_docs 负责更新mongo数据库
    all_split_docs 负责更新检索器
    """
    all_split_docs: List[Document] = []
    all_mongo_split_docs: List[Document] = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap,
        separators=["\n\n", "\n", ". ", "。", " ", ""],
    )

    for chunk in enriched_chunks:
        group = normalize_text(chunk.get("page_content", ""))
        if not group:
            continue

        chunk_meta = copy.deepcopy(chunk.get("metadata", {}) or {})
        parent_id = md5_text(group)
        parent_metadata = build_parent_metadata(chunk_meta, parent_id)

        parent_doc = Document(page_content=group, metadata=parent_metadata)
        all_mongo_split_docs.append(parent_doc)

        if len(group) < max_parent_size:
            all_split_docs.append(parent_doc)
            continue

        split_docs = text_splitter.create_documents([group])
        child_added = False

        for child_doc in split_docs:
            child_text = normalize_text(child_doc.page_content)
            if not child_text or child_text == group:
                continue

            child_id = md5_text(child_text)
            child_metadata = build_child_metadata(parent_metadata, child_id)
            final_child_doc = Document(page_content=child_text, metadata=child_metadata)
            all_split_docs.append(final_child_doc)
            all_mongo_split_docs.append(final_child_doc)
            child_added = True

        if not child_added:
            all_split_docs.append(parent_doc)

    return all_split_docs, all_mongo_split_docs

#总函数
def process_pages_to_chunks(
    pages: List[Dict[str, Any]],
    propagation_steps: int = 1,
    propagation_lambda: float = 0.18,
    cut_percentile: float = 60.0,
    base_threshold: float = 0.10,
    max_parent_size: int = _max_parent_size,
    child_chunk_size: int = _child_chunk_size,
    child_chunk_overlap: int = _child_chunk_overlap,
) -> List[Document]:
    """
    语义分割+长度分割
    """
    enriched_chunks = semantic_split_pages_to_enriched_chunks(
        pages=pages,
        propagation_steps=propagation_steps,
        propagation_lambda=propagation_lambda,
        cut_percentile=cut_percentile,
        base_threshold=base_threshold,
    )

    # save_dict_list_to_jsonl(enriched_chunks, save_Figure)
    # print(f"增加图表信息后的文档块已保存到{save_Figure}")

    all_split_docs, all_mongo_split_docs = build_parent_child_docs(
        enriched_chunks=enriched_chunks,
        max_parent_size=max_parent_size,
        child_chunk_size=child_chunk_size,
        child_chunk_overlap=child_chunk_overlap,
    )

    return  all_split_docs, all_mongo_split_docs

#---------------------------------------------------测试-----------------------------------------------------
def load_page_items(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"输入路径不存在: {path}")

    if os.path.isdir(path):
        candidates = []
        for name in sorted(os.listdir(path)):
            if name.endswith(".json") or name.endswith(".jsonl"):
                candidates.append(os.path.join(path, name))
        if not candidates:
            raise FileNotFoundError(f"目录中未找到 .json / .jsonl 文件: {path}")
        path = candidates[0]
        print(f"[INFO] 输入是目录，自动使用文件: {path}")

    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f".json 文件内容应为 List[Dict]，实际类型: {type(data)}")
        return data

    if path.endswith(".jsonl"):
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except Exception as e:
                    raise ValueError(f"jsonl 第 {line_no} 行解析失败: {e}")
        return items

    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    try:
        data = json.loads(raw)
        if isinstance(data, list):
            print(f"[INFO] 检测到无后缀 JSON 文件: {path}")
            return data
    except Exception:
        pass

    items = []
    try:
        for line_no, line in enumerate(raw.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
        if items:
            print(f"[INFO] 检测到无后缀 JSONL 文件: {path}")
            return items
    except Exception:
        pass

    raise ValueError(f"暂不支持的文件类型或内容无法解析: {path}")

def sort_pages(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def get_key(item: Dict[str, Any]):
        meta = item.get("page_metadata", {}) or item.get("metadata", {}) or {}
        page_no = meta.get("page_no", item.get("page_no", 10**9))
        orig_page_no = meta.get("orig_page_no", item.get("orig_page_no", 10**9))
        return (
            page_no if page_no is not None else 10**9,
            orig_page_no if orig_page_no is not None else 10**9,
        )

    return sorted(pages, key=get_key)

def doc_to_dict(doc: Document) -> Dict[str, Any]:
    return {"page_content": doc.page_content, "metadata": doc.metadata}

def save_dict_list_to_jsonl(items: List[Dict[str, Any]], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pages = load_page_items(INPUT_PATH)
    pages = sort_pages(pages)
    pages_10 = pages[:10]

    print(f"[INFO] 总页数: {len(pages)}")
    print(f"[INFO] 参与测试页数: {len(pages_10)}")

    all_split_docs, all_mongo_split_docs = process_pages_to_chunks(
        pages=pages_10,
        propagation_steps=1,
        propagation_lambda=0.18,
        cut_percentile=60.0,
        base_threshold=0.10,
    )

    print(f"[INFO] 检索用 split docs 数量: {len(all_split_docs)}")
    print(f"[INFO] Mongo docs 数量: {len(all_mongo_split_docs)}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for doc in all_split_docs:
            f.write(json.dumps(doc_to_dict(doc), ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
