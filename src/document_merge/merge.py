import os
import re
import json
import copy
from typing import List, Dict, Any, Optional
from src import path

# =========================================================
# 1. 路径配置
# =========================================================
INPUT_PATH = path.cleaned_docs
OUTPUT_PATH = path.merged_docs


# =========================================================
# 2. 基础工具函数
# =========================================================
def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

#粗略分割文本
def split_sentences_rough(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r'(?<=[\.\!\?。！？])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]

#判断是否是标题？
def is_likely_heading(line: str) -> bool:
    if not line:
        return False

    s = line.strip()
    if not s:
        return False

    if len(s) > 150:
        return False

    patterns = [
        r"^\d+(\.\d+)*\s+[A-Z].*",
        r"^(Chapter|CHAPTER)\s+\d+.*",
        r"^(Appendix|APPENDIX)\s+[A-Z0-9].*",
        r"^(Figure|Table)\s+\d+(\.\d+)?\b.*",
        r"^(References|REFERENCES)$",
        r"^(Bibliography|BIBLIOGRAPHY)$",
        r"^(Index|INDEX)$",
    ]
    for p in patterns:
        if re.match(p, s):
            return True

    alpha_cnt = sum(ch.isalpha() for ch in s)
    if 0 < alpha_cnt <= 40:
        upper_ratio = sum(ch.isupper() for ch in s if ch.isalpha()) / alpha_cnt
        if upper_ratio > 0.75:
            return True

    return False

#判断当前文本是否结束
def ends_like_incomplete(text: str) -> bool:
    if not text:
        return False

    s = text.rstrip()
    if not s:
        return False

    if re.search(r'[.!?。！？;；:"”’\)\]]\s*$', s):
        return False

    if s.endswith('-'):
        return True

    if re.search(r'[,\(\[]\s*$', s):
        return True

    if re.search(r'[A-Za-z0-9]\s*$', s):
        return True

    return False

#判断一个文本是否是新章节（“标题”或“编号章节）
def starts_like_new_section(text: str) -> bool:
    if not text:
        return False

    s = text.lstrip()
    if not s:
        return False
    
    #依靠第一行判断
    first_line = s.split("\n", 1)[0].strip()
    if is_likely_heading(first_line):
        return True

    if re.match(r'^\d+(\.\d+)*\s+[A-Z]', s):
        return True

    return False

#判断是否是对上一段的延续
def starts_like_continuation(text: str) -> bool:
    if not text:
        return False

    s = text.lstrip()
    if not s:
        return False

    if starts_like_new_section(s):
        return False

    if re.match(r'^[a-z]', s):
        return True

    if re.match(
        r'^(and|or|but|that|which|who|where|when|with|by|to|of|for|in|on|as|is|are|was|were|be|been|being)\b',
        s,
        flags=re.I,
    ):
        return True

    if re.match(r'^[\)\],;:]', s):
        return True

    if re.match(r'^\d+\.\s', s):
        return True

    return False

#是否是图表标题的开头
def starts_like_new_caption(text: str) -> bool:
    s = normalize_text(text)
    if not s:
        return False
    return bool(re.match(r'^(Figure|Fig\.?|Table)\s+\d+(?:\.\d+)*\b', s, flags=re.I))

#文本合并
def merge_texts(prev_text: str, next_text: str) -> str:
    prev_text = (prev_text or "").rstrip()
    next_text = (next_text or "").lstrip()

    if not prev_text:
        return next_text
    if not next_text:
        return prev_text

    if prev_text.endswith('-'):
        return prev_text[:-1] + next_text

    return prev_text + " " + next_text

#得到合并后文本的字号
def average_font_size(prev_value: Any, next_value: Any) -> Optional[float]:
    numbers: List[float] = []
    for value in (prev_value, next_value):
        if isinstance(value, (int, float)):
            numbers.append(float(value))
    if not numbers:
        return None
    return sum(numbers) / len(numbers)

#获得文本来源、页号信息
def build_source_page(page_metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "page_no": page_metadata.get("page_no"),
        "orig_page_no": page_metadata.get("orig_page_no"),
        "source": page_metadata.get("source"),
    }

#跨页合并后，对于文本来源、页号信息的去重合并
def normalize_source_pages(source_pages: Any) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    seen = set()
    for item in source_pages or []:
        if not isinstance(item, dict):
            continue
        page_no = item.get("page_no")
        orig_page_no = item.get("orig_page_no")
        source = item.get("source")
        key = (page_no, orig_page_no, source)
        if key in seen:
            continue
        seen.add(key)
        result.append({
            "page_no": page_no,
            "orig_page_no": orig_page_no,
            "source": source,
        })
    return result

#跨页合并后，对于文本来源、页号信息的去重合并的总函数
def merge_source_pages(prev_pages: Any, next_pages: Any) -> List[Dict[str, Any]]:
    return normalize_source_pages(list(prev_pages or []) + list(next_pages or []))

#把一个 unit 变成结构规范 + 内容干净 的标准格式
def normalize_unit(unit: Dict[str, Any]) -> Dict[str, Any]:
    normalized = copy.deepcopy(unit)
    normalized.setdefault("page_content", "")
    normalized.setdefault("metadata", {})
    normalized["page_content"] = normalize_text(normalized.get("page_content", ""))
    return normalized

#获取该unit的类型
def get_unit_type(unit: Dict[str, Any]) -> str:
    return (unit.get("metadata") or {}).get("unit_type", "") or ""

#获取从下往上第一个非脚注的unit
def get_last_non_footnote_unit(units: List[Dict[str, Any]]) -> Optional[tuple[int, Dict[str, Any]]]:
    for idx in range(len(units) - 1, -1, -1):
        unit = units[idx]
        if get_unit_type(unit) != "footnote":
            return idx, unit
    return None


#判断是否可能合并的逻辑，只有正文+正文、图表标题+正文、图表标题+图表标题才可能能合并
def can_consider_merge(prev_type: str, next_type: str) -> bool:
    return (prev_type, next_type) in {
        ("body", "body"),
        ("caption", "caption"),
        ("caption", "body"),
    }

#多条件合并判定器,判断是否可以合并
def should_merge_units(prev_unit: Dict[str, Any], next_unit: Dict[str, Any]) -> bool:
    """
    前置判断条件：类型是否允许、文本是否为空、前一个是否没结束
    然后分两类处理:
        通用情况 如果 next_text 明显像 continuation,直接合并。
        caption 特殊情况 如果是 caption -> caption 或 caption -> body,再额外放宽规则,但要排除： 新 section |新 caption 且 label 不同
    """
    prev_type = get_unit_type(prev_unit)
    next_type = get_unit_type(next_unit)

    # 1. 类型白名单
    if not can_consider_merge(prev_type, next_type):
        return False

    # 2. 文本判空与规范化
    prev_text = normalize_text(prev_unit.get("page_content", ""))
    next_text = normalize_text(next_unit.get("page_content", ""))
    if not prev_text or not next_text:
        return False

    # 3. 前一个单元必须像“未结束”
    if not ends_like_incomplete(prev_text):
        return False

    # 4. 通用规则：后一个明显像续写，则直接合并
    if starts_like_continuation(next_text):
        return True

    # 5. caption 特殊规则：caption 可以与后续 caption/body 放宽合并
    if prev_type == "caption" and next_type in {"caption", "body"}:
        # 新章节开头，不能并
        if starts_like_new_section(next_text):
            return False

        # 如果后一个看起来是新的 caption，则必须是同一个 caption
        if next_type == "caption" and starts_like_new_caption(next_text):
            prev_label = (prev_unit.get("metadata") or {}).get("caption_label")
            next_label = (next_unit.get("metadata") or {}).get("caption_label")
            if prev_label and next_label and prev_label != next_label:
                return False
        return True

    return False

#把 next_unit 合并进 prev_unit，并更新文本 + metadata
def merge_unit_into_previous(
    prev_unit: Dict[str, Any],
    next_unit: Dict[str, Any],
    next_page_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    merged = copy.deepcopy(prev_unit)
    merged_meta = merged.setdefault("metadata", {})
    prev_meta = prev_unit.get("metadata") or {}
    next_meta = next_unit.get("metadata") or {}
    next_page_metadata = next_page_metadata or {}

    merged["page_content"] = normalize_text(
        merge_texts(prev_unit.get("page_content", ""), next_unit.get("page_content", ""))
    )

    avg_font = average_font_size(merged_meta.get("avg_font_size"), next_meta.get("avg_font_size"))
    if avg_font is None:
        merged_meta.pop("avg_font_size", None)
    else:
        merged_meta["avg_font_size"] = avg_font

    merged_meta["page_no"] = prev_meta.get("page_no")
    merged_meta["orig_page_no"] = prev_meta.get("orig_page_no")
    merged_meta["source"] = prev_meta.get("source")
    merged_meta["source_pages"] = merge_source_pages(
        prev_meta.get("source_pages"),
        next_meta.get("source_pages") or [build_source_page(next_page_metadata)],
    )

    merged_meta.pop("cross_page", None)
    return merged

#给每个 unit 重新编号，标记它在当前页面中的顺序
def reindex_units(units: List[Dict[str, Any]]) -> None:
    for idx, unit in enumerate(units):
        metadata = unit.setdefault("metadata", {})
        metadata["order_in_page"] = idx

#对页按照页号排序
def sort_pages(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        pages,
        key=lambda item: (
            (item.get("page_metadata") or {}).get("orig_page_no", 10**9),
            (item.get("page_metadata") or {}).get("page_no", 10**9),
        ),
    )


# =========================================================
# 3. 读写页面数据
# =========================================================

#读取路径下所有文件
def load_pages(input_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"路径不存在: {input_path}")
    
    #r如果是目录，则按文件名先后顺序依次拿出
    if os.path.isdir(input_path):
        pages: List[Dict[str, Any]] = []
        #按照文件名排序
        for fname in sorted(os.listdir(input_path)):
            fpath = os.path.join(input_path, fname)
            if not os.path.isfile(fpath):
                continue
            pages.extend(_load_pages_from_file(fpath))
        return sort_pages(pages)

    return sort_pages(_load_pages_from_file(input_path))

#读取一个文件
def _load_pages_from_file(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return []

    try:
        data = json.loads(content)
        if isinstance(data, list):
            return [normalize_page(item) for item in data if isinstance(item, dict)]
        if isinstance(data, dict):
            return [normalize_page(data)]
    except json.JSONDecodeError:
        pass

    pages: List[Dict[str, Any]] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        if isinstance(item, dict):
            pages.append(normalize_page(item))
    return pages


#把 page 内所有 unit 变成带完整来源信息的标准结构
def normalize_page(page: Dict[str, Any]) -> Dict[str, Any]:
    normalized = copy.deepcopy(page)
    page_metadata = normalized.get("page_metadata", {}) or {}
    normalized["page_metadata"] = page_metadata
    units = normalized.get("units", []) or []
    normalized_units: List[Dict[str, Any]] = []
    source_page = build_source_page(page_metadata)

    for unit in units:
        if not isinstance(unit, dict):
            continue
        normalized_unit = normalize_unit(unit)
        metadata = normalized_unit.setdefault("metadata", {})
        metadata["page_no"] = page_metadata.get("page_no")
        metadata["orig_page_no"] = page_metadata.get("orig_page_no")
        metadata["source"] = page_metadata.get("source")
        existing_source_pages = metadata.get("source_pages")
        if existing_source_pages:
            metadata["source_pages"] = normalize_source_pages(existing_source_pages)
        else:
            metadata["source_pages"] = [source_page]
        normalized_units.append(normalized_unit)

    normalized["units"] = normalized_units
    return normalized


# =========================================================
# 4. units 级跨页修复主逻辑
# =========================================================


def get_first_non_footnote_idx(units: List[Dict[str, Any]]) -> Optional[int]:
    for i, unit in enumerate(units):
        if get_unit_type(unit) != "footnote":
            return i
    return None


def get_next_non_footnote_idx(units: List[Dict[str, Any]], start_idx: int) -> Optional[int]:
    for i in range(start_idx + 1, len(units)):
        if get_unit_type(units[i]) != "footnote":
            return i
    return None


def get_first_body_idx(units: List[Dict[str, Any]]) -> Optional[int]:
    for i, unit in enumerate(units):
        if get_unit_type(unit) == "body":
            return i
    return None


def page_has_body(units: List[Dict[str, Any]]) -> bool:
    return any(get_unit_type(unit) == "body" for unit in units)


def is_bridge_page(units: List[Dict[str, Any]]) -> bool:
    """
    桥接页：没有 body，且只包含 caption / footnote / heading 这类不会作为正文主链的内容
    这种页允许被“跳过”，不阻断正文跨页关系。
    """
    if not units:
        return True

    has_body = False
    allowed_types = {"caption", "footnote", "heading"}

    for unit in units:
        unit_type = get_unit_type(unit)
        if unit_type == "body":
            has_body = True
            break
        if unit_type and unit_type not in allowed_types:
            return False

    return not has_body


def get_body_candidate_from_page_for_prev_body(
    units: List[Dict[str, Any]]
) -> Optional[int]:
    """
    用于“上一页最后一个 unit 是 body”的场景：
    - 如果本页第一个非脚注是 body，返回它
    - 如果本页第一个非脚注是 caption，且其后紧跟 body，返回这个 body
    - 否则返回 None
    """
    first_idx = get_first_non_footnote_idx(units)
    if first_idx is None:
        return None

    first_type = get_unit_type(units[first_idx])

    if first_type == "body":
        return first_idx

    if first_type == "caption":
        second_idx = get_next_non_footnote_idx(units, first_idx)
        if second_idx is not None and get_unit_type(units[second_idx]) == "body":
            return second_idx

    return None


def get_candidate_from_page_for_prev_caption(
    units: List[Dict[str, Any]]
) -> Optional[int]:
    """
    用于“上一页最后一个 unit 是 caption”的场景：
    - 只取本页第一个非脚注
    - 如果它是 caption 或 body，返回它
    """
    first_idx = get_first_non_footnote_idx(units)
    if first_idx is None:
        return None

    first_type = get_unit_type(units[first_idx])
    if first_type in {"caption", "body"}:
        return first_idx

    return None


def fix_cross_page_units(
    pages: List[Dict[str, Any]],
    max_lookahead: int = 3,
    max_rounds: int = 6,
) -> List[Dict[str, Any]]:
    """
    最终版跨页合并策略：
    1. 反向遍历，避免前向原地修改污染后续匹配入口
    2. 允许跳过 bridge page（仅 caption/footnote/heading，无 body）
    3. 支持有限 lookahead，尽量恢复真正的正文连续关系
    4. 多轮迭代直到稳定，处理链式跨页
    """
    if not pages:
        return []

    fixed_pages = [normalize_page(page) for page in pages]

    for _round in range(max_rounds):
        changed = False

        # 从后向前遍历，避免先抽走中间页 body
        for idx in range(len(fixed_pages) - 2, -1, -1):
            curr_page = fixed_pages[idx]
            curr_units = curr_page.get("units", []) or []
            if not curr_units:
                continue

            prev_candidate = get_last_non_footnote_unit(curr_units)
            if prev_candidate is None:
                continue

            prev_idx, prev_unit = prev_candidate
            prev_type = get_unit_type(prev_unit)

            # 只处理 body / caption 两种末尾单元
            if prev_type not in {"body", "caption"}:
                continue

            best_target_page_idx: Optional[int] = None
            best_target_unit_idx: Optional[int] = None

            # 在后续若干页内找候选
            for step in range(1, max_lookahead + 1):
                next_page_idx = idx + step
                if next_page_idx >= len(fixed_pages):
                    break

                next_page = fixed_pages[next_page_idx]
                next_units = next_page.get("units", []) or []
                if not next_units:
                    continue

                candidate_idx: Optional[int] = None

                if prev_type == "body":
                    candidate_idx = get_body_candidate_from_page_for_prev_body(next_units)

                    if candidate_idx is not None:
                        next_unit = next_units[candidate_idx]
                        if should_merge_units(prev_unit, next_unit):
                            best_target_page_idx = next_page_idx
                            best_target_unit_idx = candidate_idx
                            break

                    # 当前页末尾是 body，但该页无可用 body 候选：
                    # 如果是 bridge page，就允许继续向后看；
                    # 否则视为正文链被阻断。
                    if is_bridge_page(next_units):
                        continue
                    else:
                        break

                elif prev_type == "caption":
                    candidate_idx = get_candidate_from_page_for_prev_caption(next_units)

                    if candidate_idx is not None:
                        next_unit = next_units[candidate_idx]
                        if should_merge_units(prev_unit, next_unit):
                            best_target_page_idx = next_page_idx
                            best_target_unit_idx = candidate_idx
                            break

                    # caption 链通常不建议跳太远。
                    # 如果这一页没有合适 caption/body 候选，则停止。
                    break

            # 真正执行合并
            if best_target_page_idx is not None and best_target_unit_idx is not None:
                target_page = fixed_pages[best_target_page_idx]
                target_units = target_page.get("units", []) or []
                target_unit = target_units[best_target_unit_idx]

                curr_units[prev_idx] = merge_unit_into_previous(
                    prev_unit,
                    target_unit,
                    target_page.get("page_metadata") or {},
                )
                del target_units[best_target_unit_idx]

                reindex_units(curr_units)
                reindex_units(target_units)
                changed = True

        # 清除完全空页
        compact_pages: List[Dict[str, Any]] = []
        for page in fixed_pages:
            units = page.get("units", []) or []
            if not units:
                continue
            reindex_units(units)
            compact_pages.append(page)
        fixed_pages = compact_pages

        if not changed:
            break

    return fixed_pages


def merge_docs(pages: List[Dict[str, Any]], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_pages: List[Dict[str, Any]] = []
    for page in pages:
        units = page.get("units", []) or []
        cleaned_units: List[Dict[str, Any]] = []
        for unit in units:
            unit_copy = copy.deepcopy(unit)
            metadata = unit_copy.get("metadata") or {}
            metadata.pop("cross_page", None)
            unit_copy["metadata"] = metadata
            cleaned_units.append(unit_copy)
        output_pages.append({
            "units": cleaned_units,
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_pages, f, ensure_ascii=False, indent=2)


# =========================================================
# 5. 主函数
# =========================================================
def main():
    raw_pages = load_pages(INPUT_PATH)
    print(f"[INFO] 读取到页面数: {len(raw_pages)}")

    fixed_pages = fix_cross_page_units(raw_pages)
    print(f"[INFO] units 级跨页修复完成: {len(fixed_pages)}")

    merge_docs(fixed_pages, OUTPUT_PATH)
    print(f"[INFO] 已保存到: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
