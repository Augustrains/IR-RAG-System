import os
import re
import json
import random
import hashlib
from collections import defaultdict
from statistics import mean, median
from typing import Dict, List, Tuple, Any, Optional
import fitz
from tqdm import tqdm
from langchain_core.documents import Document

from src import path


# =========================
# 基本路径与参数
# =========================
file_path = path.raw_docs
split_docs = path.split_docs
cleaned_docs = path.cleaned_docs
save_images = path.saved_images
save_rules=path.rule_save_path+".json"

start_page = 38
end_page = 212

head_ratio = 0.3
random_sample_count = 10
max_text_len = 100
top_threshold = 0.20
bottom_threshold = 0.88


# =========================================================
# 0. 子 PDF 提取
# =========================================================
def extract_pdf_page_range(input_pdf: str, output_pdf: str, start_page: int, end_page: int):
    """
    从 input_pdf 中提取 [start_page, end_page] 页，保存到 output_pdf
    注意：页码按人类习惯，从 1 开始。
    """
    if os.path.exists(output_pdf):
        print(f"[SKIP] 已存在: {output_pdf}")
        return

    src = fitz.open(input_pdf)
    dst = fitz.open()

    total_pages = len(src)
    start_idx = max(0, start_page - 1)
    end_idx = min(end_page - 1, total_pages - 1)

    if start_idx > end_idx:
        raise ValueError("页码范围不合法")

    dst.insert_pdf(src, from_page=start_idx, to_page=end_idx)
    dst.save(output_pdf)
    dst.close()
    src.close()

    print(f"已保存子 PDF: {output_pdf}")
    print(f"提取页码范围: {start_page} ~ {end_page}")

# =========================================================
# 1. 基础工具
# =========================================================
def normalize_text(text: str) -> str:
    if not text:
        return ""
    return " ".join(text.split()).strip()

def is_short_text(text: str, max_len: int = 100) -> bool:
    text = normalize_text(text)
    return bool(text) and len(text) <= max_len

def is_page_number_text(text: str) -> bool:
    """
    常见页码形式识别
    """
    t = normalize_text(text)
    if not t:
        return False

    patterns = [
        r"^\d+$",
        r"^page\s+\d+$",
        r"^page\s+\d+\s+of\s+\d+$",
        r"^-\s*\d+\s*-$",
        r"^\[\d+\]$",
        r"^\d+\s*/\s*\d+$",
        r"^p\.\s*\d+$",
        r"^page\s*[ivxlcdm]+$",
        r"^[ivxlcdm]+$",
    ]
    return any(re.fullmatch(p, t, flags=re.I) for p in patterns)

#定义 caption 类型识别函数，判断本行是不是一个标题的起始行
def detect_caption_type_and_label(text: str):
    t = normalize_text(text)
    if not t:
        return None

    patterns = [
        ("figure", r"^[◮•●■□]*\s*(Figure|Fig\.?)\s+(\d+(\.\d+)?)\b"),
        ("table", r"^[◮•●■□]*\s*(Table|Tab\.?)\s+(\d+(\.\d+)?)\b"),
        ("algorithm", r"^[◮•●■□]*\s*(Algorithm)\s+(\d+(\.\d+)?)\b"),
        ("box", r"^[◮•●■□]*\s*(Box)\s+(\d+(\.\d+)?)\b"),
    ]

    for region_type, pattern in patterns:
        m = re.search(pattern, t, flags=re.I)
        if m:
            label = f"{m.group(1)} {m.group(2)}"
            label = normalize_text(label)
            return {
                "region_type": region_type,
                "caption_label": label,
            }

    return None

#判断是否属于同一个图表标题的文本
def looks_like_caption_continuation(
    prev_line: Dict[str, Any],
    curr_line: Dict[str, Any],
    page_height: float,
    normal_line_gap: float,
) -> bool:
    """
    判断这一行 (curr_line) 是否是上一行 (prev_line) 的“图注续行（caption continuation）
    """
    prev_text = normalize_text(prev_line.get("text", ""))
    curr_text = normalize_text(curr_line.get("text", ""))

    if not prev_text or not curr_text:
        return False

    # 如果当前行本身又是一个新的 Figure/Table/Algorithm/Box，就不是前一个 caption 的续行
    if detect_caption_type_and_label(curr_text):
        return False

    prev_x0, prev_y0, prev_x1, prev_y1 = prev_line["bbox"]
    curr_x0, curr_y0, curr_x1, curr_y1 = curr_line["bbox"]

    y_gap = curr_y0 - prev_y1
    x_diff = abs(curr_x0 - prev_x0)

    # 必须在下方
    if y_gap < 0:
        return False

    # 左边界不要偏太多
    if x_diff > 40:
        return False

    # 基于本页正常正文行距判断，而不是固定死阈值
    max_allowed_gap = max(normal_line_gap * 1.6, page_height * 0.03)
    if y_gap > max_allowed_gap:
        return False

    return True

#处理文本，将被解析出来的文本进行拼接，使得原本可能因为解析而导致分块的文本被重新组合成一个
def merge_same_visual_line_fragments(
    lines: List[Dict[str, Any]],
    page_height: float,
) -> List[Dict[str, Any]]:
    """
    合并“本来属于同一视觉行、但被 PDF 解析拆开”的文本片段。
    例如：
        ◮Figure 8.4
        The ROC curve corresponding to ...
    实际上原文可能是一行，这里先合并成：
        ◮Figure 8.4 The ROC curve corresponding to ...
    """
    if not lines:
        return lines

    merged = []
    i = 0

    same_row_y_tol = max(2.0, page_height * 0.004)   # 同一行的纵向容忍
    max_inline_gap = 160                             # 同一行片段的最大横向间距

    while i < len(lines):
        curr = dict(lines[i])

        while i + 1 < len(lines):
            nxt = lines[i + 1]

            curr_text = normalize_text(curr.get("text", ""))
            next_text = normalize_text(nxt.get("text", ""))

            if not curr_text or not next_text:
                break

            curr_x0, curr_y0, curr_x1, curr_y1 = curr["bbox"]
            next_x0, next_y0, next_x1, next_y1 = nxt["bbox"]

            # 纵向上非常接近，认为可能是同一视觉行
            same_row = (
                abs(curr_y0 - next_y0) <= same_row_y_tol
                and abs(curr_y1 - next_y1) <= same_row_y_tol
            )

            # nxt 在 curr 右侧，且中间空白不至于大得离谱
            x_gap = next_x0 - curr_x1
            reasonable_inline_gap = 0 <= x_gap <= max_inline_gap

            if same_row and reasonable_inline_gap:
                curr["text"] = normalize_text(curr_text + " " + next_text)
                curr["bbox"] = [
                    min(curr_x0, next_x0),
                    min(curr_y0, next_y0),
                    max(curr_x1, next_x1),
                    max(curr_y1, next_y1),
                ]
                curr["x0"], curr["y0"], curr["x1"], curr["y1"] = curr["bbox"]
                i += 1
            else:
                break

        merged.append(curr)
        i += 1

    return merged

#计算过滤后的文本的行间距
def estimate_normal_line_gap(
    lines: List[Dict[str, Any]],
    page_height: float,
) -> float:
    """
    估计经过过滤后的本页正常正文的行间距（中位数）。
    用于后续判断 caption 是否继续。
    """
    if len(lines) < 2:
        return page_height * 0.012

    gaps = []

    for i in range(len(lines) - 1):
        curr = lines[i]
        nxt = lines[i + 1]

        curr_text = normalize_text(curr.get("text", ""))
        next_text = normalize_text(nxt.get("text", ""))

        if not curr_text or not next_text:
            continue

        y_gap = nxt["y0"] - curr["y1"]
        x_diff = abs(nxt["x0"] - curr["x0"])

        # 只统计“像正常上下两行正文”的候选
        if 0 <= y_gap <= page_height * 0.05 and x_diff <= 40:
            gaps.append(y_gap)

    if not gaps:
        return page_height * 0.012

    return median(gaps)

#获取完整图片表格的标题
def collect_caption_from_lines(
    lines: List[Dict[str, Any]],
    start_idx: int,
    page_height: float,
    normal_line_gap: float,
):  
    """
    lines:当前页的行列表
    start_idx:从第几行开始尝试识别caption
    目标是返回一个字典,字典保存一个图表标题的相关信息
    """
    first_line = lines[start_idx]
    meta = detect_caption_type_and_label(first_line["text"])
    if not meta:
        return None

    caption_text_parts = [normalize_text(first_line["text"])]
    caption_boxes = [first_line["bbox"]]
    end_idx = start_idx

    prev_vertical_gap = None
    max_extra_lines = 8  # 最多向后尝试合并 8 行，可按需调整

    for i in range(start_idx + 1, min(len(lines), start_idx + 1 + max_extra_lines)):
        curr = lines[i]
        prev = lines[i - 1] if i - 1 >= start_idx else first_line

        if not looks_like_caption_continuation(
            prev_line=lines[end_idx],
            curr_line=curr,
            page_height=page_height,
            normal_line_gap=normal_line_gap,
        ):
            break

        # 再加一层“行距突变终止”
        prev_box = lines[end_idx]["bbox"]
        curr_box = curr["bbox"]
        y_gap = curr_box[1] - prev_box[3]

        if prev_vertical_gap is not None:
            # 如果当前 gap 相比前一个 gap 突然变大很多，则认为 caption 到这里结束
            if y_gap > max(prev_vertical_gap * 1.8 + 1.5, normal_line_gap * 1.8):
                break

        caption_text_parts.append(normalize_text(curr["text"]))
        caption_boxes.append(curr["bbox"])
        end_idx = i
        prev_vertical_gap = y_gap

    caption_text = normalize_text(" ".join(caption_text_parts))

    x0 = min(b[0] for b in caption_boxes)
    y0 = min(b[1] for b in caption_boxes)
    x1 = max(b[2] for b in caption_boxes)
    y1 = max(b[3] for b in caption_boxes)

    return {
        "region_type": meta["region_type"],
        "caption_label": meta["caption_label"],
        "caption_text": caption_text,
        "caption_bbox": [x0, y0, x1, y1],
        "start_idx": start_idx,
        "end_idx": end_idx,
    }

#从 page_blocks 中抽取所有行，排序+合并碎片，返回干净的line流
def flatten_page_lines(page_blocks: List[Dict[str, Any]], page_height: float) -> List[Dict[str, Any]]:
    """
    传入原始利用page_blocks得到的所有行,
    返回经过处理、合并后的文本行
    """
    lines = []
    for block in page_blocks:
        for line in block.get("lines", []):
            item = dict(line)
            lines.append(item)

    lines.sort(key=lambda x: (x["y0"], x["x0"]))

    # 先合并“本来同一行但被拆开”的片段
    lines = merge_same_visual_line_fragments(lines, page_height)

    return lines

#提取一页中的图表内容
def extract_visual_regions_from_page(
    page,
    page_blocks,
    learned_rules: Dict[str, Any],
    save_dir: str,
    zoom: float = 2.0,
) -> List[Dict[str, Any]]:
    """
    传入一页的原始blcok,
    返回本页对应的图表
    """
    visual_regions = []

    page_rect = page.rect
    page_width = page_rect.width
    page_height = page_rect.height
    mat = fitz.Matrix(zoom, zoom)

    os.makedirs(save_dir, exist_ok=True)

    # 先用 learned_rules 得到页眉页脚的安全边界
    header_rule = learned_rules.get("layout_rules", {}).get("header_rule", {})
    footer_rule = learned_rules.get("layout_rules", {}).get("footer_rule", {})

    top_zone_max = header_rule.get("top_zone_max", 0.20)
    bottom_zone_min = footer_rule.get("bottom_zone_min", 0.88)
    
    #得到正文的安全上下界
    safe_top_y = page_height * top_zone_max
    safe_bottom_y = page_height * bottom_zone_min

    # 1. 先去掉 header/footer block，再从剩余内容里检测 caption
    content_blocks = []
    for block in page_blocks:
        if is_dynamic_header_block(block, learned_rules):
            continue
        if is_dynamic_footer_block(block, learned_rules):
            continue
        content_blocks.append(block)

    lines = flatten_page_lines(content_blocks, page_height)
    normal_line_gap = estimate_normal_line_gap(lines, page_height)

    used = set()
    region_idx = 0

    for i in range(len(lines)):
        if i in used:
            continue

        cap = collect_caption_from_lines(
            lines=lines,
            start_idx=i,
            page_height=page_height,
            normal_line_gap=normal_line_gap,
        )
        if not cap:
            continue

        for j in range(cap["start_idx"], cap["end_idx"] + 1):
            used.add(j)

        x0, y0, x1, y1 = cap["caption_bbox"]
        region_type = cap["region_type"]

        crop_x0 = max(0, page_width * 0.08)
        crop_x1 = min(page_width, page_width * 0.92)

        # table 默认在 caption 下方，其余默认在 caption 上方
        if str(region_type).lower() == "table":
            crop_y0 = min(page_height, y1 + 5)
            crop_y1 = min(page_height, y1 + page_height * 0.35)

            # 避开页脚区域
            crop_y1 = min(crop_y1, safe_bottom_y - 2)

            # 如果下方空间不够，再尝试 caption 上方
            if crop_y1 <= crop_y0 + 20:
                crop_y0 = max(safe_top_y + 2, y0 - page_height * 0.35)
                crop_y1 = max(safe_top_y + 2, y0 - 5)
        else:
            crop_y0 = max(safe_top_y + 2, y0 - page_height * 0.35)
            crop_y1 = max(safe_top_y + 2, y0 - 5)

            # 如果上方空间不够，再尝试 caption 下方
            if crop_y1 <= crop_y0 + 20:
                crop_y0 = min(page_height, y1 + 5)
                crop_y1 = min(page_height, y1 + page_height * 0.35)
                crop_y1 = min(crop_y1, safe_bottom_y - 2)

        if crop_y1 <= crop_y0 + 20:
            continue

        crop_rect = fitz.Rect(crop_x0, crop_y0, crop_x1, crop_y1)

        try:
            pix = page.get_pixmap(matrix=mat, clip=crop_rect, alpha=False)

            file_name = f"page_{page.number + 1}_{region_type}_{region_idx}.png"
            image_path = os.path.join(save_dir, file_name)
            pix.save(image_path)

            with open(image_path, "rb") as f:
                image_bytes = f.read()

            visual_regions.append({
                "region_id": f"page_{page.number + 1}_{region_type}_{region_idx}",
                "region_type": region_type,
                "image_type": "captioned_region",
                "image_path": image_path,
                "page_no": page.number + 1,
                "caption_label": cap["caption_label"],
                "caption_text": cap["caption_text"],
                "caption_bbox": cap["caption_bbox"],
                "region_bbox": [crop_x0, crop_y0, crop_x1, crop_y1],
                "width": pix.width,
                "height": pix.height,
                "size_bytes": len(image_bytes),
                "sha256": hashlib.sha256(image_bytes).hexdigest(),
            })
            region_idx += 1

        except Exception as e:
            visual_regions.append({
                "region_id": f"page_{page.number + 1}_{region_type}_{region_idx}",
                "region_type": region_type,
                "image_type": "captioned_region",
                "page_no": page.number + 1,
                "caption_label": cap["caption_label"],
                "caption_text": cap["caption_text"],
                "caption_bbox": cap["caption_bbox"],
                "region_bbox": [crop_x0, crop_y0, crop_x1, crop_y1],
                "error": str(e),
            })
            region_idx += 1

    return visual_regions

#计算面积
def bbox_area(bbox: List[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])

#判断两个矩阵是否重叠
def bbox_intersection_area(b1: List[float], b2: List[float]) -> float:
    """
    计算两个block是否重叠
    """
    x0 = max(b1[0], b2[0])
    y0 = max(b1[1], b2[1])
    x1 = min(b1[2], b2[2])
    y1 = min(b1[3], b2[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)

#计算block有多少比例被region重叠
def overlaps_region(block_bbox: List[float], region_bbox: List[float], threshold: float = 0.3) -> bool:
    area = bbox_area(block_bbox)
    if area <= 0:
        return False
    inter = bbox_intersection_area(block_bbox, region_bbox)
    return (inter / area) >= threshold

#判断block和regions是否重叠
def is_block_in_visual_region(
    block: Dict[str, Any],
    visual_regions: List[Dict[str, Any]],
    overlap_ratio_threshold: float = 0.3,
) -> bool:
    """
    block 只要与 caption_bbox 或 region_bbox 明显重叠，
    就认为属于图表相关内容，不应进入正文。
    """
    block_bbox = block.get("bbox")
    if not block_bbox:
        return False

    text = normalize_text(block.get("text", ""))
    if not text:
        return False

    for vr in visual_regions:
        caption_bbox = vr.get("caption_bbox")
        region_bbox = vr.get("region_bbox")

        if caption_bbox and overlaps_region(block_bbox, caption_bbox, overlap_ratio_threshold):
            return True

        if region_bbox and overlaps_region(block_bbox, region_bbox, overlap_ratio_threshold):
            return True

    return False

# =========================================================
# 2. 采样页策略
# =========================================================

#获取用于计算页眉页脚的样本
def sample_page_numbers(
    total_pages: int,
    head_ratio: float = 0.3,
    random_sample_count: int = 12,
    min_page: int = 1,
    seed: int = 42,
) -> List[int]:
    """
    返回 1-based 页码列表：
    - 先取前 head_ratio 比例页面
    - 再从剩余页面随机补样
    """
    if total_pages <= 0:
        return []

    rng = random.Random(seed)

    start_page_ = max(1, min_page)
    all_pages = list(range(start_page_, total_pages + 1))
    if not all_pages:
        return []

    head_count = max(1, int(len(all_pages) * head_ratio))
    head_pages = all_pages[:head_count]

    remaining = all_pages[head_count:]
    random_pages = rng.sample(remaining, min(random_sample_count, len(remaining))) if remaining else []

    sampled = sorted(set(head_pages + random_pages))
    return sampled

# =========================================================
# 3. 用 fitz 提取 blocks
# =========================================================

#初步提取pdf中的blocks
def extract_blocks_from_fitz_page(page) -> List[Dict[str, Any]]:
    """
    使用 fitz page.get_text('dict') 提取文本块，并保留 bbox / text / line 信息
    """
    raw = page.get_text("dict")
    page_rect = page.rect
    page_height = page_rect.height

    blocks = []
    for block in raw.get("blocks", []):
        if block.get("type", 0) != 0:
            continue

        lines_info = []
        merged_lines_text = []

        block_font_sizes = []

        for line in block.get("lines", []):
            spans = line.get("spans", [])
            span_text = "".join(span.get("text", "") for span in spans)
            span_text = normalize_text(span_text)
            if not span_text:
                continue

            line_bbox = line.get("bbox", None)
            if not line_bbox or len(line_bbox) != 4:
                continue

            line_font_sizes = [float(span.get("size", 0.0)) for span in spans if span.get("size")]
            block_font_sizes.extend(line_font_sizes)

            lx0, ly0, lx1, ly1 = line_bbox
            lines_info.append({
                "text": span_text,
                "bbox": [lx0, ly0, lx1, ly1],
                "x0": lx0,
                "y0": ly0,
                "x1": lx1,
                "y1": ly1,
                "avg_font_size": mean(line_font_sizes) if line_font_sizes else None,
            })
            merged_lines_text.append(span_text)

        text = normalize_text(" ".join(merged_lines_text))
        if not text:
            continue

        bbox = block.get("bbox", None)
        if not bbox or len(bbox) != 4:
            continue

        x0, y0, x1, y1 = bbox

        if page_height > 0:
            norm_y = ((y0 + y1) / 2.0) / page_height
            top_edge_ratio = y0 / page_height
            bottom_edge_ratio = y1 / page_height
            block_height_ratio = (y1 - y0) / page_height
        else:
            norm_y = -1.0
            top_edge_ratio = -1.0
            bottom_edge_ratio = -1.0
            block_height_ratio = 0.0

        blocks.append({
            "text": text,
            "bbox": [x0, y0, x1, y1],
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
            "norm_y": norm_y,
            "top_edge_ratio": top_edge_ratio,
            "bottom_edge_ratio": bottom_edge_ratio,
            "page_height": page_height,
            "block_height_ratio": block_height_ratio,
            "text_len": len(text),
            "word_count": len(text.split()),
            "lines": lines_info,
            "avg_font_size": mean(block_font_sizes) if block_font_sizes else None,
            "min_font_size": min(block_font_sizes) if block_font_sizes else None,
            "max_font_size": max(block_font_sizes) if block_font_sizes else None,
        })

    blocks = sorted(blocks, key=lambda x: (x["y0"], x["x0"]))

    for i, block in enumerate(blocks):
        if i < len(blocks) - 1:
            next_block = blocks[i + 1]
            gap = max(0.0, next_block["y0"] - block["y1"])
            block["gap_below_ratio"] = gap / block["page_height"] if block["page_height"] > 0 else 0.0
            block["gap_below"] = gap
        else:
            block["gap_below_ratio"] = 0.0
            block["gap_below"] = 0.0

        if i > 0:
            prev_block = blocks[i - 1]
            gap_above = max(0.0, block["y0"] - prev_block["y1"])
            block["gap_above_ratio"] = gap_above / block["page_height"] if block["page_height"] > 0 else 0.0
            block["gap_above"] = gap_above
        else:
            block["gap_above_ratio"] = 0.0
            block["gap_above"] = 0.0

    return blocks

# =========================================================
# 4. 基于 fitz blocks 学习 header/footer 规则
# =========================================================

#获取页眉页脚候选框
def extract_page_layout_candidates_from_blocks(
    page_blocks: List[Dict[str, Any]],
    top_threshold: float = 0.20,
    bottom_threshold: float = 0.88,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    从单页 block 中提取顶部/底部候选块
    这一步是只利用位置信息+轻量化的结构进行粗略判断
    """
    top_candidates = []
    bottom_candidates = []

    for block in page_blocks:
        text = normalize_text(block.get("text", ""))
        if not text:
            continue

        if block["top_edge_ratio"] <= top_threshold:
            if block["block_height_ratio"] <= 0.03 and block["text_len"] <= 200:
                top_candidates.append(block)
            continue

        if block["bottom_edge_ratio"] >= bottom_threshold:
            if block["text_len"] <= 160:
                bottom_candidates.append(block)

    return {
        "top": top_candidates,
        "bottom": bottom_candidates,
    }

#利用页眉页脚候选块得到规则
def learn_dynamic_header_footer_rules_from_blocks(
    sampled_page_blocks: Dict[int, List[Dict[str, Any]]],
    max_text_len: int = 120,
    top_threshold: float = 0.20,
    bottom_threshold: float = 0.88,
    debug: bool = True,
) -> Dict[str, Any]:
    """
    计算页眉页脚的总调用函数，
    传入已经分割好的样本,返回规则
    """
    top_text_lens = []
    top_word_counts = []
    top_gap_below_ratios = []
    top_block_heights = []
    top_norm_ys = []

    bottom_text_lens = []
    bottom_word_counts = []
    bottom_block_heights = []
    bottom_norm_ys = []

    debug_rows_top = []
    debug_rows_bottom = []

    page_items = sorted(sampled_page_blocks.items(), key=lambda x: x[0])

    for page_no, page_blocks in tqdm(page_items, desc="学习页眉页脚规则", unit="page"):
        cands = extract_page_layout_candidates_from_blocks(
            page_blocks=page_blocks,
            top_threshold=top_threshold,
            bottom_threshold=bottom_threshold,
        )

        if debug:
            print(f"\n----- page {page_no} raw top candidates -----")
            for row in cands["top"][:10]:
                print(
                    f"top={row['top_edge_ratio']:.3f} "
                    f"center={row['norm_y']:.3f} "
                    f"h={row['block_height_ratio']:.3f} "
                    f"gap={row['gap_below_ratio']:.3f} "
                    f"len={row['text_len']} "
                    f"words={row['word_count']} "
                    f"text={row['text']}"
                )

        for row in cands["top"]:
            text = row["text"]

            if is_page_number_text(text):
                if debug:
                    print(f"[DROP_PAGE_NUM][p={page_no}] {text}")
                continue

            if row["block_height_ratio"] > 0.03:
                if debug:
                    print(f"[DROP_TOO_TALL][p={page_no}] h={row['block_height_ratio']:.3f} text={text}")
                continue

            if row["text_len"] > 200:
                if debug:
                    print(f"[DROP_TOO_LONG][p={page_no}] len={row['text_len']} text={text}")
                continue

            if row["word_count"] > 30:
                if debug:
                    print(f"[DROP_TOO_MANY_WORDS][p={page_no}] words={row['word_count']} text={text}")
                continue

            top_text_lens.append(row["text_len"])
            top_word_counts.append(row["word_count"])
            top_gap_below_ratios.append(row["gap_below_ratio"])
            top_block_heights.append(row["block_height_ratio"])
            top_norm_ys.append(row["norm_y"])

            debug_rows_top.append((
                page_no,
                row["norm_y"],
                row["gap_below_ratio"],
                row["block_height_ratio"],
                text
            ))

        for row in cands["bottom"]:
            text = row["text"]

            if not is_short_text(text, max_len=max_text_len):
                if debug:
                    print(f"[DROP_BOTTOM_TOO_LONG][p={page_no}] len={row['text_len']} text={text}")
                continue

            bottom_text_lens.append(row["text_len"])
            bottom_word_counts.append(row["word_count"])
            bottom_block_heights.append(row["block_height_ratio"])
            bottom_norm_ys.append(row["norm_y"])

            debug_rows_bottom.append((page_no, row["norm_y"], row["block_height_ratio"], text))

    learned_rules = {
        "header_rule": {
            "enabled": True,
            "top_zone_max": top_threshold,
            "max_text_len": int(max(top_text_lens) if top_text_lens else max_text_len),
            "max_words": int(max(top_word_counts) if top_word_counts else 20),
            "max_block_height_ratio": float(max(top_block_heights) if top_block_heights else 0.03),
            "min_gap_below_ratio": float(median(top_gap_below_ratios) if top_gap_below_ratios else 0.005),
            "norm_y_mean": float(mean(top_norm_ys)) if top_norm_ys else None,
        },
        "footer_rule": {
            "enabled": True,
            "bottom_zone_min": bottom_threshold,
            "max_text_len": int(max(bottom_text_lens) if bottom_text_lens else max_text_len),
            "max_words": int(max(bottom_word_counts) if bottom_word_counts else 20),
            "max_block_height_ratio": float(max(bottom_block_heights) if bottom_block_heights else 0.04),
            "norm_y_mean": float(mean(bottom_norm_ys)) if bottom_norm_ys else None,
        },
        "page_number_rule": {
            "enabled": True
        }
    }

    if debug:
        print("\n===== 动态 header 学习候选（top zone） =====")
        for page_no, norm_y, gap_ratio, h_ratio, text in debug_rows_top[:200]:
            print(
                f"[TOP][p={page_no}] y={norm_y:.3f} gap_below={gap_ratio:.3f} "
                f"h_ratio={h_ratio:.3f} | {text}"
            )

        print("\n===== 动态 footer 学习候选（bottom zone） =====")
        for page_no, norm_y, h_ratio, text in debug_rows_bottom[:200]:
            print(
                f"[BOTTOM][p={page_no}] y={norm_y:.3f} h_ratio={h_ratio:.3f} | {text}"
            )

        print("\n===== 学习到的动态布局规则 =====")
        print(json.dumps(learned_rules, ensure_ascii=False, indent=2))
        print()

    return learned_rules

#调用获取页眉页脚规则的总函数
def learn_header_footer_rules_from_sampled_pages(
    file_path: str,
    sampled_pages: List[int],
    max_text_len: int = 100,
    top_threshold: float = 0.20,
    bottom_threshold: float = 0.88,
    debug: bool = True,
) -> Dict[str, Any]:
    """
    页眉页脚规则学习模块”的总入口
    完全基于 fitz 的采样页规则学习
    sampled_pages 是 1-based 页码
    """
    pdf = fitz.open(file_path)
    sampled_page_blocks: Dict[int, List[Dict[str, Any]]] = {}

    for page_no in sampled_pages:
        if page_no < 1 or page_no > len(pdf):
            continue
        page = pdf.load_page(page_no - 1)
        blocks = extract_blocks_from_fitz_page(page)
        sampled_page_blocks[page_no] = blocks

    pdf.close()

    learned_layout_rules = learn_dynamic_header_footer_rules_from_blocks(
        sampled_page_blocks=sampled_page_blocks,
        max_text_len=max_text_len,
        top_threshold=top_threshold,
        bottom_threshold=bottom_threshold,
        debug=debug,
    )

    return {
        "sampled_pages": sampled_pages,
        "layout_rules": learned_layout_rules,
        "params": {
            "max_text_len": max_text_len,
            "top_threshold": top_threshold,
            "bottom_threshold": bottom_threshold,
        }
    }

# =========================================================
# 5. header/footer 判断
# =========================================================

#判断是否是页眉
def is_dynamic_header_block(block: Dict[str, Any], learned_rules: Dict[str, Any]) -> bool:
    """
    位置优先的页眉判断
    """
    header_rule = learned_rules.get("layout_rules", {}).get("header_rule", {})
    if not header_rule.get("enabled", True):
        return False

    text = normalize_text(block.get("text", ""))
    if not text:
        return False

    top_zone_max = header_rule.get("top_zone_max", 0.20)
    top_edge_ratio = block.get("top_edge_ratio", -1.0)
    text_len = block.get("text_len", len(text))
    word_count = block.get("word_count", len(text.split()))
    block_height_ratio = block.get("block_height_ratio", 0.0)

    if top_edge_ratio < 0 or top_edge_ratio > top_zone_max:
        return False

    if is_page_number_text(text):
        return True

    learned_max_h = header_rule.get("max_block_height_ratio", 0.03)
    max_h = max(0.03, learned_max_h * 1.2)
    if block_height_ratio > max_h:
        return False

    learned_max_len = header_rule.get("max_text_len", 120)
    learned_max_words = header_rule.get("max_words", 20)

    max_len = max(160, int(learned_max_len * 1.2))
    max_words = max(24, int(learned_max_words * 1.2))

    if text_len <= max_len and word_count <= max_words:
        return True

    return False

#判断是否是页脚
def is_dynamic_footer_block(block: Dict[str, Any], learned_rules: Dict[str, Any]) -> bool:
    """
    底部区域优先的页脚判断
    """
    footer_rule = learned_rules.get("layout_rules", {}).get("footer_rule", {})
    if not footer_rule.get("enabled", True):
        return False

    text = normalize_text(block.get("text", ""))
    if not text:
        return False

    bottom_zone_min = footer_rule.get("bottom_zone_min", 0.88)
    bottom_edge_ratio = block.get("bottom_edge_ratio", -1.0)
    text_len = block.get("text_len", len(text))
    word_count = block.get("word_count", len(text.split()))
    block_height_ratio = block.get("block_height_ratio", 0.0)

    if bottom_edge_ratio < bottom_zone_min:
        return False

    if is_page_number_text(text):
        return True

    if text_len <= 160 and word_count <= 25 and block_height_ratio <= 0.08:
        return True

    lowered = text.lower()
    footer_keywords = [
        "cambridge up",
        "cambridge university press",
        "online edition",
        "feedback welcome",
        "draft",
        "copyright",
        "©",
    ]
    if any(k in lowered for k in footer_keywords):
        return True

    return False

#获得block的字号大小
def estimate_body_font_size(blocks: List[Dict[str, Any]], page_height: float) -> float:
    """
    获取已经过滤掉页眉页脚、图表内容的blocks,
    计算字号
    """
    font_sizes = []
    for block in blocks:
        for line in block.get("lines", []) or []:
            text = normalize_text(line.get("text", ""))
            avg_font_size = line.get("avg_font_size")
            y1 = line.get("y1", 0.0)
            if not text or not avg_font_size:
                continue
            if len(text) <= 3:
                continue
            font_sizes.append(float(avg_font_size))

    if not font_sizes:
        return 10.0
    return float(median(font_sizes))

#获得block的字间距
def estimate_normal_line_gap(blocks: List[Dict[str, Any]], page_height: float) -> float:
    """
    计算相邻两行的垂直间距
    """
    gaps = []
    all_lines = []
    for block in blocks:
        for line in block.get("lines", []) or []:
            text = normalize_text(line.get("text", ""))
            if text:
                all_lines.append(line)

    all_lines.sort(key=lambda x: (x.get("y0", 0.0), x.get("x0", 0.0)))

    for i in range(1, len(all_lines)):
        prev_line = all_lines[i - 1]
        curr_line = all_lines[i]
        gap = max(0.0, curr_line.get("y0", 0.0) - prev_line.get("y1", 0.0))
        if 0 < gap <= page_height * 0.05:
            gaps.append(gap)

    if not gaps:
        return page_height * 0.012
    return float(median(gaps))

#格式判断是否是脚注
def looks_like_footnote_start(text: str) -> bool:
    """
    根据格式判断是否是脚注
    """
    t = normalize_text(text)
    if not t:
        return False
    patterns = [
        r'^\d+\.\s+.+',
        r'^\[\d+\]\s+.+',
        r'^\d+\s+[A-Z].+',
    ]
    return any(re.match(p, t) for p in patterns)

#判断是否是脚注的总函数
def is_candidate_footnote_line(
    line: Dict[str, Any],
    body_font_size: float,
    page_height: float,
    normal_line_gap: float,
    require_number: bool = False,
) -> bool:
    """
    严格/宽松模式去判断是否是脚注
    严格模式:位置+字号+间隔+格式
    宽松模式:位置+字号+文本长度
    """
    text = normalize_text(line.get("text", ""))
    if not text:
        return False

    avg_font_size = float(line.get("avg_font_size") or 0.0)
    y1 = float(line.get("y1") or 0.0)
    gap_above = float(line.get("gap_above") or 0.0)

    in_bottom_zone = (y1 / page_height) >= 0.78
    smaller_than_body = avg_font_size > 0 and avg_font_size <= body_font_size * 0.88
    separated_from_body = gap_above >= max(normal_line_gap * 1.3, page_height * 0.010)

    if require_number:
        return in_bottom_zone and smaller_than_body and separated_from_body and looks_like_footnote_start(text)

    return in_bottom_zone and smaller_than_body and len(text) >= 8

#利用一行行字段重构blocks
def rebuild_block_from_lines(block: Dict[str, Any], kept_lines: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    """
    根据筛选后的行(kept_lines),重新构建一个干净的 block,并同步更新所有几何和文本特征。
    """
    #去除空行
    kept_lines = [line for line in kept_lines if normalize_text(line.get("text", ""))]
    if not kept_lines:
        return None

    new_block = dict(block)
    new_block["lines"] = kept_lines
    new_block["text"] = normalize_text(" ".join(line["text"] for line in kept_lines if line.get("text")))
    if not new_block["text"]:
        return None

    x0 = min(line["x0"] for line in kept_lines)
    y0 = min(line["y0"] for line in kept_lines)
    x1 = max(line["x1"] for line in kept_lines)
    y1 = max(line["y1"] for line in kept_lines)

    new_block["bbox"] = [x0, y0, x1, y1]
    new_block["x0"] = x0
    new_block["y0"] = y0
    new_block["x1"] = x1
    new_block["y1"] = y1
    page_height = new_block.get("page_height", 1.0) or 1.0

    new_block["norm_y"] = ((y0 + y1) / 2.0) / page_height
    new_block["top_edge_ratio"] = y0 / page_height
    new_block["bottom_edge_ratio"] = y1 / page_height
    new_block["block_height_ratio"] = (y1 - y0) / page_height
    new_block["text_len"] = len(new_block["text"])
    new_block["word_count"] = len(new_block["text"].split())
    return new_block

#分割正文提取脚注
def split_footnotes_from_blocks(
    blocks: List[Dict[str, Any]],
    body_font_size: float,
    normal_line_gap: float,
    page_height: float,
) -> tuple[List[Dict[str, Any]], List[str]]:
    """
    继续处理正文,分成正文block以及脚注文本
    """
    body_blocks: List[Dict[str, Any]] = []
    footnotes: List[str] = []
    
    #以block为单位进行脚注分离
    for block in blocks:
        lines = [dict(line) for line in (block.get("lines", []) or []) if normalize_text(line.get("text", ""))]
        if not lines:
            continue

        for i, line in enumerate(lines):
            if i == 0:
                line["gap_above"] = block.get("gap_above", 0.0)
            else:
                line["gap_above"] = max(0.0, line.get("y0", 0.0) - lines[i - 1].get("y1", 0.0))

        start_idx = None
        for idx, line in enumerate(lines):
            if is_candidate_footnote_line(
                line,
                body_font_size=body_font_size,
                page_height=page_height,
                normal_line_gap=normal_line_gap,
                require_number=True,
            ):
                start_idx = idx
                break

        if start_idx is None:
            body_blocks.append(block)
            continue

        footnote_lines = [lines[start_idx]]
        for idx in range(start_idx + 1, len(lines)):
            line = lines[idx]
            if is_candidate_footnote_line(
                line,
                body_font_size=body_font_size,
                page_height=page_height,
                normal_line_gap=normal_line_gap,
                require_number=False,
            ):
                footnote_lines.append(line)
            else:
                break

        kept_lines = lines[:start_idx]
        new_body_block = rebuild_block_from_lines(block, kept_lines)
        if new_body_block is not None:
            body_blocks.append(new_body_block)

        footnote_text = normalize_text(" ".join(line["text"] for line in footnote_lines if line.get("text")))
        if footnote_text:
            footnotes.append(footnote_text)

    return body_blocks, footnotes


# =========================================================
# 6. 生成 raw_docs: List[Document]
# =========================================================

#合并相同类型的unit
def merge_adjacent_units(units: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    根据相邻的unit是否是同类型来合并,并且更新对应信息
    这里只合并body、heading、footnote
    """
    if not units:
        return []

    mergeable_types = {"body", "heading", "footnote"}
    merged_units: List[Dict[str, Any]] = []

    for unit in units:
        if not merged_units:
            merged_units.append(unit)
            continue

        prev = merged_units[-1]
        prev_meta = prev.get("metadata", {}) or {}
        curr_meta = unit.get("metadata", {}) or {}

        prev_type = prev_meta.get("unit_type")
        curr_type = curr_meta.get("unit_type")

        if prev_type == curr_type and curr_type in mergeable_types:
            prev_text = normalize_text(prev.get("page_content", ""))
            curr_text = normalize_text(unit.get("page_content", ""))
            if prev_text and curr_text:
                prev["page_content"] = normalize_text(prev_text + "\n" + curr_text)
            elif curr_text:
                prev["page_content"] = curr_text

            prev_font = prev_meta.get("avg_font_size")
            curr_font = curr_meta.get("avg_font_size")
            fonts = [f for f in [prev_font, curr_font] if isinstance(f, (int, float))]
            if fonts:
                prev_meta["avg_font_size"] = sum(fonts) / len(fonts)
            continue

        merged_units.append(unit)

    for i, unit in enumerate(merged_units):
        unit.setdefault("metadata", {})["order_in_page"] = i

    return merged_units

#
def extract_and_clean_full_pdf_to_units(
    file_path: str,
    learned_rules: Dict[str, Any],
    start_page: int,
    debug: bool = True,
) -> List[Dict[str, Any]]:
    """
    利用学习好的规则,过滤页眉页脚,并且生成不同类型的unit,
    返回类型：
    {
    "page_metadata": {...},
    "units": [...]
    }
    """

    pdf = fitz.open(file_path)
    page_results = []

    for idx in tqdm(range(len(pdf)), desc="按页抽取PDF", unit="page"):
        page_no = idx + 1
        orig_page_no = start_page + page_no - 1
        page = pdf.load_page(idx)

        # ===== 1. blocks =====
        blocks = extract_blocks_from_fitz_page(page)

        # ===== 3. 去 header/footer =====
        content_blocks = []

        for block in blocks:
            if is_dynamic_header_block(block, learned_rules):
                continue
            if is_dynamic_footer_block(block, learned_rules):
                continue

            text = normalize_text(block["text"])
            if not text:
                continue

            new_block = dict(block)
            new_block["text"] = text
            content_blocks.append(new_block)

        # ===== 4. visual regions =====
        visual_regions = extract_visual_regions_from_page(
            page=page,
            page_blocks=content_blocks,
            learned_rules=learned_rules,
            save_dir=save_images
        )

        # ===== 5. 剔除图表区域 =====
        non_visual_blocks = []
        for block in content_blocks:
            if is_block_in_visual_region(block, visual_regions):
                continue
            non_visual_blocks.append(block)

        # ===== 6. 脚注处理 =====
        body_font_size = estimate_body_font_size(non_visual_blocks, page.rect.height)
        normal_line_gap = estimate_normal_line_gap(non_visual_blocks, page.rect.height)

        body_blocks, footnotes = split_footnotes_from_blocks(
            non_visual_blocks,
            body_font_size=body_font_size,
            normal_line_gap=normal_line_gap,
            page_height=page.rect.height,
        )

        # ===== 7. 标题识别（简单版：用字号）=====
        page_body_font = estimate_body_font_size(body_blocks, page.rect.height)

        def is_heading(block):
            fs = block.get("avg_font_size") or 0
            text = normalize_text(block.get("text", ""))

            if not text:
                return False

            if detect_caption_type_and_label(text):
                return False

            return fs >= page_body_font * 1.2 and len(text.split()) <= 20

        # ===== 8. 构建 units =====
        units = []
        order = 0

        # ---- body + heading ----
        for block in body_blocks:
            text = normalize_text(block["text"])
            if not text:
                continue

            unit_type = "heading" if is_heading(block) else "body"

            units.append({
                "page_content": text,
                "metadata": {
                    "unit_type": unit_type,
                    "order_in_page": order,
                    "avg_font_size": block.get("avg_font_size"),
                    "_sort_y": (block.get("bbox") or [0, 10**9, 0, 10**9])[1],
                    "_sort_x": (block.get("bbox") or [10**9, 0, 10**9, 0])[0],
                }
            })
            order += 1

        # ---- caption ----
        for vr in visual_regions:
            caption_text = normalize_text(vr.get("caption_text", ""))
            if not caption_text:
                continue

            units.append({
                "page_content": caption_text,
                "metadata": {
                    "unit_type": "caption",
                    "order_in_page": order,
                    "region_id": vr.get("region_id"),
                    "region_type": vr.get("region_type"),
                    "caption_label": vr.get("caption_label"),
                    "image_path": vr.get("image_path"),
                    "_sort_y": (vr.get("caption_bbox") or [0, 10**9, 0, 10**9])[1],
                    "_sort_x": (vr.get("caption_bbox") or [10**9, 0, 10**9, 0])[0],
                }
            })
            order += 1

        # ---- footnote ----
        footnote_sort_y = page.rect.height * 0.95
        for fn in footnotes:
            units.append({
                "page_content": fn,
                "metadata": {
                    "unit_type": "footnote",
                    "order_in_page": order,
                    "_sort_y": footnote_sort_y + order * 0.001,
                    "_sort_x": 0,
                }
            })
            order += 1

        # ===== 9. 排序（按阅读顺序）=====
        units.sort(key=lambda u: (
            u["metadata"].get("_sort_y", 10**9),
            u["metadata"].get("_sort_x", 10**9),
            u["metadata"].get("order_in_page", 10**9),
        ))

        units = merge_adjacent_units(units)

        # 清理临时排序字段
        for u in units:
            u["metadata"].pop("_sort_y", None)
            u["metadata"].pop("_sort_x", None)

        # ===== 10. page_metadata =====
        page_metadata = {
            "page_no": page_no,
            "orig_page_no": orig_page_no,
            "source": file_path,
        }

        page_results.append({
            "page_metadata": page_metadata,
            "units": units
        })

        if debug:
            print(f"[PAGE {page_no}] units={len(units)}")

    pdf.close()
    return page_results


# =========================================================
# 7. 一体化主流程  分割pdf+计算规则+过滤pdf
# =========================================================
def build_raw_docs_with_sampled_learning(
    file_path: str,
    start_page: int,
    head_ratio: float = 0.3,
    random_sample_count: int = 12,
    min_page: int = 1,
    sample_seed: int = 42,
    max_text_len: int = 100,
    top_threshold: float = 0.20,
    bottom_threshold: float = 0.88,
    debug: bool = True,
) -> Tuple[List[Document], Dict[str, Any]]:
    pdf = fitz.open(file_path)
    total_pages = len(pdf)
    pdf.close()
    
    sampled_pages = sample_page_numbers(
        total_pages=total_pages,
        head_ratio=head_ratio,
        random_sample_count=random_sample_count,
        min_page=min_page,
        seed=sample_seed,
    )

    if debug:
        print("===== 采样页 =====")
        print(sampled_pages)
        print()

    learned_rules = learn_header_footer_rules_from_sampled_pages(
        file_path=file_path,
        sampled_pages=sampled_pages,
        max_text_len=max_text_len,
        top_threshold=top_threshold,
        bottom_threshold=bottom_threshold,
        debug=debug,
    )

    page_units = extract_and_clean_full_pdf_to_units(
        file_path=file_path,
        learned_rules=learned_rules,
        start_page=start_page,
        debug=debug,
    )

    return page_units, learned_rules


# =========================================================
# 8. 调试入口
# =========================================================

def load_pdf(
    file_path: str,
    start_page: int,
):
    page_units, learned_rules = build_raw_docs_with_sampled_learning(
        file_path=file_path,
        start_page=start_page,
        head_ratio=0.3,
        random_sample_count=10,
        min_page=1,
        sample_seed=42,
        max_text_len=100,
        top_threshold=0.20,
        bottom_threshold=0.88,
        debug=False,
    )

    json_data = page_units

    # with open(cleaned_docs, "w", encoding="utf-8") as f:
    #     json.dump(json_data, f, ensure_ascii=False, indent=2)
    # print(f"\n已保存 raw_docs(JSON) 到: {cleaned_docs}")

    # if rule_save_path:
    #     with open(rule_save_path, "w", encoding="utf-8") as f:
    #         json.dump(learned_rules, f, ensure_ascii=False, indent=2)
    #     print(f"\n已保存规则到: {rule_save_path}")

    return json_data, learned_rules


# =========================================================
# 9. 主程序
# =========================================================
if __name__ == "__main__":
    extract_pdf_page_range(
        input_pdf=file_path,
        output_pdf=split_docs,
        start_page=start_page,
        end_page=end_page,
    )

    raw_docs, learned_rules = load_pdf(
        file_path=split_docs,
        start_page=start_page,
    )
    
