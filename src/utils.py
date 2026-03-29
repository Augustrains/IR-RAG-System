
import torch
import time
import re
import hashlib
import json
import fitz
import copy
import tiktoken
from tqdm import tqdm

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo.collection import Collection
from typing_extensions import List
from typing import Any, Dict, Optional
from langchain_core.documents import Document
from transformers import MarianMTModel, MarianTokenizer
from src import path
from src.fields.manual_info_mongo import ManualInfo
from src.client.mongodb_config import MongoConfig
from pymongo import UpdateOne


# 全局配置
_chunk_size = 256
_chunk_overlap = 50

encoding = tiktoken.get_encoding("cl100k_base")
#增加unique_id检索，用于保证Mongo去重
manual_text_collection: Collection = MongoConfig.get_collection("manual_text")
manual_text_collection.create_index("unique_id", unique=True)
manual_text_collection.create_index("metadata.source")

file_path = path.raw_docs
split_docs=path.split_docs
cleaned_docs=path.cleaned_docs
start_page=38
end_page=211
manual_collection = MongoConfig.get_collection("manual_text")

class TranslatorZh2En:
    def __init__(self, device=None):
        model_name = "Helsinki-NLP/opus-mt-zh-en"

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name).to(self.device)

    def translate(self, text: str) -> str:
        if not text.strip():
            return text

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        outputs = self.model.generate(**inputs)

        translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated.strip()

class TranslatorEn2Zh:
    def __init__(self, device=None):
        model_name = "Helsinki-NLP/opus-mt-en-zh"

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name).to(self.device)

    def translate(self, text: str) -> str:
        if not text.strip():
            return text

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        outputs = self.model.generate(**inputs)

        translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated.strip()



def build_unique_id(
    source: str,
    text: str,
    chunk_type: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    构建稳定唯一ID（用于 Mongo / Milvus / BM25 对齐）

    设计原则：
    1. 同一文档块 → ID 必须稳定（重复构建不变）
    2. 不同文档块 → 尽量避免冲突
    3. 支持未来扩展（metadata）

    参数：
    - source: 文档来源（通常是 pdf 路径）
    - text: 文档内容
    - chunk_type: page / parent / child 等
    - metadata: 可选扩展字段（如 page_no / parent_id 等）

    返回：
    - md5 字符串（32位）
    """

    metadata = metadata or {}

    # 🔹只保留“稳定字段”，避免引入波动
    stable_meta = {
        "source": source or "",
        "chunk_type": chunk_type or "",
        "page_no": metadata.get("page_no"),
        "parent_id": metadata.get("parent_id"),
        "child_id": metadata.get("child_id"),
    }

    # 🔹构造原始字符串（保证顺序一致）
    payload = {
        "text": (text or "").strip(),
        "meta": stable_meta,
    }

    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)

    # 🔹生成 hash
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=_chunk_size,
    chunk_overlap=_chunk_overlap,
    # 按这个优先级递归切
    separators=["\n\n", "\n"],
    length_function=lambda text: len(encoding.encode(text))
)



def save_2_mongo(split_docs):
    operations = []
    for doc in split_docs:
        # 从 metadata 中提取关键参数
        metadata = doc.metadata
        
        # 构造唯一性 unique_id,这里设计区分source
        unique_id = metadata.get("unique_id")
        if not unique_id:
            continue

        # 创建文档记录对象
        doc_record = ManualInfo(
            unique_id=unique_id,
            page_content=doc.page_content,
            metadata=metadata
        )

        operations.append(
            UpdateOne(
                {"unique_id": doc_record.unique_id},
                {"$set": doc_record.model_dump()},
                upsert=True
            )
        )

    if operations:
        manual_text_collection.bulk_write(operations, ordered=False)


def sync_source_documents(split_docs: list[Document], source: str) -> tuple[list[Document], list[str]]:
    """
    对于同一个source来源的内容,去除旧的内容，更新Mongo，同时返回新增的和去掉的内容
    """
    dedup_docs = {}
    for doc in split_docs:
        unique_id = doc.metadata.get("unique_id")
        if unique_id:
            dedup_docs[unique_id] = doc
    
    existing_ids = {
        item["unique_id"]
        for item in manual_text_collection.find(
            {"metadata.source": source},
            {"unique_id": 1, "_id": 0}
        )
    }
   

    new_ids = set(dedup_docs.keys())
    stale_ids = sorted(existing_ids - new_ids)

    save_2_mongo(list(dedup_docs.values()))

    if stale_ids:
        manual_text_collection.delete_many({"unique_id": {"$in": stale_ids}})

    added_docs = [
        doc for unique_id, doc in dedup_docs.items()
        if unique_id not in existing_ids
    ]
    return added_docs, stale_ids



def merge_docs(bm25_docs: list[Document], milvus_docs: list[Document]) -> list[Document]:
    """
    合并 BM25 和 Milvus 检索结果并去重。

    去重策略与当前检索模块保持一致：
    1. 优先使用 metadata.unique_id
    2. 如果缺少 unique_id，则退化为 source + content_hash
    3. 如果 content_hash 也缺失，则对 page_content 计算临时 hash

    顺序策略：
    - 保留输入顺序
    - 默认先保留 BM25 结果，再追加 Milvus 中尚未出现的结果
    """
    merged_docs = []
    seen = set()

    for doc in list(bm25_docs or []) + list(milvus_docs or []):
        if doc is None:
            continue

        metadata = doc.metadata or {}
        unique_id = metadata.get("unique_id")

        if unique_id:
            dedup_key = ("unique_id", unique_id)
        else:
            source = metadata.get("source", "")
            content_hash = metadata.get("content_hash")
            if not content_hash:
                content_hash = hashlib.md5((doc.page_content or "").strip().encode("utf-8")).hexdigest()
            dedup_key = ("fallback", source, content_hash)

        if dedup_key in seen:
            continue

        seen.add(dedup_key)
        merged_docs.append(doc)

    return merged_docs


def post_processing(response, docs):
    #提取出来think和answer这两部分
    def _extract_tag(text: str, tag: str) -> str:
        pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
        match = re.search(pattern, text or "", flags=re.S | re.I)
        if not match:
            return ""
        return match.group(1).strip()
    answer_block = _extract_tag(response, "answer")
    if not answer_block:
        answer_block = re.sub(r"<think>.*?</think>", "", response, flags=re.S | re.I)
        answer_block = re.sub(r"<answer>|</answer>", "", answer_block, flags=re.I)
    
    #提取引用编号
    all_cites = re.findall("[【](.*?)[】]", answer_block)
    cites = []
    for cite in all_cites:
        cite = re.sub("[{} 【】]", "", cite)
        cite = cite.replace(",", "，")
        cite = [int(k) for k in cite.split("，") if k.isdigit()]
        cites.extend(cite)
    
    #引用编号去重
    ordered_cites = []
    seen_cites = set()
    for cite in cites:
        if cite <= 0 or cite in seen_cites:
            continue
        seen_cites.add(cite)
        ordered_cites.append(cite)

    answer = re.sub("[【](.*?)[】]", "", answer_block)
    answer = re.sub("[{}【】]", "", answer).strip()
    
    #从文档中提取页码
    def _normalize_page(meta: Dict[str, Any]) -> Optional[int]:
        for key in ("orig_page_no", "page_no", "page"):
            value = meta.get(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                return value
            if isinstance(value, float) and value.is_integer():
                return int(value)
            if isinstance(value, str) and value.strip().isdigit():
                return int(value.strip())
        return None

    def _normalize_image_item(item: Dict[str, Any], fallback_meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(item, dict):
            return None

        image_path = item.get("image_path") or item.get("path") or item.get("img_path")
        title = item.get("title") or item.get("caption_text") or item.get("caption_label") or ""
        if not image_path and not title:
            return None

        page_value = (
            item.get("orig_page_no")
            or item.get("page_no")
            or item.get("page")
            or _normalize_page(fallback_meta)
        )

        return {
            "title": title,
            "caption_label": item.get("caption_label", ""),
            "caption_text": item.get("caption_text") or item.get("title", ""),
            "image_path": image_path,
            "page_no": page_value,
            "source": item.get("source") or fallback_meta.get("source"),
        }

    def _collect_images_from_meta(meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        images: List[Dict[str, Any]] = []

        for figure in meta.get("figure_refs") or []:
            normalized = _normalize_image_item(figure, meta)
            if normalized:
                images.append(normalized)

        for image in meta.get("images_info") or []:
            normalized = _normalize_image_item(image, meta)
            if normalized:
                images.append(normalized)

        return images

    def _fetch_mongo_metadata(meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        collected: List[Dict[str, Any]] = []
        seen_ids = set()

        def _try_add(doc_item: Optional[Dict[str, Any]]):
            if not doc_item:
                return
            doc_meta = doc_item.get("metadata") or {}
            doc_uid = doc_item.get("unique_id") or doc_meta.get("unique_id")
            dedup_key = doc_uid or (
                doc_meta.get("source"),
                doc_meta.get("orig_page_no"),
                doc_meta.get("page_no"),
            )
            if dedup_key in seen_ids:
                return
            seen_ids.add(dedup_key)
            collected.append(doc_meta)

        try:
            for uid in (meta.get("parent_id"), meta.get("unique_id")):
                if not uid or uid in seen_ids:
                    continue
                _try_add(
                    manual_text_collection.find_one(
                        {"unique_id": uid},
                        {"_id": 0, "unique_id": 1, "metadata": 1},
                    )
                )

            source = meta.get("source")
            orig_page_no = meta.get("orig_page_no")
            page_no = meta.get("page_no")
            if source and (orig_page_no is not None or page_no is not None):
                page_query = {"metadata.source": source}
                if orig_page_no is not None:
                    page_query["metadata.orig_page_no"] = orig_page_no
                elif page_no is not None:
                    page_query["metadata.page_no"] = page_no

                cursor = manual_text_collection.find(
                    page_query,
                    {"_id": 0, "unique_id": 1, "metadata": 1},
                ).limit(10)
                for item in cursor:
                    _try_add(item)
        except Exception:
            return collected

        return collected

    page_set = set()
    image_key_set = set()
    related_images = []
    cited_docs = []
    cited_doc_keys = set()

    def _consume_metadata(meta: Dict[str, Any]):
        if not isinstance(meta, dict):
            return

        page = _normalize_page(meta)
        if page is not None:
            page_set.add(page)

        for image in _collect_images_from_meta(meta):
            dedup_key = (
                image.get("image_path"),
                image.get("caption_label"),
                image.get("caption_text"),
                image.get("page_no"),
            )
            if dedup_key in image_key_set:
                continue
            image_key_set.add(dedup_key)
            related_images.append(image)

    for index in ordered_cites:
        if index > len(docs):
            continue

        doc = docs[index - 1]
        metadata = doc.metadata or {}
        _consume_metadata(metadata)

        doc_key = metadata.get("unique_id") or (metadata.get("source"), metadata.get("page_no"), doc.page_content)
        if doc_key not in cited_doc_keys:
            cited_doc_keys.add(doc_key)
            cited_docs.append({
                "rank": index,
                "page_content": doc.page_content,
                "metadata": metadata,
            })

        for mongo_meta in _fetch_mongo_metadata(metadata):
            _consume_metadata(mongo_meta)

    return {
        "answer": answer,
        "cite_pages": sorted(page_set),
        "related_images": related_images,
        "cited_docs": cited_docs,
        "raw_response": response,
    }


def __main__():
    
    translator = TranslatorZh2En()

    t0 = time.time()
    query = "你好，世界！"
    query_en = translator.translate(query)
    print(query_en)
    print("1.Translation time:", time.time() - t0)
    
    query = "今天天气怎么样？"
    query_en = translator.translate(query)
    print(query_en)
    print("2.Translation time:", time.time() - t0)


    