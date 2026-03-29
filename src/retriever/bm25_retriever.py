"""
双语 BM25 检索器

功能：
1. 文档按中英文分别存放并建立两个 BM25 索引
2. 查询时双路检索：
   - 原语言检索一次
   - 翻译成另一种语言再检索一次
3. 合并结果、去重、映射 parent
4. 可选接入外部 reranker

说明：
- 中文文档 → zh_retriever
- 英文文档 → en_retriever
- 中文问题 → 中文直查 + 翻译成英文查英文
- 英文问题 → 英文直查 + 翻译成中文查中文
"""

import re
import os
import pickle
import jieba
import hashlib
from typing import Any, Callable, Dict, List, Optional, Sequence

from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever

from src.path import bm25_pickle_path, stopwords_path
from src.utils import build_unique_id

# 你需要在 src.utils 中实现这两个翻译器
# 若你已有 TranslatorZh2En，则补一个 TranslatorEn2Zh 即可
from src.utils import TranslatorZh2En, TranslatorEn2Zh


DOCS_PICKLE_PATH = bm25_pickle_path.replace(".pkl", "_docs.pkl")
ZH_INDEX_PICKLE_PATH = bm25_pickle_path.replace(".pkl", "_zh_index.pkl")
EN_INDEX_PICKLE_PATH = bm25_pickle_path.replace(".pkl", "_en_index.pkl")

MIN_TOKEN_LENGTH = 1

# 加载停用词
with open(stopwords_path, "r", encoding="utf-8") as fd:
    _stopwords = set(t.strip().lower() for t in fd if t.strip())


class BM25:
    def __init__(self, docs: Optional[Sequence[Document]] = None):
        """
        1. docs=None:
           直接加载本地保存的 documents 和双语 BM25 索引
        2. docs 不为 None:
           传入的是最新全量文档集合
           会覆盖 self.documents，并全量重建双语 BM25 索引
        """
        self.zh_retriever: Optional[BM25Retriever] = None
        self.en_retriever: Optional[BM25Retriever] = None

        self.translator_zh2en = TranslatorZh2En()
        self.translator_en2zh = TranslatorEn2Zh()

        self.documents: List[Document] = []
        self.zh_documents: List[Document] = []
        self.en_documents: List[Document] = []

        self.doc_store: Dict[str, Document] = {}

        if docs is not None:
            self.documents = list(docs)
            self._split_documents_by_language()
            self.rebuild_doc_store()
            self.rebuild_retrievers()
        else:
            self.documents = self.load_documents() if os.path.exists(DOCS_PICKLE_PATH) else []
            self._split_documents_by_language()
            self.rebuild_doc_store()
            self.load_or_rebuild_retrievers()

    # =========================
    # 基础存取
    # =========================
    def load_documents(self) -> List[Document]:
        with open(DOCS_PICKLE_PATH, "rb") as f:
            return pickle.load(f)

    def save_documents(self) -> None:
        with open(DOCS_PICKLE_PATH, "wb") as f:
            pickle.dump(self.documents, f)

    def save_retrievers(self) -> None:
        with open(ZH_INDEX_PICKLE_PATH, "wb") as f:
            pickle.dump(self.zh_retriever, f)
        with open(EN_INDEX_PICKLE_PATH, "wb") as f:
            pickle.dump(self.en_retriever, f)

    def load_or_rebuild_retrievers(self) -> None:
        zh_ok = os.path.exists(ZH_INDEX_PICKLE_PATH)
        en_ok = os.path.exists(EN_INDEX_PICKLE_PATH)

        if zh_ok:
            with open(ZH_INDEX_PICKLE_PATH, "rb") as f:
                self.zh_retriever = pickle.load(f)
        if en_ok:
            with open(EN_INDEX_PICKLE_PATH, "rb") as f:
                self.en_retriever = pickle.load(f)

        if not zh_ok and self.zh_documents:
            self.zh_retriever = BM25Retriever.from_documents(
                self.zh_documents,
                preprocess_func=self.tokenize,
            )

        if not en_ok and self.en_documents:
            self.en_retriever = BM25Retriever.from_documents(
                self.en_documents,
                preprocess_func=self.tokenize,
            )

        if (self.zh_documents and not zh_ok) or (self.en_documents and not en_ok):
            self.save_retrievers()

    # =========================
    # 文档处理
    # =========================
    def rebuild_doc_store(self) -> None:
        self.doc_store = {}
        for doc in self.documents:
            metadata = doc.metadata or {}
            uid = metadata.get("unique_id")
            if uid:
                self.doc_store[str(uid)] = doc

    def _split_documents_by_language(self) -> None:
        """
        将文档按语言拆分：
        - 中文文档进入 zh_documents
        - 英文文档进入 en_documents

        若 metadata 中已有 lang，则优先使用；
        否则自动检测，并回写到 metadata["lang"]。
        """
        self.zh_documents = []
        self.en_documents = []

        for doc in self.documents:
            text = (doc.page_content or "").strip()
            metadata = dict(doc.metadata or {})

            lang = metadata.get("lang")
            if not lang:
                lang = self.detect_text_language(text)

            metadata["lang"] = lang
            doc.metadata = metadata

            if lang == "zh":
                self.zh_documents.append(doc)
            else:
                self.en_documents.append(doc)

    def rebuild_retrievers(self) -> None:
        """
        根据 self.zh_documents / self.en_documents 全量重建双语 BM25 索引，并持久化
        """
        if self.zh_documents:
            self.zh_retriever = BM25Retriever.from_documents(
                self.zh_documents,
                preprocess_func=self.tokenize,
            )
        else:
            self.zh_retriever = None
            if os.path.exists(ZH_INDEX_PICKLE_PATH):
                os.remove(ZH_INDEX_PICKLE_PATH)

        if self.en_documents:
            self.en_retriever = BM25Retriever.from_documents(
                self.en_documents,
                preprocess_func=self.tokenize,
            )
        else:
            self.en_retriever = None
            if os.path.exists(EN_INDEX_PICKLE_PATH):
                os.remove(EN_INDEX_PICKLE_PATH)

        self.save_documents()
        self.save_retrievers()

    # =========================
    # 文本处理
    # =========================
    def tokenize(self, text: str) -> List[str]:
        text = text.strip().lower()
        text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)

        raw_parts = text.split()
        tokens: List[str] = []

        for part in raw_parts:
            if re.search(r"[\u4e00-\u9fff]", part):
                tokens.extend(jieba.lcut(part))
            else:
                tokens.append(part)

        tokens = [
            t.strip()
            for t in tokens
            if t.strip() and t not in _stopwords and len(t.strip()) >= MIN_TOKEN_LENGTH
        ]
        return tokens

    @staticmethod
    def detect_text_language(text: str) -> str:
        """
        判断文本语言：
        - 中文字符多于英文字符 → zh
        - 否则 → en
        """
        text = text.strip()
        if not text:
            return "en"

        english_chars = sum(c.isascii() and c.isalpha() for c in text)
        chinese_chars = sum("\u4e00" <= c <= "\u9fff" for c in text)

        return "zh" if chinese_chars > english_chars else "en"

    def translate_zh_to_en(self, text: str) -> str:
        return self.translator_zh2en.translate(text)

    def translate_en_to_zh(self, text: str) -> str:
        return self.translator_en2zh.translate(text)

    # =========================
    # 检索相关
    # =========================
    def _invoke_retriever(
        self,
        retriever: Optional[BM25Retriever],
        query: str,
        fetch_limit: int,
    ) -> List[Document]:
        if retriever is None or not query.strip():
            return []
        retriever.k = fetch_limit
        return retriever.invoke(query)

    def deduplicate_documents_for_results(self, all_docs: Sequence[Document]) -> List[Document]:
        """
        检索结果去重：
        1. 优先按 unique_id 去重
        2. 如果缺少 unique_id，则退化为 source + content_hash
        """
        seen = set()
        unique_docs: List[Document] = []

        for doc in all_docs:
            metadata = doc.metadata or {}
            doc_id = metadata.get("unique_id")

            if doc_id:
                dedup_key = ("unique_id", str(doc_id))
            else:
                source = metadata.get("source", "")
                content_hash = metadata.get("content_hash")
                if not content_hash:
                    content_hash = hashlib.md5(
                        doc.page_content.strip().encode("utf-8")
                    ).hexdigest()
                dedup_key = ("fallback", source, content_hash)

            if dedup_key not in seen:
                seen.add(dedup_key)
                unique_docs.append(doc)

        return unique_docs

    def map_to_parent_documents(self, docs: Sequence[Document]) -> List[Document]:
        """
        将检索到的 fine-grained 文档映射回 parent 文档
        """
        parent_docs: List[Document] = []
        seen_parent_ids = set()

        for doc in docs:
            metadata = doc.metadata or {}
            parent_uid = metadata.get("parent_id") or metadata.get("unique_id")
            if not parent_uid:
                continue

            parent_doc = self.doc_store.get(str(parent_uid), doc)
            parent_meta = parent_doc.metadata or {}
            final_uid = parent_meta.get("unique_id") or parent_uid

            if final_uid in seen_parent_ids:
                continue

            seen_parent_ids.add(final_uid)
            parent_docs.append(parent_doc)

        return parent_docs

    def _dual_route_retrieve(self, query: str, topk: int) -> List[Document]:
        """
        双路召回：
        - 原语言直查
        - 翻译后跨语言补查

        中文问题：
            中文查中文库 + 翻译成英文查英文库
        英文问题：
            英文查英文库 + 翻译成中文查中文库
        """
        query_lang = self.detect_text_language(query)
        fetch_limit = max(topk * 4, topk)

        all_docs: List[Document] = []

        if query_lang == "zh":
            # 中文直查中文库
            all_docs.extend(
                self._invoke_retriever(self.zh_retriever, query, fetch_limit)
            )

            # 中文翻译成英文查英文库
            en_query = self.translate_zh_to_en(query)
            all_docs.extend(
                self._invoke_retriever(self.en_retriever, en_query, fetch_limit)
            )

        else:
            # 英文直查英文库
            all_docs.extend(
                self._invoke_retriever(self.en_retriever, query, fetch_limit)
            )

            # 英文翻译成中文查中文库
            zh_query = self.translate_en_to_zh(query)
            all_docs.extend(
                self._invoke_retriever(self.zh_retriever, zh_query, fetch_limit)
            )

        return all_docs

    def retrieve_topk(
        self,
        query: str,
        topk: int = 5,
        reranker: Optional[Callable[[str, List[Document]], List[Document]]] = None,
    ) -> List[Document]:
        """
        最终检索流程：
        1. 双路召回
        2. 去重
        3. 映射回 parent
        4. 可选 rerank
        5. 返回 topk
        """
        all_docs = self._dual_route_retrieve(query=query, topk=topk)
        fine_grained_docs = self.deduplicate_documents_for_results(all_docs)
        parent_docs = self.map_to_parent_documents(fine_grained_docs)

        if reranker is not None and parent_docs:
            parent_docs = reranker(query, parent_docs)

        return parent_docs[:topk]


if __name__ == "__main__":
    texts = [
        ("打开车窗", "zh"),
        ("空调加热", "zh"),
        ("加热座椅", "zh"),
        ("heated seats", "en"),
        ("window defogger", "en"),
        ("air conditioner heating", "en"),
    ]

    docs = []
    for idx, (text, lang) in enumerate(texts):
        source = f"demo_source_{idx}"
        unique_id = build_unique_id(source, text, chunk_type="demo")
        metadata = {
            "unique_id": unique_id,
            "source": source,
            "chunk_type": "demo",
            "lang": lang,
            "content_hash": hashlib.md5(text.encode("utf-8")).hexdigest(),
        }
        docs.append(Document(page_content=text, metadata=metadata))

    bm25 = BM25(docs)

    print("中文问题检索：")
    zh_res = bm25.retrieve_topk("座椅加热", topk=3)
    for d in zh_res:
        print(d)

    print("\n英文问题检索：")
    en_res = bm25.retrieve_topk("heated seat", topk=3)
    for d in en_res:
        print(d)