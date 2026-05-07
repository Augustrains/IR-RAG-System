import time
import hashlib
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    AnnSearchRequest,
    RRFRanker,
)
from langchain_core.documents import Document
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from src.path import test_doc_path, bge_m3_model_path, milvus_db_path
from src.client.mongodb_config import MongoConfig
from src.utils import (
    build_unique_id,
    TranslatorZh2En,
    TranslatorEn2Zh,
)

EMB_BATCH = 50
MAX_TEXT_LENGTH = 512
ID_MAX_LENGTH = 100

ZH_COL_NAME = "hybrid_bge_m3_zh"
EN_COL_NAME = "hybrid_bge_m3_en"

mongo_collection = MongoConfig.get_collection("manual_text")
connections.connect(uri=milvus_db_path)
embedding_handler = BGEM3EmbeddingFunction(
    model_name=bge_m3_model_path,
    device="cuda"
)


class MilvusRetriever:
    """
    双语 Milvus 检索器：
    1. 文档按中英文分别建 collection
    2. 查询时双路检索：
       - 原语言直查
       - 翻译后跨语言补查
    3. 合并、去重、映射 parent
    """

    def __init__(self, docs: Optional[Sequence[Document]] = None, rebuild: bool = False):
        self.translator_zh2en = TranslatorZh2En()
        self.translator_en2zh = TranslatorEn2Zh()

        self.zh_col = self._prepare_collection(ZH_COL_NAME, rebuild=rebuild)
        self.en_col = self._prepare_collection(EN_COL_NAME, rebuild=rebuild)

        if docs:
            self.update_vectorstore(list(docs))

    # =========================
    # Collection 初始化
    # =========================
    def _prepare_collection(self, col_name: str, rebuild: bool = False) -> Collection:
        fields = [
            FieldSchema(
                name="unique_id",
                dtype=DataType.VARCHAR,
                max_length=ID_MAX_LENGTH,
                is_primary=True,
            ),
            FieldSchema(
                name="dense_vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=embedding_handler.dim["dense"],
            ),
            FieldSchema(
                name="sparse_vector",
                dtype=DataType.SPARSE_FLOAT_VECTOR,
            ),
            FieldSchema(
                name="source",
                dtype=DataType.VARCHAR,
                max_length=1024,
            ),
        ]
        schema = CollectionSchema(
            fields,
            description=f"Hybrid search collection for {col_name}",
        )

        sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
        dense_index = {"index_type": "AUTOINDEX", "metric_type": "IP"}

        if rebuild:
            if utility.has_collection(col_name):
                utility.drop_collection(col_name)

            col = Collection(col_name, schema=schema, consistency_level="Strong")
            col.create_index("sparse_vector", sparse_index)
            col.create_index("dense_vector", dense_index)
        else:
            if utility.has_collection(col_name):
                col = Collection(col_name, consistency_level="Strong")
            else:
                col = Collection(col_name, schema=schema, consistency_level="Strong")
                col.create_index("sparse_vector", sparse_index)
                col.create_index("dense_vector", dense_index)

        col.load()
        return col

    # =========================
    # 语言判定
    # =========================
    @staticmethod
    def detect_text_language(text: str) -> str:
        text = (text or "").strip()
        if not text:
            return "en"

        english_chars = sum(c.isascii() and c.isalpha() for c in text)
        chinese_chars = sum("\u4e00" <= c <= "\u9fff" for c in text)

        return "zh" if chinese_chars > english_chars else "en"

    # =========================
    # 文档入库
    # =========================
    def update_vectorstore(self, docs: List[Document]):
        """
        文档按语言分桶：
        - 中文文档更新到 zh_col
        - 英文文档更新到 en_col
        """
        dedup_docs = {}
        skipped = 0

        for doc in docs:
            metadata = doc.metadata or {}
            uid = metadata.get("unique_id")
            source = metadata.get("source")

            if not uid or not source:
                skipped += 1
                continue

            lang = metadata.get("lang")
            if not lang:
                lang = self.detect_text_language(doc.page_content)
                metadata["lang"] = lang
                doc.metadata = metadata

            dedup_docs[uid] = doc

        if not dedup_docs:
            print("没有有效文档:(全部缺少 unique_id 或 source)")
            return

        zh_docs = []
        en_docs = []
        for doc in dedup_docs.values():
            lang = (doc.metadata or {}).get("lang", "en")
            if lang == "zh":
                zh_docs.append(doc)
            else:
                en_docs.append(doc)

        inserted_zh = self._update_one_language_collection(self.zh_col, zh_docs, lang="zh")
        inserted_en = self._update_one_language_collection(self.en_col, en_docs, lang="en")

        print(
            f"向量库更新完成：共处理 {len(dedup_docs)} 条有效文档，"
            f"跳过 {skipped} 条无效文档，"
            f"中文插入 {inserted_zh} 条，英文插入 {inserted_en} 条，"
            f"中文集合总数 {self.zh_col.num_entities}，英文集合总数 {self.en_col.num_entities}"
        )

    def _update_one_language_collection(
        self,
        col: Collection,
        docs: List[Document],
        lang: str,
    ) -> int:
        if not docs:
            return 0

        docs_by_source = defaultdict(list)
        for doc in docs:
            source = doc.metadata["source"]
            docs_by_source[source].append(doc)

        total_inserted = 0

        for source, source_docs in docs_by_source.items():
            safe_source = source.replace("\\", "\\\\").replace('"', '\\"')
            expr = f'source == "{safe_source}"'
            col.delete(expr)
            col.flush()

            raw_texts = [doc.page_content for doc in source_docs]
            unique_ids = [doc.metadata["unique_id"] for doc in source_docs]
            sources = [doc.metadata["source"] for doc in source_docs]

            texts_embeddings = embedding_handler(raw_texts)

            for i in range(0, len(source_docs), EMB_BATCH):
                batch_entities = [
                    unique_ids[i:i + EMB_BATCH],
                    texts_embeddings["dense"][i:i + EMB_BATCH],
                    texts_embeddings["sparse"][i:i + EMB_BATCH],
                    sources[i:i + EMB_BATCH],
                ]
                col.insert(batch_entities)

            total_inserted += len(source_docs)
            print(f"[Milvus-{lang}] source={source} 更新完成，当前插入 {len(source_docs)} 条")

        col.flush()
        return total_inserted

    # =========================
    # 单路 hybrid search
    # =========================
    def _hybrid_search_one_collection(
        self,
        col: Collection,
        query: str,
        limit: int = 10,
    ):
        if not query.strip():
            return [[]]

        query_embedding = embedding_handler.encode_queries([query])
        query_dense_embedding = query_embedding["dense"][0]
        query_sparse_embedding = query_embedding["sparse"][[0]]

        dense_search_params = {"metric_type": "IP", "params": {}}
        dense_req = AnnSearchRequest(
            [query_dense_embedding],
            "dense_vector",
            dense_search_params,
            limit=limit,
        )

        sparse_search_params = {"metric_type": "IP", "params": {}}
        sparse_req = AnnSearchRequest(
            [query_sparse_embedding],
            "sparse_vector",
            sparse_search_params,
            limit=limit,
        )

        rerank = RRFRanker()
        res = col.hybrid_search(
            [sparse_req, dense_req],
            rerank=rerank,
            limit=limit,
            output_fields=["unique_id"],
        )
        return res

    # =========================
    # 双路检索
    # =========================
    def _dual_route_search(self, query: str, limit: int) -> List[str]:
        """
        返回双路召回后的 unique_id 列表（已去重前）
        """
        query_lang = self.detect_text_language(query)
        all_hit_uids: List[str] = []

        if query_lang == "zh":
            # 中文直查中文库
            zh_res = self._hybrid_search_one_collection(self.zh_col, query, limit=limit)
            for hit in zh_res[0]:
                uid = hit.get("unique_id")
                if uid:
                    all_hit_uids.append(uid)

            # 中文翻译成英文查英文库
            en_query = self.translator_zh2en.translate(query)
            en_res = self._hybrid_search_one_collection(self.en_col, en_query, limit=limit)
            for hit in en_res[0]:
                uid = hit.get("unique_id")
                if uid:
                    all_hit_uids.append(uid)

        else:
            # 英文直查英文库
            en_res = self._hybrid_search_one_collection(self.en_col, query, limit=limit)
            for hit in en_res[0]:
                uid = hit.get("unique_id")
                if uid:
                    all_hit_uids.append(uid)

            # 英文翻译成中文查中文库
            zh_query = self.translator_en2zh.translate(query)
            zh_res = self._hybrid_search_one_collection(self.zh_col, zh_query, limit=limit)
            for hit in zh_res[0]:
                uid = hit.get("unique_id")
                if uid:
                    all_hit_uids.append(uid)

        return all_hit_uids

    # =========================
    # 最终检索接口
    # =========================
    def retrieve_topk(self, query: str, topk: int = 10) -> List[Document]:
        """
        1. 双路 hybrid search
        2. 根据 uid 从 Mongo 回查
        3. child / parent 统一映射回 parent
        4. 合并去重
        5. 返回 topk
        """
        fetch_limit = max(topk * 4, topk)
        all_hit_uids = self._dual_route_search(query, limit=fetch_limit)

        related_docs = []
        seen_parent_ids = set()
        seen_hit_uids = set()

        for hit_uid in all_hit_uids:
            if not hit_uid or hit_uid in seen_hit_uids:
                continue
            seen_hit_uids.add(hit_uid)

            search_res = mongo_collection.find_one(
                {"unique_id": hit_uid},
                {"_id": 0, "page_content": 1, "metadata": 1, "unique_id": 1},
            )
            if not search_res:
                continue

            hit_meta = search_res.get("metadata") or {}
            parent_uid = hit_meta.get("parent_id") or hit_meta.get("unique_id") or hit_uid
            if parent_uid in seen_parent_ids:
                continue

            parent_res = mongo_collection.find_one(
                {"unique_id": parent_uid},
                {"_id": 0, "page_content": 1, "metadata": 1, "unique_id": 1},
            )

            final_res = parent_res or search_res
            final_meta = final_res.get("metadata") or {}
            final_uid = (
                final_meta.get("unique_id")
                or final_res.get("unique_id")
                or parent_uid
            )

            if final_uid in seen_parent_ids:
                continue

            seen_parent_ids.add(final_uid)

            doc = Document(
                page_content=final_res.get("page_content", ""),
                metadata=final_meta,
            )
            related_docs.append(doc)

            if len(related_docs) >= topk:
                break

        return related_docs

    # =========================
    # 调试辅助：获取所有已索引文档
    # =========================
    def _get_all_indexed_docs_from_one_collection(self, col: Collection, batch_size: int = 1000):
        all_docs = []
        offset = 0

        while True:
            results = col.query(
                expr='unique_id != ""',
                output_fields=["unique_id"],
                limit=batch_size,
                offset=offset,
            )

            if not results:
                break

            unique_ids = [item.get("unique_id") for item in results if item.get("unique_id")]
            if not unique_ids:
                break

            mongo_results = list(
                mongo_collection.find(
                    {"unique_id": {"$in": unique_ids}},
                    {"_id": 0, "page_content": 1, "metadata": 1, "unique_id": 1},
                )
            )

            mongo_map = {item["unique_id"]: item for item in mongo_results}

            for uid in unique_ids:
                item = mongo_map.get(uid)
                if not item:
                    continue

                all_docs.append(
                    Document(
                        page_content=item["page_content"],
                        metadata=item["metadata"],
                    )
                )

            if len(results) < batch_size:
                break

            offset += batch_size

        return all_docs

    def get_all_indexed_docs(self, batch_size: int = 1000):
        zh_docs = self._get_all_indexed_docs_from_one_collection(self.zh_col, batch_size=batch_size)
        en_docs = self._get_all_indexed_docs_from_one_collection(self.en_col, batch_size=batch_size)
        return zh_docs + en_docs


if __name__ == "__main__":
    texts = [k.strip() for k in open(test_doc_path, encoding="utf-8").readlines() if k.strip()]
    docs = []

    source = test_doc_path
    for text in texts:
        unique_id = build_unique_id(source, text, chunk_type="demo")

        # 自动判断文档语言
        lang = MilvusRetriever.detect_text_language(text)

        metadata = {
            "unique_id": unique_id,
            "source": source,
            "chunk_type": "demo",
            "lang": lang,
            "content_hash": hashlib.md5(text.encode("utf-8")).hexdigest(),
        }
        docs.append(Document(page_content=text, metadata=metadata))

    retriever = MilvusRetriever(docs=docs, rebuild=False)

    query = "Model3支持的钥匙类型"
    results = retriever.retrieve_topk(query, topk=2)
    print("中文问题检索结果：")
    for res in results:
        print(res)
        print("=" * 100)

    query2 = "What key types are supported by Model 3?"
    results2 = retriever.retrieve_topk(query2, topk=2)
    print("英文问题检索结果：")
    for res in results2:
        print(res)
        print("=" * 100)
