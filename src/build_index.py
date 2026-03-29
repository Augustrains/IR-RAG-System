import os
import json
from typing import List
from src.retriever.bm25_retriever import BM25
from src.retriever.milvus_retriever import MilvusRetriever
from src.retriever.old_bm25_retriever import BM25 as OLD_BM25
from src.retriever.old_milvus_retriever import MilvusRetriever as OLD_MilvusRetriever
from src.utils import sync_source_documents
from langchain.schema import Document
from src.chunking.chunk_pipeline import process_pages_to_chunks
from src.document_merge.merge import fix_cross_page_units
from src.document_split.document_splitter import load_pdf
from src import path

row_docs_path = path.split_docs
cleaned_docs_path = path.cleaned_docs  #去除页眉页脚
save_images_path = path.saved_images   
save_rules_path = path.rule_save_path + ".json" 
merge_docs_path = path.merged_docs   #跨页合并
split_docs_path=path.final_split_docs
mongo_docs_path=path.mongo_docs      #存放给mongo的数据
start_page = 38   #起始页
"""
PDF -> 清洗 -> 语义切分 -> MongoDB -> BM25/Milvus索引
"""

def load_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_documents_to_jsonl(docs: List[Document], file_path: str):
    """
    将 Document 列表保存为 jsonl 文件（每行一个 doc）
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        for doc in docs:
            row = {
                "page_content": doc.page_content,
                "metadata": doc.metadata or {},
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def load_jsonl_as_documents(file_path: str) -> List[Document]:
    """
    从 jsonl 文件读取，并恢复为 Document 列表
    """
    docs = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)

            docs.append(
                Document(
                    page_content=item.get("page_content", ""),
                    metadata=item.get("metadata", {}) or {}
                )
            )
    return docs

# 解析pdf
if not os.path.exists(cleaned_docs_path) or not os.path.exists(save_rules_path):
    json_data, learned_rules = load_pdf(row_docs_path, start_page)
    print("清洗后的文档数:", len(json_data))
    with open(cleaned_docs_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    if save_rules_path:
        with open(save_rules_path, "w", encoding="utf-8") as f:
            json.dump(learned_rules, f, ensure_ascii=False, indent=2)
        print(f"\n已保存规则到: {save_rules_path}")
else:
    json_data = load_json_file(cleaned_docs_path)
    print("加载清洗文档数:", len(json_data))

# 文档合并
if not os.path.exists(merge_docs_path):
    merged_pages = fix_cross_page_units(json_data)
    print("合并后的文档数:", len(merged_pages))
    with open(merge_docs_path, "w", encoding="utf-8") as f:
        json.dump(merged_pages, f, ensure_ascii=False, indent=2)
else:
    merged_pages = load_json_file(merge_docs_path)
    print("加载合并后的文档数:", len(merged_pages))

# 文档切分
if not os.path.exists(split_docs_path) or not os.path.exists(mongo_docs_path):
    split_docs, mongo_docs = process_pages_to_chunks(merged_pages)
    print("解析后文档总数:", len(split_docs))
    save_documents_to_jsonl(split_docs, split_docs_path)
    save_documents_to_jsonl(mongo_docs,mongo_docs_path)
else:
    split_docs = load_jsonl_as_documents(split_docs_path)
    mongo_docs=load_jsonl_as_documents(mongo_docs_path)
    print("加载解析后的文档数:", len(split_docs))

#更新mongo
source = row_docs_path
added_docs, stale_ids = sync_source_documents(mongo_docs, source)
print("新增chunk数:", len(added_docs))
print("失效chunk数:", len(stale_ids))

# #更新检索器
milvus_retriever = MilvusRetriever(split_docs,rebuild=True)
new_docs=milvus_retriever.get_all_indexed_docs()
print("Milvus检索器中的文档数:", len(new_docs))
bm25_retriever = BM25(new_docs)


#更新旧检索器,用于评测对比
# old_bem25=OLD_BM25(new_docs)
# old_milvus=OLD_MilvusRetriever(split_docs)

# 简单验证
candidate_docs = milvus_retriever.retrieve_topk("信息检索系统可以根据规模分为哪几类？各自有什么特点？", topk=3)
print("BGE-M3召回样例:")
print(candidate_docs)
candidate_docs = bm25_retriever.retrieve_topk("信息检索系统可以根据规模分为哪几类？各自有什么特点？", topk=3)
print("BM25召回样例:")
print(candidate_docs)