import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("IR_RAG_CACHE_ROOT", "/root/autodl-tmp/IR-RAG-System/.cache")
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(os.environ["IR_RAG_CACHE_ROOT"], "xdg"))
os.environ.setdefault("HF_HOME", os.path.join(os.environ["IR_RAG_CACHE_ROOT"], "huggingface"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(os.environ["HF_HOME"], "transformers"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(os.environ["HF_HOME"], "datasets"))
os.environ.setdefault("MODELSCOPE_CACHE", os.path.join(os.environ["IR_RAG_CACHE_ROOT"], "modelscope"))
os.environ.setdefault("PIP_CACHE_DIR", os.path.join(os.environ["IR_RAG_CACHE_ROOT"], "pip"))
os.environ.setdefault("TMPDIR", os.path.join(os.environ["IR_RAG_CACHE_ROOT"], "tmp"))
os.environ.setdefault("TEMP", os.environ["TMPDIR"])
os.environ.setdefault("TMP", os.environ["TMPDIR"])
for _cache_dir in (
    os.environ["XDG_CACHE_HOME"],
    os.environ["HF_HOME"],
    os.environ["HUGGINGFACE_HUB_CACHE"],
    os.environ["TRANSFORMERS_CACHE"],
    os.environ["HF_DATASETS_CACHE"],
    os.environ["MODELSCOPE_CACHE"],
    os.environ["PIP_CACHE_DIR"],
    os.environ["TMPDIR"],
):
    os.makedirs(_cache_dir, exist_ok=True)
import json
import pickle
import time
from src.retriever.bm25_retriever import BM25
from src.retriever.milvus_retriever import MilvusRetriever 
from src.client.llm_local_client import request_chat
# from src.reranker.qwen3_4B_reranker import QWenReRanker   
from src.reranker.bge_m3_reranker import BGEM3ReRanker 
from src.path import bge_reranker_tuned_model_path
from src.utils import merge_docs, post_processing

# warmstart
bm25_retriever = BM25(docs=None)
milvus_retriever = MilvusRetriever(docs=None) 
bge_m3_reranker = BGEM3ReRanker(model_path=bge_reranker_tuned_model_path)
# qwen3_4b_reranker = QWenReRanker(model_path=Qwen3_Reranker_path)#用于精排
milvus_retriever.retrieve_topk("这是一条测试数据", topk=3)


def build_doc_context(idx, doc):
    metadata = doc.metadata or {}
    page_no = metadata.get("orig_page_no") or metadata.get("page_no") or metadata.get("page") or "未知"
    chunk_level = metadata.get("chunk_level") or "unknown"
    source = metadata.get("source") or ""
    figure_refs = metadata.get("figure_refs") or metadata.get("images_info") or []
    footnotes = metadata.get("related_footnotes") or []

    lines = [
        f"【{idx}】",
        f"页码: {page_no}",
        f"分块层级: {chunk_level}",
    ]
    if source:
        lines.append(f"来源: {source}")

    lines.append("正文:")
    lines.append(doc.page_content)

    if figure_refs:
        lines.append("图表信息:")
        for figure in figure_refs:
            if not isinstance(figure, dict):
                continue
            fig_page = figure.get("orig_page_no") or figure.get("page_no") or page_no
            fig_label = figure.get("caption_label") or ""
            fig_text = figure.get("caption_text") or figure.get("title") or ""
            fig_path = figure.get("image_path") or figure.get("path") or figure.get("img_path") or ""
            lines.append(f"- 页码: {fig_page}; 标识: {fig_label}; 描述: {fig_text}; 路径: {fig_path}")

    if footnotes:
        lines.append("脚注信息:")
        for footnote in footnotes:
            lines.append(f"- {footnote}")

    return "\n".join(lines)


def build_context(docs):
    return "\n\n".join(build_doc_context(idx + 1, doc) for idx, doc in enumerate(docs))


while True:
    query = input("输入—>")

    # 检索
    # BM25召回
    t1 = time.time()
    bm25_docs = bm25_retriever.retrieve_topk(query, topk=3)
    # print("BM25召回样例:")
    # print(bm25_docs)
    # print("="*100)
    # t2 = time.time()


    # BGE-M3稠密+稀疏召回+RRF初排
    milvus_docs = milvus_retriever.retrieve_topk(query, topk=3)
    # print("BGE-M3召回样例:")
    # print(milvus_docs)
    # print("="*100)
    # t3 = time.time()


    #去重
    merged_docs = merge_docs(bm25_docs, milvus_docs)


    #精排 
    ranked_docs = bge_m3_reranker.rank(query, merged_docs, topk=5)
    # print(len(ranked_docs))
    # print("="*100)


    #答案
    context = build_context(ranked_docs)
    print("query chars =", len(query))
    print("context chars =", len(context))
    llm_start = time.time()
    res_handler = request_chat(query, context, stream=True)
    response = ""
    first_visible_token_at = None
    for r in res_handler:
        delta = r.choices[0].delta
        uttr = getattr(delta, "content", None) or ""

        # Some backends may stream hidden reasoning fields before visible answer tokens.
        # We only show final answer content here, but keep waiting for visible content.
        if not uttr:
            continue

        if first_visible_token_at is None:
            first_visible_token_at = time.time()
            print(f"first visible token latency = {first_visible_token_at - llm_start:.2f}s")

        response += uttr
        print(uttr, end="")
    print()
    print("llm total time =", round(time.time() - llm_start, 2), "s")
    print("=" * 100)

    #后处理
    result = post_processing(response, ranked_docs)
    result_payload = {
        "query": query,
        "thought": result.get("thought", ""),
        "answer": result.get("answer", ""),
        "cite_pages": result.get("cite_pages", []),
        "related_images": result.get("related_images", []),
        "cited_docs": result.get("cited_docs", []),
    }

    print(json.dumps(result_payload, ensure_ascii=False, indent=2))
