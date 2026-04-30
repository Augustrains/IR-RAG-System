#改为使用图索引方式构建稠密索引
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

    # dense_vector 走 HNSW 图索引
    dense_search_params = {
        "metric_type": "IP",
        "params": {
            "ef": max(64, limit),
        },
    }

    dense_req = AnnSearchRequest(
        [query_dense_embedding],
        "dense_vector",
        dense_search_params,
        limit=limit,
    )

    # sparse_vector 仍然走稀疏倒排索引
    sparse_search_params = {
        "metric_type": "IP",
        "params": {},
    }

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
