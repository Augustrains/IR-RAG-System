# 路径都以 main.py 为参考路径
base_path="/root/autodl-tmp/IR-RAG-System/"

#排序模型
Qwen3_Reranker_path=base_path+"models/Qwen3-Reranker-4B"
bge_reranker_tuned_model_path=base_path+"models/bge-reranker-v2-m3"



#检索
bm25_pickle_path=base_path+"data/saved_index/bm25_retriever.pkl"
bge_m3_model_path = base_path + "models/bge-m3"
milvus_db_path = base_path + "data/saved_index/milvus.db"

#回答模型
qwen3_8b_base_model_path=base_path+"models/Qwen3-8B"
qwen3_8b_lora_model_name="ir-rag-lora"
qwen3_8b_tune_model_name=qwen3_8b_lora_model_name



#评测数据
eval_bm25_path=base_path+"data/evaluate_data/bm25_eval.jsonl"
cache_dir=base_path+"hf_cache"


#数据路径
test_doc_path = base_path + "data/test_docs.txt"
stopwords_path=base_path+"data/stopwords.txt"
raw_docs_path = base_path + "data/processed_docs/raw_docs"
clean_docs_path = base_path + "data/processed_docs/clean_docs"
split_docs_path = base_path + "data/processed_docs/split_docs"
raw_docs = base_path + "data/pdf/irbook.pdf"
split_docs=base_path + "data/pdf/finnal_irbook.pdf"
final_split_docs=base_path+"data/processed_docs/final_split_docs"
cleaned_docs=base_path+"data/processed_docs/cleaned_docs"
saved_images = base_path + "data/saved_images"
rule_save_path=base_path+"data/rules/rule"
merged_docs=base_path+"data/processed_docs/merged_docs"
mongo_docs=base_path+"data/processed_docs/mongo_docs"