import torch
from typing import Sequence, Optional
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class BGEM3ReRanker(BaseDocumentCompressor):
    def __init__(self, model_path: str, max_length: int = 4096, topk: int = 10):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.model.half()
        self.model.cuda()
        self.max_length = max_length
        self.topk = topk

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[list] = None,
    ) -> Sequence[Document]:
        if not documents:
            return []
        pairs = [(query, doc.page_content) for doc in documents]
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
        ).to("cuda")
        with torch.no_grad():
            scores = self.model(**inputs).logits
        scores = scores.detach().cpu().clone().numpy()
        response = [
            doc
            for score, doc in sorted(
                zip(scores, documents), reverse=True, key=lambda x: x[0]
            )
        ][: self.topk]
        return response

    def rank(self, query: str, candidate_docs: Sequence[Document], topk: int = 10):
        self.topk = topk
        return self.compress_documents(candidate_docs, query)


if __name__ == "__main__":
    bge_reranker_large = "./models/BAAI/bge-reranker-v2-m3/"
    # bce_reranker_base = "../../models/bce-reranker-base-v1"
    bge_rerank = BGEM3ReRanker(bge_reranker_large)
    query = "今天天气怎么样"
    docs = ["你好", "今天天气不错", "今天有雨吗"]
    docs = [Document(page_content=doc, metadata={}) for doc in docs]
    response = bge_rerank.rank(query, docs)
    print(response)
