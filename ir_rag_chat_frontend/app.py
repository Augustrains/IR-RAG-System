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
import time
import uuid
from datetime import datetime
from pathlib import Path
from threading import Lock

from flask import Flask, abort, jsonify, render_template, request, send_file
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from src.retriever.bm25_retriever import BM25
from src.retriever.milvus_retriever import MilvusRetriever
from src.client.llm_local_client import request_chat
from src.reranker.bge_m3_reranker import BGEM3ReRanker
from src.path import bge_reranker_tuned_model_path
from src.utils import merge_docs, post_processing


BASE_DIR = Path("/root/autodl-tmp/IR-RAG-System")
CHAT_ROOT = BASE_DIR / "data" / "chat_sessions"
CHAT_ROOT.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)


class RAGPipeline:
    def __init__(self):
        self._init_lock = Lock()
        self._ready = False
        self.bm25_retriever = None
        self.milvus_retriever = None
        self.bge_m3_reranker = None
        self.chain = None

    def _ensure_ready(self):
        if self._ready:
            return
        with self._init_lock:
            if self._ready:
                return
            self.bm25_retriever = BM25(docs=None, search_kwargs={"k": 3})
            self.milvus_retriever = MilvusRetriever(docs=None, search_kwargs={"k": 3})
            self.bge_m3_reranker = BGEM3ReRanker(model_path=bge_reranker_tuned_model_path, topk=5)
            self.milvus_retriever.retrieve_topk("这是一条测试数据", topk=3)
            self._build_chain()
            self._ready = True

    def _build_chain(self):
        bm25 = self.bm25_retriever
        milvus = self.milvus_retriever
        reranker = self.bge_m3_reranker

        def _generate(inputs: dict) -> dict:
            llm_start = time.time()
            res_handler = request_chat(inputs["query"], inputs["context"], stream=True)
            response = ""
            first_visible_token_latency = None
            for r in res_handler:
                delta = r.choices[0].delta
                uttr = getattr(delta, "content", None) or ""
                if not uttr:
                    continue
                if first_visible_token_latency is None:
                    first_visible_token_latency = round(time.time() - llm_start, 2)
                response += uttr
            return {
                "response": response,
                "first_visible_token_latency": first_visible_token_latency,
                "llm_total_time": round(time.time() - llm_start, 2),
            }

        self.chain = (
            {"query": RunnablePassthrough()}
            | RunnablePassthrough.assign(
                bm25_docs=RunnableLambda(lambda x: bm25.invoke(x["query"])),
                milvus_docs=RunnableLambda(lambda x: milvus.invoke(x["query"])),
            )
            | RunnablePassthrough.assign(
                documents=lambda x: merge_docs(x["bm25_docs"], x["milvus_docs"]),
            )
            | RunnablePassthrough.assign(
                ranked_docs=RunnableLambda(lambda x: reranker.invoke({"documents": x["documents"], "query": x["query"]})),
            )
            | RunnablePassthrough.assign(
                context=lambda x: self.build_context(x["ranked_docs"]),
            )
            | RunnablePassthrough.assign(
                gen=RunnableLambda(_generate),
            )
            | RunnableLambda(lambda x: {
                "query": x["query"],
                **post_processing(x["gen"]["response"], x["ranked_docs"]),
                "debug": {
                    "query_chars": len(x["query"]),
                    "context_chars": len(x["context"]),
                    "first_visible_token_latency": x["gen"]["first_visible_token_latency"],
                    "llm_total_time": x["gen"]["llm_total_time"],
                },
            })
        )

    @staticmethod
    def build_doc_context(idx, doc):
        return str(doc.page_content or "").strip()

    def build_context(self, docs):
        return "\n\n".join(
            part for idx, doc in enumerate(docs)
            if (part := self.build_doc_context(idx + 1, doc))
        )

    def ask(self, query: str) -> dict:
        self._ensure_ready()
        query = (query or "").strip()
        if not query:
            raise ValueError("问题不能为空")
        return self.chain.invoke(query)


pipeline = RAGPipeline()


def chat_file(chat_id: str) -> Path:
    return CHAT_ROOT / f"{chat_id}.json"


def load_chat(chat_id: str) -> dict:
    path = chat_file(chat_id)
    if not path.exists():
        raise FileNotFoundError(f"chat not found: {chat_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_chat_payload(payload: dict) -> None:
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    chat_file(payload["id"]).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/chats")
def api_list_chats():
    chats = []
    for path in sorted(CHAT_ROOT.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            chats.append({
                "id": data.get("id"),
                "title": data.get("title") or "新对话",
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at"),
                "message_count": len(data.get("messages", [])),
            })
        except Exception:
            continue
    return jsonify(chats)


@app.post("/api/chats")
def api_create_chat():
    chat_id = uuid.uuid4().hex
    now = datetime.now().isoformat(timespec="seconds")
    payload = {
        "id": chat_id,
        "title": "新对话",
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }
    save_chat_payload(payload)
    return jsonify(payload)


@app.get("/api/chats/<chat_id>")
def api_get_chat(chat_id):
    try:
        return jsonify(load_chat(chat_id))
    except FileNotFoundError:
        abort(404, description="未找到对应聊天记录")


@app.post("/api/chats/<chat_id>/ask")
def api_ask(chat_id):
    try:
        chat = load_chat(chat_id)
    except FileNotFoundError:
        abort(404, description="未找到对应聊天记录")

    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        abort(400, description="query 不能为空")

    result = pipeline.ask(query)

    if chat["title"] == "新对话":
        chat["title"] = query[:20] + ("..." if len(query) > 20 else "")

    chat["messages"].append({
        "role": "user",
        "content": query,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    })
    chat["messages"].append({
        "role": "assistant",
        "content": result.get("answer", ""),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "payload": result,
    })
    save_chat_payload(chat)
    return jsonify(result)


@app.post("/api/chats/<chat_id>/clear")
def api_clear_chat(chat_id):
    try:
        chat = load_chat(chat_id)
    except FileNotFoundError:
        abort(404, description="未找到对应聊天记录")
    chat["messages"] = []
    chat["title"] = "新对话"
    save_chat_payload(chat)
    return jsonify(chat)


@app.post("/api/chats/<chat_id>/save")
def api_save_chat(chat_id):
    try:
        chat = load_chat(chat_id)
    except FileNotFoundError:
        abort(404, description="未找到对应聊天记录")
    save_chat_payload(chat)
    return jsonify({"ok": True, "path": str(chat_file(chat_id))})


@app.get("/api/images")
def api_image_proxy():
    image_path = request.args.get("path", "")
    if not image_path:
        abort(400, description="path 不能为空")

    path = Path(image_path).resolve()
    allowed_roots = [
        (BASE_DIR / "data").resolve(),
        BASE_DIR.resolve(),
    ]
    if not any(str(path).startswith(str(root)) for root in allowed_roots):
        abort(403, description="不允许访问该图片路径")
    if not path.exists() or not path.is_file():
        abort(404, description="图片不存在")
    return send_file(path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
