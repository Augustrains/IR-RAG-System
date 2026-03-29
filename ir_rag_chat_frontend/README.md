# IR-RAG Chat Frontend

这个前端页面满足以下要求：

- 输入框输入问题
- 提交按钮发送问题
- 显示 answer
- 有引用文档时显示“引用文本如下”，并展示文档编号 + 文档文本
- 有图表信息时，根据本地 `image_path` 直接展示图片
- 页面风格接近 ChatGPT：左侧会话列表 + 右侧聊天区
- 支持清空当前聊天
- 支持保存当前聊天上下文
- 支持新建聊天窗口并点击历史聊天继续查看
- 聊天记录保存目录：`/root/autodl-tmp/IR-RAG-System/data/chat_sessions`

## 放置方式

建议把整个目录复制到：

```bash
/root/autodl-tmp/IR-RAG-System/web_chat
```

## 安装

```bash
cd /root/autodl-tmp/IR-RAG-System
pip install -r web_chat/requirements.txt
```

## 运行

```bash
cd /root/autodl-tmp/IR-RAG-System
python web_chat/app.py
```

然后浏览器访问：

```text
http://127.0.0.1:7860
```

## 说明

1. `app.py` 中保留了你原先的核心流程：
   - BM25 召回
   - Milvus 召回
   - `merge_docs`
   - `bge_m3_reranker.rank`
   - `request_chat`
   - `post_processing`

2. 前端只是把原来的 `while True + input()` 改成了 HTTP 接口调用，不改变问答主链路。

3. 图片展示走 `/api/images?path=...`，后端根据本地路径读取图片并返回。

4. 聊天记录每个窗口一个 JSON 文件，便于后续继续扩展“重命名聊天”“删除聊天”等功能。
