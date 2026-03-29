import os
import json
import re
import time
from openai import OpenAI
import httpx
from openai import APIStatusError, InternalServerError, NotFoundError
from langchain_core.documents import Document
from src.path import qwen3_8b_tune_model_name, qwen3_8b_base_model_path


LLM_CHAT_PROMPT = """
### 检索信息
{context}

### 任务
你是《Introduction to Information Retrieval》教材的问答助手。你必须综合考虑每条检索结果中的正文、页码、图表信息、脚注信息，再回答用户问题。

### 回答要求
1. 只能依据给定的检索信息作答，不允许编造。
2. 如果检索信息不足以支持回答，直接输出“无答案”。
3. 如果答案依赖图表或脚注，在最终答案中自然说明，并给出对应引用编号。
4. 不要输出思考过程、分析过程、解释过程。
5. 不要输出 <think>、</think> 或任何额外标签。
6. 必须严格按照以下格式输出，且只能输出这一段：
<answer>
这里写最终回答
</answer>

### 用户问题
{query}
"""

llm = OpenAI(
    api_key="EMPTY",
    base_url="http://127.0.0.1:8000/v1",
    http_client=httpx.Client(trust_env=False)
)


def request_chat(query, context, stream=False, max_retries=3):
    prompt = LLM_CHAT_PROMPT.format(context=context, query=query)
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            out = llm.chat.completions.create(
                model=qwen3_8b_tune_model_name,
                messages=[
                    {"role":"system","content":"你是一个人工智能助手。禁止输出思考过程，只输出最终答案。"},
                    {"role":"user","content":prompt}
                ],
                max_tokens=512,
                stream=stream,
                top_p=0.9,
                frequency_penalty=0.0,
                temperature=0.1,
            )
            print(f"[info] using model: {qwen3_8b_tune_model_name}")
            if not stream:
                return out.choices[0].message.content
            return out
        except NotFoundError as e:
            raise RuntimeError(
                "vLLM 当前没有暴露 LoRA 模型 'ir-rag-lora'。"
                "请确认服务已用 --enable-lora 和 --lora-modules ir-rag-lora=... 启动，"
                "并且 /v1/models 能看到该模型。"
            ) from e
        except APIStatusError as e:
            status_code = getattr(e, 'status_code', None)
            if status_code == 502:
                raise RuntimeError(
                    "vLLM 服务当前返回 502，LoRA 模型实例不健康。"
                    "请先重启 8000 端口上的 vLLM 服务，再重新执行 main.py。"
                ) from e
            last_error = e
            if attempt == max_retries:
                break
            time.sleep(1.5 * attempt)
        except InternalServerError as e:
            last_error = e
            if attempt == max_retries:
                break
            time.sleep(1.5 * attempt)

    raise last_error

if __name__ == "__main__":

    context = """
    【1】### 倒排索引
    倒排索引(inverted index)是信息检索系统中最重要的数据结构之一。它的核心思想不是按文档组织词项，而是按词项组织文档。对于每个词项，系统都维护一个列表，记录哪些文档包含该词项。

    【2】### 词项词典与倒排记录表
    倒排索引通常由两个主要部分组成:词项词典(dictionary)和倒排记录表(postings)。词项词典保存系统中出现过的全部词项，并且通常为每个词项保存一个指针；这个指针指向对应的倒排记录表。倒排记录表中保存包含该词项的文档编号集合。

    【3】### 倒排记录表中的内容
    在最简单的形式下,倒排记录表只需要存储文档编号(docID)。例如,如果词项“retrieval”出现在文档 2、4、7 中，那么它的倒排记录表就至少需要包含这几个文档编号。通过这种结构，系统可以快速找出哪些文档匹配查询词项。

    【4】### 位置信息的作用
    如果倒排记录表中不仅保存文档编号，还进一步保存词项在文档中的出现位置，那么系统就可以支持更复杂的查询。例如，短语查询要求多个词项以特定顺序紧邻出现，仅靠文档编号无法完成，而位置信息可以用来判断这种相邻关系。

    【5】### 为什么倒排索引高效
    倒排索引之所以高效，是因为查询时不需要逐篇扫描全部文档。对于查询中的每个词项，系统只需找到对应的倒排记录表，再对这些记录表进行合并或比较，就能得到满足条件的文档集合。这比顺序扫描整个文档集合要高效得多。
    """

    query = "倒排索引通常由哪两个主要部分组成？如果希望系统支持短语查询，还需要在倒排记录表中额外保存什么信息？"

    res = request_chat(query, context, stream=True)
    for r in res:
        print(r.choices[0].delta.content, end='')
    print()