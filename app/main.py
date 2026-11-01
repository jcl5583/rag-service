"""FastAPI 服务：检索 + 生成。

接口：
    GET  /health       健康检查
    POST /chat         一次性返回完整答案（供评测脚本和调试使用）
    POST /chat/stream  流式返回答案，引用信息放在响应头 X-Sources
"""

import base64
import json
import logging
import os

from dotenv import load_dotenv

# 必须在读取任何环境变量之前执行
load_dotenv()

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from openai import OpenAI  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from app.embedder import VectorRetriever  # noqa: E402
from app.hybrid_retriever import HybridRetriever  # noqa: E402
from app.retriever import Retriever  # noqa: E402
from app.splitter import load_documents, split_by_paragraph  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("rag")

TOP_K = int(os.getenv("TOP_K", "5"))
RETRIEVER_KIND = os.getenv("RETRIEVER", "hybrid")

app = FastAPI(title="RAG Service")

client = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
)

# ---- 启动时加载并切分文档 ----
CHUNKS = []
for doc in load_documents():
    for chunk in split_by_paragraph(doc["text"]):
        CHUNKS.append({"source": doc["source"], "text": chunk})

if RETRIEVER_KIND == "vector":
    retriever = VectorRetriever(CHUNKS)
elif RETRIEVER_KIND == "bm25":
    retriever = Retriever(CHUNKS)
else:
    retriever = HybridRetriever(CHUNKS)

log.info("检索方式：%s，top_k=%s，共 %d 个文本块", RETRIEVER_KIND, TOP_K, len(CHUNKS))


# ---- 核心逻辑：检索 + 拼 prompt ----
def build_prompt(question: str) -> tuple[str, list[dict]]:
    hits = retriever.search(question, top_k=TOP_K)
    context = "\n\n---\n\n".join(
        f"[{i + 1}] {h['source']}\n{h['text']}" for i, h in enumerate(hits)
    )
    prompt = f"""只根据下面的【资料】回答问题。
资料里没有提到的内容，直接回答"资料中没有提到"，不要猜测。

【资料】
{context}

【问题】
{question}"""
    return prompt, hits


def answer(question: str) -> tuple[str, list[dict]]:
    prompt, hits = build_prompt(question)
    resp = client.chat.completions.create(
        model=os.getenv("LLM_MODEL"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return resp.choices[0].message.content, hits


def pack_sources(hits: list[dict]) -> list[dict]:
    return [{"doc": h["source"], "text": h["text"]} for h in hits]


class ChatRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    text, hits = answer(req.question)
    context_chars = sum(len(h["text"]) for h in hits)
    log.info("问答完成，上下文 %d 字", context_chars)
    return {
        "answer": text,
        "sources": pack_sources(hits),
        "context_chars": context_chars,
    }


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    prompt, hits = build_prompt(req.question)

    # 响应头只能放 ASCII，所以用 base64 编码
    sources_header = base64.b64encode(
        json.dumps(pack_sources(hits), ensure_ascii=False).encode("utf-8")
    ).decode("ascii")

    def generate():
        stream = client.chat.completions.create(
            model=os.getenv("LLM_MODEL"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield delta

    return StreamingResponse(
        generate(),
        media_type="text/plain; charset=utf-8",
        headers={
            "X-Sources": sources_header,
            "X-Context-Chars": str(sum(len(h["text"]) for h in hits)),
        },
    )
