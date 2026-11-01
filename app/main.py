import os

from dotenv import load_dotenv
from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel
from app.splitter import load_documents, split_by_paragraph
from app.retriever import Retriever
from app.embedder import VectorRetriever
from app.hybrid_retriever import HybridRetriever
TOP_K = int(os.getenv("TOP_K", "5"))

load_dotenv()

app = FastAPI(title="RAG Service")

client = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
)

# KNOWLEDGE = """
# 成都东软学院图书馆借阅规则（2026 年修订）
#
# 1. 开放时间：周一至周五 8:00-22:00，周六周日 9:00-21:00。
# 2. 借阅数量：本科生一次最多借 5 册，教师最多借 20 册。
# 3. 借阅期限：普通图书 30 天，可续借 1 次，续借期 15 天。
# 4. 逾期处理：每册每天 0.2 元，累计超过 20 元暂停借阅权限。
# 5. 考试周期间，所有在借图书到期日顺延 7 天。成都东软学院图书馆借阅规则（2026 年修订）
#
# """
# 启动时加载所有文档并切分，拼成一份上下文
CHUNKS = []
for doc in load_documents():
    for chunk in split_by_paragraph(doc["text"]):
        CHUNKS.append({"source": doc["source"], "text": chunk})

ALL_CONTEXT = "\n\n---\n\n".join(c["text"] for c in CHUNKS)
RETRIEVER_KIND = os.getenv("RETRIEVER", "bm25")
if RETRIEVER_KIND == "vector":
    retriever = VectorRetriever(CHUNKS)
elif RETRIEVER_KIND == "hybrid":
    retriever = HybridRetriever(CHUNKS)
else:
    retriever = Retriever(CHUNKS)
print(f"检索方式：{RETRIEVER_KIND}")

print(f"已加载 {len(CHUNKS)} 个文本块，总长度 {len(ALL_CONTEXT)} 字")

class ChatRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


# @app.post("/chat")
# def chat(req: ChatRequest):
#     prompt = f"""只根据下面的【资料】回答用户问题。
# 资料里没有提到的内容，直接回答"资料中没有提到"，不要猜测。
#
# 【资料】
# {ALL_CONTEXT}
#
# 【问题】
# {req.question}"""
#
#     resp = client.chat.completions.create(
#         model=os.getenv("LLM_MODEL"),
#         messages=[
#             {"role": "user", "content": prompt},
#         ],
#         temperature=0.1,
#     )
#     return {
#         "answer": resp.choices[0].message.content,
#         "sources": [],
#     }
@app.post("/chat")
def chat(req: ChatRequest):
    hits = retriever.search(req.question, top_k=TOP_K)
    context = "\n\n---\n\n".join(
        f"[{i + 1}] {h['source']}\n{h['text']}" for i, h in enumerate(hits)
    )
    print(f"检索到 {len(hits)} 块，上下文 {len(context)} 字")

    prompt = f"""只根据下面的【资料】回答问题。
资料里没有提到的内容，直接回答"资料中没有提到"，不要猜测。

【资料】
{context}

【问题】
{req.question}"""

    resp = client.chat.completions.create(
        model=os.getenv("LLM_MODEL"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return {
        "answer": resp.choices[0].message.content,
        "sources": [{"doc": h["source"], "text": h["text"]} for h in hits],
        "context_chars": len(context),  # ← 加这行，评测时能算出平均上下文长度
    }