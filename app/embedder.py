"""向量检索：调 embedding 接口把文本变成向量，再用余弦相似度找最像的块。

注意：环境变量在函数内部读取，不在模块级读取。
原因是模块级代码在 import 那一刻就执行，而此时 .env 可能还没被 load_dotenv() 加载。
"""

import hashlib
import json
import math
import os
import urllib.request
from pathlib import Path

DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_MODEL = "BAAI/bge-m3"

CACHE_FILE = Path(__file__).resolve().parent.parent / "data" / "vectors.json"


def _config() -> tuple[str, str, str]:
    """每次调用时才读环境变量，避免 import 顺序问题"""
    base_url = os.getenv("EMBEDDING_BASE_URL") or DEFAULT_BASE_URL
    model = os.getenv("EMBEDDING_MODEL") or DEFAULT_MODEL
    api_key = os.getenv("EMBEDDING_API_KEY")
    if not api_key:
        raise SystemExit("缺少 EMBEDDING_API_KEY，请检查 .env 是否配置")
    return base_url, model, api_key


def _key(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def embed(texts: list[str]) -> list[list[float]]:
    """调一次 embeddings 接口，返回每个文本的向量"""
    base_url, model, api_key = _config()

    payload = json.dumps(
        {"model": model, "input": texts, "encoding_format": "float"}
    ).encode("utf-8")

    req = urllib.request.Request(
        f"{base_url}/embeddings",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    # 接口返回顺序不保证和输入一致，按 index 排回去
    items = sorted(data["data"], key=lambda x: x["index"])
    return [item["embedding"] for item in items]


def cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度：方向越接近，值越接近 1"""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def embed_chunks(chunks: list[dict]) -> list[list[float]]:
    """给所有块算向量；算过的缓存到 data/vectors.json，不重复花钱"""
    cache: dict[str, list[float]] = {}
    if CACHE_FILE.exists():
        cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))

    missing = [c["text"] for c in chunks if _key(c["text"]) not in cache]
    if missing:
        print(f"需要新算 {len(missing)} 个块的向量……")
        vectors = embed(missing)
        for text, vec in zip(missing, vectors):
            cache[_key(text)] = vec
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(json.dumps(cache), encoding="utf-8")

    return [cache[_key(c["text"])] for c in chunks]


class VectorRetriever:
    """接口和 BM25 的 Retriever 一致：都有 search(query, top_k)"""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.vectors = embed_chunks(chunks)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        query_vec = embed([query])[0]
        scored = []
        for i, vec in enumerate(self.vectors):
            scored.append((cosine(query_vec, vec), i))
        scored.sort(reverse=True)
        return [self.chunks[i] for score, i in scored[:top_k]]
