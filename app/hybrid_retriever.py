"""混合检索：BM25 + 向量，用 RRF（倒数排名融合）合并两路结果。

RRF 的思路：不看两路的分数（量纲不同，没法直接相加），只看排名。
每个块得分 = 1/(k + BM25排名) + 1/(k + 向量排名)
两路都靠前的块拿双份分，所以两边互补的块能排到最前面。
k 一般取 60，来自原始论文的经验值。
"""

from app.embedder import VectorRetriever
from app.retriever import Retriever


class HybridRetriever:
    """接口和另外两个一致：search(query, top_k)"""

    def __init__(self, chunks: list[dict], k: int = 60):
        self.chunks = chunks
        self.k = k
        self.bm25 = Retriever(chunks)
        self.vector = VectorRetriever(chunks)
        self.index_of = {id(c): i for i, c in enumerate(chunks)}

    def _ranks(self, hits: list[dict]) -> dict[int, int]:
        return {self.index_of[id(c)]: r for r, c in enumerate(hits)}

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        # 候选池放大，让两路各多给一些，融合后才有东西可挑
        pool = max(top_k * 4, 20)

        bm25_ranks = self._ranks(self.bm25.search(query, pool))
        vector_ranks = self._ranks(self.vector.search(query, pool))

        scores: dict[int, float] = {}
        for idx, rank in bm25_ranks.items():
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (self.k + rank + 1)
        for idx, rank in vector_ranks.items():
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (self.k + rank + 1)

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
        return [self.chunks[i] for i, _ in ranked]
