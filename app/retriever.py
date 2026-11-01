import math
from collections import Counter


def tokenize(text: str) -> list[str]:
    """把文本切成"相邻两个字"的小片段"""
    text = text.lower()
    tokens = []
    for i in range(len(text) - 1):
        gram = text[i:i + 2]
        if gram.strip():
            tokens.append(gram)
    return tokens


class Retriever:
    """BM25 关键词检索"""

    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b

        # 每个块的词频表
        self.bags = [Counter(tokenize(c["text"])) for c in chunks]
        # 每个块的长度（token 数）
        self.lengths = [sum(bag.values()) for bag in self.bags]
        self.avg_len = sum(self.lengths) / len(self.lengths)

        # 每个词出现在多少个块里（文档频率）
        self.df = Counter()
        for bag in self.bags:
            for term in bag:
                self.df[term] += 1
        self.N = len(chunks)

    def idf(self, term: str) -> float:
        """词越少见，分越高；出现在所有块里的词，分接近 0"""
        df = self.df.get(term, 0)
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        terms = set(tokenize(query))
        scored = []

        for i, bag in enumerate(self.bags):
            score = 0.0
            for term in terms:
                f = bag.get(term, 0)
                if f == 0:
                    continue
                # 长度归一化：块越长，这一项越小
                norm = 1 - self.b + self.b * (self.lengths[i] / self.avg_len)
                # TF 饱和 + IDF
                score += self.idf(term) * (f * (self.k1 + 1)) / (f + self.k1 * norm)
            scored.append((score, i))

        scored.sort(reverse=True)
        return [self.chunks[i] for s, i in scored[:top_k] if s > 0]