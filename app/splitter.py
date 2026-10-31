from pathlib import Path

# 以脚本自身位置为基准，不管从哪里运行都找得到
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"


def load_documents() -> list[dict]:
    print(f"当前工作目录: {Path.cwd()}")
    print(f"文档目录:     {DATA_DIR.resolve()}")
    if not DATA_DIR.exists():
        raise SystemExit(f"目录不存在：{DATA_DIR.resolve()}")
    """读 data/docs 下所有 txt/md 文件"""
    docs = []
    for path in sorted(DATA_DIR.glob("*")):
        if path.suffix.lower() not in {".txt", ".md"}:
            continue
        text = path.read_text(encoding="utf-8")
        docs.append({"source": path.name, "text": text})
    return docs


def split_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """按固定长度切，相邻块重叠 overlap 个字符"""
    if chunk_size <= overlap:
        raise ValueError("chunk_size 必须大于 overlap")
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        if start + chunk_size >= len(text):
            break
        start += chunk_size - overlap
    return chunks


if __name__ == "__main__":
    docs = load_documents()
    print(f"读到 {len(docs)} 个文档\n")

    for doc in docs:
        chunks = split_text(doc["text"])
        print(f"--- {doc['source']}：{len(doc['text'])} 字 → {len(chunks)} 块")
        for i, chunk in enumerate(chunks[:3]):
            preview = chunk[:70].replace("\n", " ")
            print(f"    [{i}] {preview}...")
        print()