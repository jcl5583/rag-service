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
def split_by_paragraph(text: str, max_chars: int = 400) -> list[str]:
    """按空行分段，再把短段落合并到接近 max_chars"""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paras:
        if buf and len(buf) + len(p) + 2 > max_chars:
            chunks.append(buf)
            buf = p
        else:
            buf = f"{buf}\n\n{p}" if buf else p
    if buf:
        chunks.append(buf)
    return chunks

# if __name__ == "__main__":
#     docs = load_documents()
#     print(f"读到 {len(docs)} 个文档\n")
#
#     for doc in docs:
#         chunks = split_text(doc["text"])
#         print(f"--- {doc['source']}：{len(doc['text'])} 字 → {len(chunks)} 块")
#         for i, chunk in enumerate(chunks[:3]):
#             preview = chunk[:70].replace("\n", " ")
#             print(f"    [{i}] {preview}...")
#         print()
if __name__ == "__main__":
    docs = load_documents()
    print(f"读到 {len(docs)} 个文档\n")

    for doc in docs:
        text = doc["text"]
        chars = split_text(text)
        paras = split_by_paragraph(text)

        print(f"=== {doc['source']}（{len(text)} 字）")
        print(f"  按字符切：{len(chars):2d} 块，长度 {min(map(len, chars))} ~ {max(map(len, chars))}")
        print(f"  按段落切：{len(paras):2d} 块，长度 {min(map(len, paras))} ~ {max(map(len, paras))}")
        print("  按段落切的前 3 块：")
        for i, c in enumerate(paras[:3]):
            preview = c[:55].replace("\n", " ")
            print(f"    [{i}] {preview}...")
        print()