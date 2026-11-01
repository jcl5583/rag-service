"""黄金集回归测试。

用法（必须先在 rag_service 目录下）：
    python eval/run_eval.py                              # 默认跑 golden.jsonl
    python eval/run_eval.py eval/golden_hard.jsonl       # 跑指定测试集
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.main import ChatRequest, chat  # noqa: E402

DEFAULT_GOLDEN = Path(__file__).resolve().parent / "golden.jsonl"

# 拒答时可能出现的说法
REFUSAL_WORDS = [
    "没有提到",
    "未提及",
    "没有规定",
    "未规定",
    "资料中没有",
    "无法确定",
    "没有明确",
]


def normalize(text: str) -> str:
    """去掉空格，避免「2023 年 8 月 15 日」和「2023年8月15日」对不上"""
    return "".join(text.split())


def check(item: dict, answer: str) -> tuple[bool, str]:
    if item["type"] == "answerable":
        flat = normalize(answer)
        for keyword in item["expect"]:
            if normalize(keyword) in flat:
                return True, f"命中关键词「{keyword}」"
        return False, "未命中任何期望关键词"
    else:
        for word in REFUSAL_WORDS:
            if word in answer:
                return True, f"正确拒答（{word}）"
        return False, "应该拒答但答了内容"


def main() -> None:
    if len(sys.argv) > 1:
        golden = Path(sys.argv[1])
    else:
        golden = DEFAULT_GOLDEN

    if not golden.exists():
        raise SystemExit(f"测试集不存在：{golden}")

    items = [
        json.loads(line)
        for line in golden.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    print(f"测试集：{golden.name}，共 {len(items)} 条\n")

    passed = 0
    context_sizes = []
    failures = []

    for i, item in enumerate(items, 1):
        started = time.time()
        resp = chat(ChatRequest(question=item["q"]))
        cost = time.time() - started

        answer = resp["answer"]
        if "context_chars" in resp:
            context_sizes.append(resp["context_chars"])

        ok, reason = check(item, answer)
        passed += 1 if ok else 0

        flag = "OK  " if ok else "FAIL"
        print(f"[{flag}] {i:2d}. {item['q']}  ({cost:.1f}s)")
        if not ok:
            failures.append((i, item, answer, reason))
            print(f"        原因：{reason}")
            print(f"        实际：{answer[:150]}")

    total = len(items)
    print()
    print("=" * 60)
    print(f"通过率：{passed}/{total}  ({passed / total:.0%})")
    if context_sizes:
        print(f"平均上下文：{sum(context_sizes) / len(context_sizes):.0f} 字")
    print("=" * 60)

    if failures:
        print(f"\n失败的 {len(failures)} 题就是下一步要优化的目标。")


if __name__ == "__main__":
    main()
