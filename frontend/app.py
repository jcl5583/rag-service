"""Streamlit 前端：输入问题 → 调后端流式接口 → 逐字显示答案 + 引用原文。

启动前必须先跑后端：
    uvicorn app.main:app --reload

然后：
    streamlit run frontend/app.py
"""
import os
import base64
import codecs
import json
import urllib.error
import urllib.request

import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/chat/stream")

st.set_page_config(page_title="政策文件问答", page_icon="📚", layout="centered")

st.title("📚 政策文件问答")
st.caption("基于《生成式人工智能服务管理暂行办法》与《人工智能生成合成内容标识办法》回答")

with st.sidebar:
    st.header("示例问题")
    st.markdown(
        "- 生成的内容需要打标识吗？\n"
        "- 发现违法内容应该怎么处理？\n"
        "- 违反规定会被罚多少钱？\n\n"
        "---\n"
        "**技术栈**\n\n"
        "FastAPI · BM25 + 向量混合检索 · bge-m3 · RRF 融合 · 流式输出"
    )


def stream_text(resp):
    """把响应体一小块一小块吐出来，并正确处理被切断的多字节字符"""
    decoder = codecs.getincrementaldecoder("utf-8")()
    read = getattr(resp, "read1", resp.read)
    while True:
        chunk = read(256)
        if not chunk:
            break
        text = decoder.decode(chunk)
        if text:
            yield text


question = st.text_input("输入你的问题", placeholder="例如：AI 生成的图片要打标识吗？")

if st.button("提问", type="primary", use_container_width=True):
    if not question.strip():
        st.warning("先输入一个问题")
        st.stop()

    payload = json.dumps({"question": question}).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )

    try:
        resp = urllib.request.urlopen(req, timeout=180)
    except urllib.error.URLError as exc:
        st.error(f"连不上后端（{exc}）。请先启动：uvicorn app.main:app --reload")
        st.stop()

    # 引用信息在响应头里
    raw_sources = resp.headers.get("X-Sources", "")
    sources = json.loads(base64.b64decode(raw_sources).decode("utf-8")) if raw_sources else []
    context_chars = resp.headers.get("X-Context-Chars", "")

    st.markdown("### 回答")
    st.write_stream(stream_text(resp))

    st.markdown(f"### 引用来源（{len(sources)} 段）")
    for i, item in enumerate(sources, 1):
        with st.expander(f"[{i}] {item.get('doc', '')}"):
            st.text(item.get("text", ""))

    if context_chars:
        st.caption(f"本次送入模型的上下文：{context_chars} 字")
