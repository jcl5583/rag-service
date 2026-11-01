"""Streamlit 前端：输入问题 → 调后端 /chat → 显示答案和引用原文。

启动前必须先跑后端：
    uvicorn app.main:app --reload

然后本目录下：
    streamlit run frontend/app.py
"""

import json
import urllib.error
import urllib.request

import streamlit as st

API_URL = "http://127.0.0.1:8000/chat"

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
        "FastAPI · BM25 + 向量混合检索 · bge-m3 · RRF 融合"
    )

question = st.text_input("输入你的问题", placeholder="例如：AI 生成的图片要打标识吗？")

if st.button("提问", type="primary", use_container_width=True):
    if not question.strip():
        st.warning("先输入一个问题")
    else:
        payload = json.dumps({"question": question}).encode("utf-8")
        req = urllib.request.Request(
            API_URL,
            data=payload,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with st.spinner("检索并生成中……"):
                with urllib.request.urlopen(req, timeout=180) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            st.error(f"连不上后端（{exc}）。请先启动：uvicorn app.main:app --reload")
        else:
            st.markdown("### 回答")
            st.markdown(data["answer"])

            sources = data.get("sources", [])
            st.markdown(f"### 引用来源（{len(sources)} 段）")
            for i, item in enumerate(sources, 1):
                if isinstance(item, dict):
                    title = f"[{i}] {item.get('doc', '')}"
                    body = item.get("text", "")
                else:
                    title = f"[{i}] {item}"
                    body = ""
                with st.expander(title):
                    st.text(body)

            context_chars = data.get("context_chars")
            if context_chars:
                st.caption(f"本次送入模型的上下文：{context_chars} 字")
