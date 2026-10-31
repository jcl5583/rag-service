# RAG Service

一个基于 FastAPI 的文档问答服务：把资料交给大模型，让它只根据资料回答；资料里没有的，就说"没有"。

## 当前进度

| 能力 | 状态 |
|---|---|
| 调用大模型 API（DeepSeek，OpenAI 兼容协议） | ✅ 已完成 |
| FastAPI 服务，`/chat` + `/health` 接口 | ✅ 已完成 |
| 资料注入 prompt，约束模型只依据资料回答 | ✅ 已完成 |
| 文档上传 → 解析 → 切分 | 🚧 开发中 |
| 向量检索（embedding + 向量库） | 📋 计划中 |
| 评测（黄金集 + 命中率） | 📋 计划中 |
| Docker 部署 | 📋 计划中 |

> 当前版本把一段固定资料直接拼进 prompt，是 RAG 的最小雏形，**还不算真正的检索增强**——没有检索，只有注入。

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env      # Windows: copy .env.example .env
# 编辑 .env，填入你的 key
uvicorn app.main:app --reload