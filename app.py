"""环境法规问答助手 · Streamlit 问答应用

用法：streamlit run app.py
流程：用户提问 → bge 向量化 → Chroma 检索 top-k 条文 → DeepSeek 生成 → 展示（含出处）
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from retrieval import HybridRetriever

dotenv.load_dotenv()

# ---------- 配置 ----------
BASE_DIR = Path(__file__).resolve().parent
VECTOR_DB_DIR = BASE_DIR / "data" / "vector_db"
TOP_K = 8
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"


@st.cache_resource(show_spinner="加载检索引擎（向量模型 + 关键词索引，首次约需十几秒）...")
def load_resources():
    """进程级缓存：混合检索器（向量 + BM25），只加载一次。"""
    if not VECTOR_DB_DIR.exists() or not any(VECTOR_DB_DIR.iterdir()):
        return None
    return HybridRetriever(top_k=TOP_K)


def build_llm() -> ChatOpenAI | None:
    """从 .env 读取 Key 构建 DeepSeek LLM；未配置返回 None。"""
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        return None
    return ChatOpenAI(
        base_url=DEEPSEEK_BASE_URL,
        api_key=api_key,
        model=DEEPSEEK_MODEL,
        temperature=0.3,
        timeout=60,
    )


def extract_cited_indexes(answer: str) -> list[int]:
    """从回答文本中解析引用的条文编号，如 [1]、[2]。"""
    return sorted({int(x) for x in re.findall(r"\[(\d+)\]", answer)})


def render_sources(items: list[dict]) -> None:
    """在展开器里展示引用条文：来源、条文号、条文原文。"""
    with st.expander("引用出处（含条文原文）"):
        for item in items:
            st.markdown(f"**{item['src']} · {item['num'] or '全文'}**")
            st.markdown(f"> {item['doc']}")


def main() -> None:
    st.set_page_config(page_title="环境法规问答助手", layout="centered")
    st.title("环境法规问答助手")
    st.caption(
        "基于本地法规库的检索问答，回答均附条文出处。"
        "当前数据：10 部环境法规。"
    )

    resources = load_resources()
    if resources is None:
        st.warning("尚未生成向量库。请在项目根目录先运行：`python ingest.py`")
        st.stop()
    retriever = resources

    llm = build_llm()
    if llm is None:
        st.error(
            "未检测到 DeepSeek API Key。请在项目根目录 `.env` 文件中填入 "
            "`DEEPSEEK_API_KEY=sk-...` 后刷新页面。"
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 展示历史
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                render_sources(msg["sources"])

    if prompt := st.chat_input("请输入问题，例如：建设项目开工前需要办理哪些环保手续？"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # 1. 检索
            with st.spinner("检索相关条文..."):
                docs, metas = retriever.retrieve(prompt)

            # 2. 构造提示词
            payloads: list[dict] = []
            cited: list[str] = []
            for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
                src = meta["source"].rsplit(".", 1)[0]  # 去掉 .docx/.pdf 后缀
                num = meta.get("num") or ""
                payloads.append({"src": src, "num": num, "doc": doc})
                cited.append(f"[{i}] 来源：{src} {num}：{doc}")

            cited_text = "\n\n".join(cited)
            system = "你是环境法规问答助手，精通中国环境法律法规。请严格按照要求作答。"
            user_prompt = f"""下面是检索到的相关条文（编号 [1]~[{len(cited)}]）：

{cited_text}

【用户问题】
{prompt}

要求：
1. 仅依据上述条文回答，忽略与本问题无关的条文，不要使用条文之外的知识
2. 若条文不足以回答用户问题，请明确说明"资料中未找到相关依据"
3. 正文中引用条文时，请写明"[编号]《法律名称》第X条"（如：[1]《排污许可管理条例》第三十四条），让用户知道编号对应的具体条文
4. 回答结尾请按 [编号] 列出本次引用的条文来源"""

            # 3. 生成
            if llm is None:
                answer = "（未配置 API Key，无法生成回答。请先在 `.env` 填写 DEEPSEEK_API_KEY。）"
            else:
                try:
                    with st.spinner("DeepSeek 生成中..."):
                        resp = llm.invoke(
                            [SystemMessage(content=system), HumanMessage(content=user_prompt)]
                        )
                    answer = resp.content
                except Exception as exc:
                    err = str(exc)
                    if "401" in err or "authentication" in err.lower() or "invalid" in err.lower():
                        st.error("API Key 无效或未授权，请检查 .env 中的 DEEPSEEK_API_KEY。")
                    elif "429" in err or "rate" in err.lower():
                        st.error("DeepSeek 限流（429）。请稍等片刻后重试。")
                    elif "timeout" in err.lower() or "timed out" in err.lower():
                        st.error("请求超时。可能网络波动或服务繁忙，请重试。")
                    else:
                        st.error(f"调用失败：{err}")
                    st.stop()

            st.markdown(answer)
            # 只显示回答实际引用到的条文，附原文
            used = extract_cited_indexes(answer)
            display = (
                [payloads[i - 1] for i in used if 1 <= i <= len(payloads)]
                or payloads
            )
            render_sources(display)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": display}
            )


if __name__ == "__main__":
    main()
