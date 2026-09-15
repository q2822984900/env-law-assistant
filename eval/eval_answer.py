"""回答质量评测：跑完整链路（检索 + DeepSeek 生成），量化引用幻觉与引用准确率。

用法：python eval/eval_answer.py
需要 .env 中已配置 DEEPSEEK_API_KEY，将调用 20 次 DeepSeek API。

指标：
- 越界引用（幻觉）：回答引用的 [n] 超出注入条数范围（编号不存在）
- 期望引用命中率：回答引用的条文是否属于该题的期望条文
- 问题级正确率：至少引用一条期望条文的问题占比
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# 让脚本无论从哪个工作目录运行，都能找到 src/ 下的检索模块
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retrieval import HybridRetriever

dotenv.load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
TOP_K = 16
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"

SYSTEM_PROMPT = "你是环境法规问答助手，精通中国环境法律法规。请严格按照要求作答。"


def build_llm() -> ChatOpenAI:
    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not key:
        print("未配置 DEEPSEEK_API_KEY，无法评测回答。请先填写 .env。")
        sys.exit(1)
    return ChatOpenAI(
        base_url=DEEPSEEK_BASE_URL,
        api_key=key,
        model=DEEPSEEK_MODEL,
        temperature=0.3,
        timeout=60,
    )


def cited_indexes(text: str) -> list[int]:
    return [int(x) for x in re.findall(r"\[(\d+)\]", text)]


def main() -> None:
    retriever = HybridRetriever(top_k=TOP_K)
    llm = build_llm()
    questions = json.load(open(BASE_DIR / "eval" / "questions.json", encoding="utf-8"))["questions"]

    total_refs = 0
    total_oob = 0
    total_expected_hit = 0
    q_expected_hit = 0
    q_with_ref = 0

    print(f"{'ID':<5}{'引用':<5}{'期望命中':<8}{'越界':<5} 问题")
    print("-" * 72)
    for q in questions:
        docs, metas = retriever.retrieve(q["question"])
        payloads: list[dict] = []
        cited: list[str] = []
        for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
            src = meta["source"].rsplit(".", 1)[0]
            num = meta.get("num") or ""
            payloads.append({"src": src, "num": num})
            cited.append(f"[{i}] 来源：{src} {num}：{doc}")
        cited_text = "\n\n".join(cited)

        user_prompt = f"""下面是检索到的相关条文（编号 [1]~[{len(cited)}]）：

{cited_text}

【用户问题】
{q["question"]}

要求：
1. 先直接回答用户问题：用清晰、通顺的语言给出结论与要点，不要只用条文原文堆砌
2. 每条结论后引用 1-2 条最相关的条文作佐证（写明"[编号]《法律名称》第X条"，如[1]《排污许可管理条例》第三十四条），不要把所有相关条文都罗列
3. 引用时摘录条文关键原句即可，不必整条照搬；内容必须忠实原文，不得编造
4. 若条文不足以回答用户问题，请明确说明"资料中未找到相关依据"，不要自行推断
5. 回答结尾请按 [编号] 列出本次引用的条文来源"""

        answer = llm.invoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_prompt)]
        ).content

        idxs = cited_indexes(answer)
        oob = [i for i in idxs if not (1 <= i <= len(payloads))]
        valid_idxs = [i for i in idxs if 1 <= i <= len(payloads)]

        expected = {(e["source"].rsplit(".", 1)[0], e["num"]) for e in q["expected_sources"]}
        expected_hit = sum(
            1 for i in valid_idxs if (payloads[i - 1]["src"], payloads[i - 1]["num"]) in expected
        )

        total_refs += len(idxs)
        total_oob += len(oob)
        total_expected_hit += expected_hit
        if idxs:
            q_with_ref += 1
        if expected_hit > 0:
            q_expected_hit += 1

        print(
            f"{q['id']:<5}{len(idxs):<5}{expected_hit}/{len(q['expected_sources']):<7}"
            f"{len(oob):<5} {q['question']}"
        )

    print()
    print("=" * 72)
    print(f"问题数：{len(questions)}")
    print(f"总引用次数：{total_refs}")
    print(f"越界引用（疑似幻觉）：{total_oob} 次（{total_oob / total_refs:.1%}）" if total_refs else "无引用")
    print(
        f"期望引用命中率：{total_expected_hit}/{total_refs}（{total_expected_hit / total_refs:.1%}）"
        if total_refs
        else ""
    )
    print(f"引用到期望条文的问题：{q_expected_hit}/{len(questions)}（{q_expected_hit / len(questions):.1%}）")


if __name__ == "__main__":
    main()
