"""评测脚本：量化检索系统的召回率。

用法：python eval/run_eval.py [top_k]
默认 top_k=8（与 app.py 一致）。纯检索评测，不调用 API，可随时运行。

指标：
- 条文召回率@k：期望条文在检索前 k 名中命中的比例
- 问题全部命中率：期望条文全部命中的问题占比
- 问题至少命中率：期望条文至少命中一条的问题占比
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from retrieval import HybridRetriever

BASE_DIR = Path(__file__).resolve().parent.parent
TOP_K = int(sys.argv[1]) if len(sys.argv) > 1 else 8


def main() -> None:
    retriever = HybridRetriever(top_k=TOP_K)
    questions = json.load(open(BASE_DIR / "eval" / "questions.json", encoding="utf-8"))["questions"]

    total_expected = 0
    total_hit = 0
    q_all_hit = 0
    q_any_hit = 0

    print(f"{'ID':<5}{'结果':<6} 问题")
    print("-" * 62)
    for q in questions:
        _docs, metadatas = retriever.retrieve(q["question"])
        found = {(m["source"], m.get("num", "")) for m in metadatas}
        expected = {(e["source"], e["num"]) for e in q["expected_sources"]}
        hit_set = expected & found

        total_expected += len(expected)
        total_hit += len(hit_set)
        if hit_set == expected:
            q_all_hit += 1
        if hit_set:
            q_any_hit += 1

        status = "全部命中" if hit_set == expected else f"{len(hit_set)}/{len(expected)}"
        print(f"{q['id']:<5}{status:<6} {q['question']}")
        for e in q["expected_sources"]:
            src, num = e["source"], e["num"]
            mark = "  [命中]" if (src, num) in hit_set else "  [漏检]"
            name = src.rsplit(".", 1)[0].replace("中华人民共和国", "")
            print(f"{mark}  {name} · {num}")

    print()
    print("=" * 62)
    print(f"期望条文总数：{total_expected}")
    print(f"召回命中数：  {total_hit}")
    print(f"条文召回率@{TOP_K}：{total_hit / total_expected:.1%}")
    print(f"问题全部命中：{q_all_hit}/{len(questions)}（{q_all_hit / len(questions):.1%}）")
    print(f"问题至少命中：{q_any_hit}/{len(questions)}（{q_any_hit / len(questions):.1%}）")


if __name__ == "__main__":
    main()
