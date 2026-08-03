"""统一检索模块：向量 + BM25 混合检索（RRF 融合）。

向量通道：bge 语义相似度，理解"意思"。
关键词通道：BM25 字面匹配，不丢"字面"。
两者结果按排名做倒数加权融合（Reciprocal Rank Fusion），
取综合排名前 top_k 返回。

接口：HybridRetriever(query, top_k) -> (documents, metadatas)
与 Chroma 的 collection.query 返回结构兼容，可直接替换。
"""

from __future__ import annotations

from pathlib import Path

import chromadb
import jieba
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
VECTOR_DB_DIR = BASE_DIR / "data" / "vector_db"
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
RRF_CONSTANT = 60  # RRF 标准常数


def _tokenize(text: str) -> list[str]:
    """中文分词：过滤空白与纯符号 token。"""
    return [t.strip() for t in jieba.cut(text) if t.strip()]


class HybridRetriever:
    def __init__(self, top_k: int = 8) -> None:
        self.top_k = top_k
        self.embedder = SentenceTransformer(MODEL_NAME, local_files_only=True)
        self.client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
        self.collection = self.client.get_collection("env_law")
        # 一次性拉取全部条目，构建 BM25 索引（774 条，构建很快）
        self._load_all()

    def _load_all(self) -> None:
        ids: list[str] = []
        all_docs: list[str] = []
        all_metas: list[dict] = []
        offset = 0
        while True:
            r = self.collection.get(
                limit=100, offset=offset, include=["documents", "metadatas"]
            )
            if not r["ids"]:
                break
            ids.extend(r["ids"])
            all_docs.extend(r["documents"])
            all_metas.extend(r["metadatas"])
            offset += len(r["ids"])
        self.all_ids = ids
        self.all_docs = all_docs
        self.all_metas = all_metas
        self.id2index = {doc_id: i for i, doc_id in enumerate(ids)}
        self.bm25 = BM25Okapi([_tokenize(d) for d in all_docs])

    def _vector_search(self, query: str, k: int) -> dict:
        qv = self.embedder.encode(
            [BGE_QUERY_PREFIX + query], normalize_embeddings=True
        )
        return self.collection.query(
            query_embeddings=qv.tolist(), n_results=k, include=["documents", "metadatas"]
        )

    def _bm25_search(self, query: str, k: int) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(_tokenize(query))
        order = scores.argsort()[::-1][:k]
        return [(self.all_ids[i], float(scores[i])) for i in order if scores[i] > 0]

    def retrieve(self, query: str, top_k: int | None = None) -> tuple[list[str], list[dict]]:
        k = top_k or self.top_k
        vec_hits = self._vector_search(query, k)

        # RRF 融合（Chroma query 返回嵌套列表，取 [0]）
        rrf_scores: dict[str, float] = {}
        for rank, doc_id in enumerate(vec_hits["ids"][0], start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1 / (RRF_CONSTANT + rank)
        bm25_ranked = self._bm25_search(query, k)
        for rank, (doc_id, _score) in enumerate(bm25_ranked, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1 / (RRF_CONSTANT + rank)

        ranked_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:k]
        documents = [self.all_docs[self.id2index[d]] for d in ranked_ids]
        metadatas = [self.all_metas[self.id2index[d]] for d in ranked_ids]
        return documents, metadatas
