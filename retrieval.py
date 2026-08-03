"""统一检索模块：混合检索（向量 + BM25 RRF 融合）+ rerank 精排。

流程：
1. 混合检索取候选池（默认 20 条）——向量看语义、BM25 看字面，RRF 按排名融合，保证"别漏"。
2. cross-encoder reranker 对每条候选与问题逐对打分——精排，保证"别杂"。
3. 取分数最高的 top_k 条注入大模型。

若无 reranker 模型（下载失败等），自动回退为纯混合检索，不报错。

接口：HybridRetriever(query, top_k) -> (documents, metadatas)
与 Chroma 的 collection.query 返回结构兼容，可直接替换。
"""

from __future__ import annotations

from pathlib import Path

import chromadb
import jieba
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
VECTOR_DB_DIR = BASE_DIR / "data" / "vector_db"
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
RERANKER_PATH = "D:/models/bge-reranker-base"
BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
RRF_CONSTANT = 60  # RRF 标准常数
RERANK_CANDIDATES = 20  # rerank 前的候选池大小


def _tokenize(text: str) -> list[str]:
    """中文分词：过滤空白与纯符号 token。"""
    return [t.strip() for t in jieba.cut(text) if t.strip()]


class HybridRetriever:
    def __init__(self, top_k: int = 8, use_rerank: bool = False) -> None:
        self.top_k = top_k
        self.embedder = SentenceTransformer(MODEL_NAME, local_files_only=True)
        self.client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
        self.collection = self.client.get_collection("env_law")
        self._load_all()
        self.reranker: CrossEncoder | None = None
        if use_rerank:
            self._load_reranker()

    def _load_reranker(self) -> None:
        """加载 rerank 精排模型；失败则回退纯混合检索。"""
        try:
            self.reranker = CrossEncoder(RERANKER_PATH, max_length=512)
            print("[检索] rerank 精排模型已加载")
        except Exception as exc:
            print(f"[警告] rerank 模型加载失败，回退纯混合检索：{exc}")
            self.reranker = None

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

    def _hybrid_top(self, query: str, k: int) -> tuple[list[str], list[dict]]:
        """向量 + BM25 RRF 融合，取前 k 条候选。"""
        vec_hits = self._vector_search(query, k)
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

    def retrieve(self, query: str, top_k: int | None = None) -> tuple[list[str], list[dict]]:
        """检索主入口：候选池精排后返回 top_k 条。"""
        k = top_k or self.top_k
        if self.reranker is None:
            return self._hybrid_top(query, k)
        # 1. 候选池（宁多勿缺）
        docs, metas = self._hybrid_top(query, RERANK_CANDIDATES)
        # 2. cross-encoder 逐对打分（宁缺毋滥）
        pairs = [[query, doc] for doc in docs]
        scores = self.reranker.predict(pairs)
        order = scores.argsort()[::-1][:k]
        return [docs[i] for i in order], [metas[i] for i in order]
