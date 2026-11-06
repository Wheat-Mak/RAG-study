import os
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from itertools import batched
from math import ceil
from time import monotonic
from typing import Literal, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from ..document.repository import DocumentRepository, PublishedDocument
from .chunking import TABLE_LAYOUT, SourceValue, table_rows_for_rerank
from .embedding import Embedder
from .errors import UpstreamFailure
from .recall import SearchHit, recall
from .rerank import BGE_MODEL_HASH, Reranker
from .vector_index import VectorIndex

Mode = Literal["hybrid", "vector", "keyword"]
RECALL_LIMIT = 20
RRF_CONSTANT = 60
CANDIDATE_LIMIT = 10
RECALL_TIMEOUT = 20.0
TABLE_RERANK_PASSAGE_LIMIT = 32


@dataclass
class Candidate:
    hit: SearchHit
    rrf_score: float = 0.0
    vector_rank: int | None = None
    vector_score: float | None = None
    keyword_rank: int | None = None
    keyword_score: float | None = None
    rerank_score: float | None = None


@dataclass
class Evidence:
    document_id: str
    generation: str
    filename: str
    chunk_index: int
    kind: str
    text: str
    source: dict[str, SourceValue]
    score: float
    original_url: str
    markdown_url: str | None
    matched_chunk_indices: list[int] = field(default_factory=list)


@dataclass
class RetrievalResult:
    query: str
    mode: Mode
    vector_count: int
    keyword_count: int
    rrf_constant: int
    rerank_min_score: float
    reranker_model_hash: str | None
    candidates: list[Candidate]
    hits: list[Candidate]
    evidence: list[Evidence]


class RetrievalState(TypedDict, total=False):
    query: str
    mode: Mode
    k: int
    document_id: str | None
    expand_parents: bool
    active: dict[str, PublishedDocument]
    vector: list[SearchHit]
    keyword: list[SearchHit]
    candidates: list[Candidate]
    hits: list[Candidate]
    knn_candidates: list[Document]
    bm25_candidates: list[Document]
    rrf_merged: list[Document]
    reranked_topk: list[Document]
    parent_contexts: list[Document]
    result: RetrievalResult


class ScopeChanged(RuntimeError):
    pass


def chunk_reference(item: SearchHit | Evidence, **metadata: str | int) -> Document:
    return Document(
        id=f"{item.document_id}#{item.chunk_index}",
        page_content="",
        metadata={
            "document_id": item.document_id,
            "generation": item.generation,
            "chunk_index": item.chunk_index,
            **metadata,
        },
    )


def generation_context_slices(evidence: list[Evidence]) -> Iterator[tuple[Evidence, int]]:
    remaining = 8000
    for item in evidence[:8]:
        if not remaining:
            break
        end_char = min(len(item.text), remaining, 2500)
        yield item, end_char
        remaining -= end_char


def fuse(vector: list[SearchHit], keyword: list[SearchHit]) -> list[Candidate]:
    candidates: dict[tuple[str, str, int], Candidate] = {}
    for route, hits in (("vector", vector), ("keyword", keyword)):
        seen: set[tuple[str, str, int]] = set()
        for rank, hit in enumerate(hits, 1):
            identity = hit.identity
            if identity in seen:
                continue
            seen.add(identity)
            candidate = candidates.setdefault(identity, Candidate(hit))
            candidate.rrf_score += 1 / (RRF_CONSTANT + rank)
            if route == "vector":
                candidate.vector_rank, candidate.vector_score = rank, hit.score
            else:
                candidate.keyword_rank, candidate.keyword_score = rank, hit.score
    return sorted(candidates.values(), key=lambda item: (-item.rrf_score, item.hit.identity))


class Retrieval:
    def __init__(
        self, repository: DocumentRepository, pipeline: str, embedder: Embedder,
        index: VectorIndex, reranker: Reranker,
    ) -> None:
        self.repository = repository
        self.pipeline = pipeline
        self.embedder = embedder
        self.index = index
        self.reranker = reranker
        self.min_score = float(os.getenv("RAG_RERANK_MIN_SCORE", "0.07585818"))
        if not 0 <= self.min_score <= 1:
            raise ValueError("Invalid RAG_RERANK_MIN_SCORE")

        graph = StateGraph(RetrievalState)
        graph.add_node("load_scope", self.load_scope)
        graph.add_node("vector_recall", self.vector_recall)
        graph.add_node("keyword_recall", self.keyword_recall)
        graph.add_node("fuse", self.fuse_routes)
        graph.add_node("rerank", self.rerank)
        graph.add_node("expand_evidence", self.expand_evidence)
        graph.add_edge(START, "load_scope")
        graph.add_edge("load_scope", "vector_recall")
        graph.add_edge("load_scope", "keyword_recall")
        graph.add_edge(["vector_recall", "keyword_recall"], "fuse")
        graph.add_edge("fuse", "rerank")
        graph.add_edge("rerank", "expand_evidence")
        graph.add_edge("expand_evidence", END)
        self.graph = graph.compile()

    def search(self, query: str, k: int, mode: Mode, document_id: str | None) -> RetrievalResult:
        return self.search_state(query, k, mode, document_id)["result"]

    def search_state(
        self, query: str, k: int, mode: Mode, document_id: str | None, *, expand_parents: bool = True,
    ) -> RetrievalState:
        state: RetrievalState = {
            "query": query, "k": k, "mode": mode, "document_id": document_id,
            "expand_parents": expand_parents,
        }
        for attempt in range(2):
            try:
                return self.graph.invoke(state)
            except ScopeChanged as exc:
                if attempt == 1:
                    raise UpstreamFailure("retrieval_scope", "文档代次正在切换，请重试", 409) from exc
        raise AssertionError("Retrieval attempt did not return a result")

    def load_scope(self, state: RetrievalState) -> RetrievalState:
        active = self.repository.active_documents(self.pipeline, state["document_id"])
        return {"active": active}

    def vector_recall(self, state: RetrievalState) -> RetrievalState:
        if not state["active"] or state["mode"] == "keyword":
            return {"vector": [], "knn_candidates": []}
        deadline = monotonic() + RECALL_TIMEOUT
        vector = self.embedder.embed([state["query"]], timeout=RECALL_TIMEOUT)[0]
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise UpstreamFailure("vector_recall", "向量召回超时", 504)
        hits = recall(
            self.index, state["active"], RECALL_LIMIT, route="vector", query=vector, timeout=remaining,
        )
        return {"vector": hits, "knn_candidates": [chunk_reference(hit) for hit in hits]}

    def keyword_recall(self, state: RetrievalState) -> RetrievalState:
        if not state["active"] or state["mode"] == "vector":
            return {"keyword": [], "bm25_candidates": []}
        hits = recall(
            self.index, state["active"], RECALL_LIMIT,
            route="keyword", query=state["query"], timeout=RECALL_TIMEOUT,
        )
        return {"keyword": hits, "bm25_candidates": [chunk_reference(hit) for hit in hits]}

    def fuse_routes(self, state: RetrievalState) -> RetrievalState:
        candidates = fuse(state["vector"], state["keyword"])[:CANDIDATE_LIMIT]
        return {
            "candidates": candidates,
            "rrf_merged": [chunk_reference(item.hit) for item in candidates],
        }

    def rerank(self, state: RetrievalState) -> RetrievalState:
        candidates = state["candidates"]
        if not candidates:
            return {"hits": [], "reranked_topk": []}
        scores = self.reranker.score(state["query"], [item.hit.text for item in candidates])
        if state["mode"] == "hybrid":
            self.rescore_table_candidates(state["query"], candidates[:state["k"]], scores)
        pairs = sorted(
            ((score, replace(item, rerank_score=score))
             for score, item in zip(scores, candidates, strict=True)),
            key=lambda pair: (-pair[0], -pair[1].rrf_score, pair[1].hit.identity),
        )
        hits = [item for score, item in pairs if score >= self.min_score][:state["k"]]
        return {
            "candidates": [item for _, item in pairs],
            "hits": hits,
            "reranked_topk": [chunk_reference(item.hit) for item in hits],
        }

    def rescore_table_candidates(
        self, query: str, candidates: list[Candidate], scores: list[float],
    ) -> None:
        tables: dict[int, list[str]] = {}
        for index, candidate in enumerate(candidates):
            hit = candidate.hit
            if (
                hit.kind != "table" or hit.source.get("oversized") or scores[index] >= self.min_score
                or candidate.vector_rank is None or candidate.keyword_rank is None
            ):
                continue
            rows = table_rows_for_rerank(
                hit.text, row_layout=hit.source.get("table_layout") == TABLE_LAYOUT,
            )
            if len(rows) > 1:
                tables[index] = rows
        if not tables:
            return

        # Reserve space for every eligible table and keep every row in the input.
        per_table = TABLE_RERANK_PASSAGE_LIMIT // len(tables)
        passages = [
            (index, "\n".join(group))
            for index, rows in tables.items()
            for group in batched(rows, ceil(len(rows) / per_table))
        ]
        row_scores = self.reranker.score(query, [text for _, text in passages])
        for (index, _), score in zip(passages, row_scores, strict=True):
            scores[index] = max(scores[index], score)

    def expand_evidence(self, state: RetrievalState) -> RetrievalState:
        selected = state["hits"]
        active = state["active"]
        parent_keys = {
            (item.hit.document_id, item.hit.parent_index) for item in selected
            if item.hit.parent_index is not None
        } if state["expand_parents"] else set()
        parents = self.repository.read_parents(self.pipeline, state["document_id"], active, parent_keys)
        if parents is None:
            raise ScopeChanged()
        if parents.keys() != parent_keys:
            raise UpstreamFailure("evidence", "命中的子块缺少对应的父块")

        evidence: dict[tuple[str, int], Evidence] = {}
        for item in selected:
            hit = item.hit
            index = (
                hit.parent_index if state["expand_parents"] and hit.parent_index is not None
                else hit.chunk_index
            )
            key = hit.document_id, index
            if key not in evidence:
                chunk = parents.get(key, hit)
                document = active[hit.document_id]
                url = f"/api/documents/{hit.document_id}"
                assert item.rerank_score is not None
                evidence[key] = Evidence(
                    hit.document_id, hit.generation, document.filename, index, chunk.kind,
                    chunk.text, chunk.source, item.rerank_score, f"{url}/original",
                    f"{url}/markdown" if document.format in {"md", "pdf", "docx"} else None,
                )
            evidence[key].matched_chunk_indices.append(hit.chunk_index)

        expanded = list(evidence.values())
        return {
            "result": RetrievalResult(
                state["query"], state["mode"], len(state["vector"]), len(state["keyword"]),
                RRF_CONSTANT, self.min_score, BGE_MODEL_HASH if state["candidates"] else None,
                state["candidates"], selected, expanded,
            ),
            "parent_contexts": [
                chunk_reference(item, context_id=f"C{position}", end_char=end_char)
                for position, (item, end_char) in enumerate(generation_context_slices(expanded), 1)
            ],
        }
