from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal

from ..document.repository import PublishedDocument
from .chunking import SourceValue
from .errors import UpstreamFailure
from .vector_index import VectorIndex


@dataclass(frozen=True)
class SearchHit:
    document_id: str
    generation: str
    chunk_index: int
    parent_index: int | None
    kind: str
    text: str
    source: dict[str, SourceValue]
    score: float

    @property
    def identity(self) -> tuple[str, str, int]:
        return self.document_id, self.generation, self.chunk_index


def recall(
    index: VectorIndex, active: dict[str, PublishedDocument], limit: int,
    *, route: Literal["vector", "keyword"], query: str | list[float], timeout: float,
) -> list[SearchHit]:
    scope = {"terms": {"generation": sorted(doc.generation for doc in active.values())}}
    body: dict[str, Any] = {
        "size": limit, "_source": {"excludes": ["vector"]},
    }
    if route == "vector":
        body["knn"] = {
            "field": "vector", "query_vector": query, "k": limit,
            "num_candidates": max(50, limit * 5), "filter": scope,
        }
    else:
        body["query"] = {"bool": {
            "filter": [scope], "must": [{"match": {"text": {"query": query, "operator": "or"}}}],
        }}
    stage = f"{route}_recall"
    try:
        result = index.request(
            "POST", "/_search", timeout=timeout, params={"allow_partial_search_results": "false"},
            json=body,
        )
    except UpstreamFailure as exc:
        raise UpstreamFailure(stage, str(exc), exc.status) from exc

    try:
        if result["timed_out"] or result["_shards"]["failed"]:
            raise ValueError("Partial search response")
        hits = []
        for item in result["hits"]["hits"]:
            record = SearchHit(score=item["_score"], **item["_source"])
            document = active.get(record.document_id)
            if (
                document is None or record.generation != document.generation
                or not isfinite(record.score) or record.score < 0
            ):
                raise ValueError("Unpublished generation or invalid score")
            hits.append(record)
        return hits
    except (KeyError, TypeError, ValueError) as exc:
        raise UpstreamFailure(stage, "Elasticsearch 返回的召回结果不完整或不符合当前代次") from exc
