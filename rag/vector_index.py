import json
import os
import re
from typing import Any

import httpx

from .chunking import Chunk
from .errors import UpstreamFailure


class VectorIndex:
    def __init__(self, client: httpx.Client, model: str, dimensions: int) -> None:
        self.client = client
        self.url = os.getenv("RAG_ES_URL", "http://127.0.0.1:19200").rstrip("/")
        self.name = os.getenv("RAG_ES_INDEX", "rag-python-chunks")
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,126}", self.name):
            raise ValueError("Invalid Python Elasticsearch index name")
        self.model = model
        self.dimensions = dimensions
        key = os.getenv("RAG_ES_API_KEY")
        self.headers = {"Authorization": f"ApiKey {key}"} if key else {}

    def request(
        self, method: str, path: str, *, allow_missing: bool = False,
        content_type: str | None = None, timeout: float = 60.0, **kwargs: Any,
    ) -> Any:
        headers = self.headers if content_type is None else {
            **self.headers, "Content-Type": content_type,
        }
        try:
            response = self.client.request(
                method, f"{self.url}/{self.name}{path}", headers=headers, timeout=timeout,
                **kwargs,
            )
            if allow_missing and response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException as exc:
            raise UpstreamFailure("index", "Elasticsearch 请求超时", 504) from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            raise UpstreamFailure("index", f"Elasticsearch 请求失败（HTTP {status}）") from exc
        except (httpx.RequestError, ValueError) as exc:
            raise UpstreamFailure("index", "无法读取 Elasticsearch 响应") from exc

    def initialize(self) -> None:
        expected = {"schema": "python-chunks-v1", "embedding_model": self.model}
        mapping = self.request("GET", "/_mapping", allow_missing=True)
        if mapping is None:
            self.request("PUT", "", json={
                "settings": {"number_of_shards": 1, "number_of_replicas": 0},
                "mappings": {
                    "dynamic": "strict", "_meta": expected,
                    "properties": {
                        "document_id": {"type": "keyword"},
                        "generation": {"type": "keyword"},
                        "chunk_index": {"type": "integer"},
                        "parent_index": {"type": "integer"},
                        "kind": {"type": "keyword"},
                        "text": {"type": "text"},
                        "source": {"type": "object", "enabled": False},
                        "vector": {
                            "type": "dense_vector", "dims": self.dimensions,
                            "index": True, "similarity": "cosine",
                            "index_options": {"type": "hnsw"},
                        },
                    },
                },
            })
            mapping = self.request("GET", "/_mapping")
        actual = mapping.get(self.name, {}).get("mappings", {})
        vector = actual.get("properties", {}).get("vector", {})
        if actual.get("_meta") != expected or (
            vector.get("type"), vector.get("dims"), vector.get("index"), vector.get("similarity")
        ) != ("dense_vector", self.dimensions, True, "cosine"):
            raise UpstreamFailure("index", "Elasticsearch 索引与向量模型或维度不匹配")

    def write(
        self, document_id: str, generation: str, entries: list[tuple[int, Chunk]],
        vectors: list[list[float]],
    ) -> None:
        lines = []
        for (ordinal, chunk), vector in zip(entries, vectors, strict=True):
            lines.append(json.dumps({"index": {"_id": f"{generation}:{ordinal}"}}))
            lines.append(json.dumps({
                "document_id": document_id, "generation": generation, "chunk_index": ordinal,
                "kind": chunk.kind, "text": chunk.text, "source": chunk.source,
                "parent_index": chunk.parent_index, "vector": vector,
            }, ensure_ascii=False, allow_nan=False))
        payload = self.request(
            "POST", "/_bulk", params={"refresh": "wait_for"},
            content=("\n".join(lines) + "\n").encode(), content_type="application/x-ndjson",
        )
        items = payload.get("items", [])
        if payload.get("errors") or len(items) != len(entries) or any(
            item.get("index", {}).get("status") not in (200, 201) for item in items
        ):
            raise UpstreamFailure("index", "Elasticsearch 批量写入有分块失败")

    def count(self, document_id: str, generation: str) -> int:
        query = {"bool": {"filter": [
            {"term": {"document_id": document_id}}, {"term": {"generation": generation}},
        ]}}
        response = self.request("POST", "/_count", json={"query": query})
        if response.get("_shards", {}).get("failed") or not isinstance(response.get("count"), int):
            raise UpstreamFailure("index", "Elasticsearch 分块数量核对失败")
        return response["count"]

    def delete_generations(self, document_id: str, generation: str, *, keep: bool) -> None:
        query = {"bool": {
            "filter": [{"term": {"document_id": document_id}}],
        }}
        generation_clause = {"term": {"generation": generation}}
        if keep:
            query["bool"]["must_not"] = [generation_clause]
        else:
            query["bool"]["filter"].append(generation_clause)
        response = self.request(
            "POST", "/_delete_by_query", params={"refresh": "true"}, json={"query": query},
        )
        if response.get("timed_out") or response.get("failures") or response.get("version_conflicts"):
            raise UpstreamFailure("index", "Elasticsearch 旧向量清理未完成")
