import math
import os

import httpx

from .errors import UpstreamFailure


class Embedder:
    def __init__(self, client: httpx.Client) -> None:
        self.client = client
        self.url = os.getenv(
            "RAG_EMBEDDING_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
        )
        self.model = os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-v4")
        self.dimensions = int(os.getenv("RAG_EMBEDDING_DIMENSIONS", "1536"))
        self.batch_size = int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "10"))
        self.api_key = os.environ["RAG_EMBEDDING_API_KEY"].strip()
        if not self.api_key or not 1 <= self.batch_size <= 10 or self.dimensions < 1:
            raise ValueError("Embedding key, batch size or dimensions are invalid")

    def embed(self, texts: list[str], *, timeout: float = 80.0) -> list[list[float]]:
        try:
            response = self.client.post(
                self.url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model, "input": texts, "dimensions": self.dimensions,
                    "encoding_format": "float",
                },
                timeout=timeout,
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise UpstreamFailure("embedding", "向量化请求超时", 504) from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            raise UpstreamFailure("embedding", f"向量化请求失败（HTTP {status}）") from exc
        except httpx.RequestError as exc:
            raise UpstreamFailure("embedding", "无法连接向量化服务") from exc

        try:
            payload = response.json()
            if payload.get("model", self.model) != self.model or len(payload["data"]) != len(texts):
                raise ValueError("Embedding model or result count mismatch")
            vectors: dict[int, list[float]] = {}
            for item in payload["data"]:
                position = item["index"]
                vector = item["embedding"]
                if (
                    not isinstance(position, int) or position in vectors
                    or not 0 <= position < len(texts)
                    or not isinstance(vector, list) or len(vector) != self.dimensions
                    or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in vector)
                    or not any(vector)
                ):
                    raise ValueError("Invalid embedding index or vector")
                vectors[position] = vector
            return [vectors[position] for position in range(len(texts))]
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as exc:
            raise UpstreamFailure("embedding", "向量化返回的数据与模型配置不符") from exc
