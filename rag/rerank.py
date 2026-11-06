from math import isfinite
import os
from threading import Lock
from time import monotonic, sleep

import httpx

from .errors import UpstreamFailure

BGE_MODEL_HASH = "912fc1215c2dbff6499700534bd8d31253af01573861abbfc43afd1fab6cce5d"
RERANK_TIMEOUT = 30.0


class Reranker:
    def __init__(self, client: httpx.Client) -> None:
        self.client = client
        self.url = os.getenv("RAG_RERANK_URL", "http://127.0.0.1:18091/rerank")
        self.slot = Lock()

    def score(self, query: str, texts: list[str]) -> list[float]:
        try:
            response = self.request_scores(query, texts)
        except httpx.TimeoutException as exc:
            raise UpstreamFailure("rerank", "BGE 重排请求超时", 504) from exc
        except httpx.HTTPError as exc:
            raise UpstreamFailure("rerank", "BGE 重排请求失败") from exc

        try:
            payload = response.json()
            scores = payload["scores"]
            if (
                payload["model_hash"] != BGE_MODEL_HASH or not isinstance(scores, list)
                or len(scores) != len(texts)
                or not all(isfinite(score) and 0 <= score <= 1 for score in scores)
            ):
                raise ValueError("Unexpected model or scores")
            return scores
        except (KeyError, TypeError, ValueError) as exc:
            raise UpstreamFailure("rerank", "BGE 返回的分数或模型指纹不匹配") from exc

    def request_scores(self, query: str, texts: list[str]) -> httpx.Response:
        deadline = monotonic() + RERANK_TIMEOUT
        if not self.slot.acquire(timeout=RERANK_TIMEOUT):
            raise UpstreamFailure("rerank", "等待 BGE 重排超时", 504)
        try:
            while (remaining := deadline - monotonic()) > 0:
                response = self.client.post(
                    self.url, json={"query": query, "texts": texts}, timeout=remaining,
                )
                # Other applications can share the one-slot BGE service.
                if response.status_code == 503 and response.text.strip() == "reranker busy":
                    sleep(min(0.2, max(0, deadline - monotonic())))
                    continue
                response.raise_for_status()
                return response
            raise UpstreamFailure("rerank", "等待 BGE 重排超时", 504)
        finally:
            self.slot.release()
