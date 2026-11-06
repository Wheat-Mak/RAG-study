from dataclasses import dataclass
import json
import os

import httpx

from ..rag.errors import UpstreamFailure


@dataclass(frozen=True)
class JsonCompletion:
    value: dict[str, object]
    model: str | None


class ChatModel:
    def __init__(self, client: httpx.Client) -> None:
        self.client = client
        self.url = os.getenv("RAG_CHAT_URL", "https://api.deepseek.com/chat/completions")
        self.model = os.getenv("RAG_CHAT_MODEL", "deepseek-v4-flash")
        self.key = os.getenv("RAG_CHAT_API_KEY", "").strip()

    def complete(
        self, stage: str, system: str, data: dict[str, object], *, temperature: float = 0.0,
    ) -> JsonCompletion:
        if not self.key:
            raise UpstreamFailure(stage, "模型服务尚未配置", 503)
        try:
            response = self.client.post(
                self.url, headers={"Authorization": f"Bearer {self.key}"}, timeout=80.0,
                json={
                    "model": self.model, "max_tokens": 2048, "temperature": temperature,
                    "thinking": {"type": "disabled"},
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": json.dumps(data, ensure_ascii=False)},
                    ],
                },
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise UpstreamFailure(stage, "模型请求超时", 504) from exc
        except httpx.HTTPStatusError as exc:
            raise UpstreamFailure(stage, f"模型返回 HTTP {exc.response.status_code}") from exc
        except httpx.RequestError as exc:
            raise UpstreamFailure(stage, "无法连接模型服务") from exc

        try:
            payload = response.json()
            choice = payload["choices"][0]
            if choice["finish_reason"] != "stop":
                raise ValueError("Model output did not complete")
            value = json.loads(choice["message"]["content"])
            if not isinstance(value, dict):
                raise ValueError("Expected JSON object")
            response_model = payload.get("model")
            return JsonCompletion(value, response_model if isinstance(response_model, str) else None)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise UpstreamFailure(stage, "模型未返回完整的 JSON 对象") from exc
