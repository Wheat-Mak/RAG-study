from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints, ValidationError

from ..ai.chat import ChatModel
from .errors import UpstreamFailure
from .retrieval import Evidence, generation_context_slices

UNKNOWN_ANSWER = "现有资料不足以回答这个问题。"
ANSWER_PROMPT = (
    Path(__file__).resolve().parents[1] / "prompts" / "rag-answer-system.txt"
).read_text(encoding="utf-8").strip()


class ModelAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    answer: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8192)]
    citation_ids: list[str]
    sufficient: bool


@dataclass(frozen=True)
class AnswerContext:
    id: str
    filename: str
    text: str
    original_url: str
    truncated: bool


@dataclass
class AnswerResult:
    status: Literal["answered", "insufficient_evidence", "error"]
    answer: str = ""
    citation_ids: list[str] = field(default_factory=list)
    contexts: list[AnswerContext] = field(default_factory=list)
    requested_model: str | None = None
    response_model: str | None = None
    error: str | None = None


def context_payload(contexts: list[AnswerContext]) -> list[dict[str, object]]:
    return [
        {"id": item.id, "filename": item.filename, "text": item.text, "truncated": item.truncated}
        for item in contexts
    ]


def answer_question(model: ChatModel, query: str, evidence: list[Evidence]) -> AnswerResult:
    contexts = []
    for position, (item, end_char) in enumerate(generation_context_slices(evidence), 1):
        contexts.append(AnswerContext(
            f"C{position}", item.filename, item.text[:end_char],
            item.original_url, end_char < len(item.text),
        ))
    if not contexts:
        return AnswerResult("insufficient_evidence", UNKNOWN_ANSWER)

    try:
        completion = model.complete("generation", ANSWER_PROMPT, {
            "question": query, "contexts": context_payload(contexts),
        })
        result = ModelAnswer.model_validate(completion.value)
        if not result.sufficient:
            # Refusals are rendered from a fixed response and carry no model citations.
            return AnswerResult(
                "insufficient_evidence", UNKNOWN_ANSWER, contexts=contexts,
                requested_model=model.model, response_model=completion.model,
            )
        ids = set(result.citation_ids)
        markers = set(re.findall(r"\[(C[0-9]+)\]", result.answer))
        available_ids = {item.id for item in contexts}
        if not ids:
            raise UpstreamFailure("generation", "模型声明证据充足，但未提供引用编号")
        if not (ids | markers) <= available_ids:
            raise UpstreamFailure("generation", "模型引用了本次未提供的证据编号")
        if len(ids) != len(result.citation_ids):
            raise UpstreamFailure("generation", "模型引用列表包含重复编号")
        if ids != markers:
            raise UpstreamFailure("generation", "答案正文与引用列表的编号不一致")
    except (UpstreamFailure, ValidationError) as exc:
        reason = str(exc) if isinstance(exc, UpstreamFailure) else "模型答案不符合 JSON 结构"
        return AnswerResult("error", contexts=contexts, requested_model=model.model, error=reason)

    return AnswerResult("answered", result.answer, result.citation_ids, contexts, model.model, completion.model)
