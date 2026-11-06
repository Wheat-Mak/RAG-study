from dataclasses import dataclass
from time import perf_counter

from ..ai.chat import ChatModel
from ..rag.answer import AnswerResult, answer_question
from ..rag.retrieval import Mode, Retrieval, RetrievalResult
from .judge import Judgement, judge_answer
from .metrics import JUDGE_METRICS, Rating, RetrievalScores, calculate, calculate_context_metrics


@dataclass(frozen=True)
class EvaluationCase:
    query: str
    k: int
    mode: Mode
    document_id: str | None
    ground_truth_document_ids: list[str]
    ground_truth_chunk_ids: list[str]
    reference_answer: str
    generate: bool
    judge: bool
    expand_parents: bool = True


@dataclass(frozen=True)
class EvaluationResult:
    retrieval_metrics: RetrievalScores
    retrieval: RetrievalResult
    answer: AnswerResult | None
    judgement: Judgement | None
    timings_ms: dict[str, int]
    metrics: dict[str, Rating]


class UnknownLabels(ValueError):
    def __init__(self, labels: set[str]) -> None:
        super().__init__("标准标签不在当前已入库的语料中")
        self.labels = sorted(labels)


class Evaluation:
    def __init__(self, retrieval: Retrieval, chat: ChatModel) -> None:
        self.retrieval = retrieval
        self.chat = chat

    def run(self, case: EvaluationCase) -> EvaluationResult:
        chunk_ids = set(case.ground_truth_chunk_ids)
        document_ids = set(case.ground_truth_document_ids) if not chunk_ids else set()
        if chunk_ids or document_ids:
            chunk_keys = {
                (doc_id, int(index)) for doc_id, index in (label.split("#") for label in chunk_ids)
            }
            found_docs, found_chunks = self.retrieval.repository.existing_labels(
                self.retrieval.pipeline, document_ids, chunk_keys,
            )
            missing = (document_ids - found_docs) | {
                f"{doc_id}#{index}" for doc_id, index in chunk_keys - found_chunks
            }
            if missing:
                raise UnknownLabels(missing)

        start = perf_counter()
        state = self.retrieval.search_state(
            case.query, case.k, case.mode, case.document_id, expand_parents=case.expand_parents,
        )
        result = state["result"]
        timings = {"retrieval": round((perf_counter() - start) * 1000)}
        retrieval_metrics = calculate(result.hits, case.k, document_ids, chunk_ids)
        metrics = calculate_context_metrics(state, case.k, chunk_ids)
        answer = judgement = None
        if case.generate:
            stage_start = perf_counter()
            answer = answer_question(self.chat, case.query, result.evidence)
            timings["generation"] = round((perf_counter() - stage_start) * 1000)
            if case.judge:
                stage_start = perf_counter()
                judgement = judge_answer(self.chat, self.retrieval.embedder, case.query, answer, result.evidence)
                timings["judge"] = round((perf_counter() - stage_start) * 1000)
        metrics.update(
            judgement.metrics if judgement is not None
            else {name: Rating("skipped", "未启用 LLM 评分") for name in JUDGE_METRICS}
        )
        timings["total"] = round((perf_counter() - start) * 1000)
        return EvaluationResult(retrieval_metrics, result, answer, judgement, timings, metrics)
