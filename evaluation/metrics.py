from dataclasses import dataclass
from typing import Literal

from ..rag.retrieval import Candidate, RetrievalState

RECALL_STAGES = (
    ("context_recall_knn", "knn_candidates", "keyword"),
    ("context_recall_bm25", "bm25_candidates", "vector"),
    ("context_recall_rrf", "rrf_merged", None),
    ("context_recall_topk", "reranked_topk", None),
)
JUDGE_METRICS = ("context_relevancy", "faithfulness", "answer_relevancy")
RETRIEVAL_METRICS = tuple(name for name, _, _ in RECALL_STAGES) + ("context_precision",)
METRIC_NAMES = RETRIEVAL_METRICS + JUDGE_METRICS


@dataclass(frozen=True)
class Rating:
    status: Literal["ok", "skipped", "error"]
    reason: str
    score: float | None = None


def calculate_context_metrics(state: RetrievalState, k: int, truth: set[str]) -> dict[str, Rating]:
    if not truth:
        return {
            name: Rating("skipped", "缺少标准切片标签")
            for name in RETRIEVAL_METRICS
        }

    ratings: dict[str, Rating] = {}
    for name, stage, disabled_mode in RECALL_STAGES:
        if state["mode"] == disabled_mode:
            ratings[name] = Rating("skipped", "未启用该召回分支")
            continue
        matched = truth.intersection(item.id for item in state[stage])
        ratings[name] = Rating("ok", f"命中 {len(matched)} / {len(truth)} 个标注切片", len(matched) / len(truth))

    matched: set[str] = set()
    precision_sum = 0.0
    for rank, item in enumerate(state["reranked_topk"][:k], 1):
        label = item.id
        if label is not None and label in truth and label not in matched:
            matched.add(label)
            precision_sum += len(matched) / rank
    denominator = min(k, len(truth))
    ratings["context_precision"] = Rating(
        "ok", f"Top-{k} 命中 {len(matched)} 个标注切片，归一化分母 {denominator}",
        precision_sum / denominator,
    )
    return ratings


@dataclass(frozen=True)
class RetrievalScores:
    status: Literal["ok", "missing_labels"]
    k: int
    level: Literal["chunk", "document"] | None
    hit_at_k: int | None
    mrr: float | None
    recall_at_k: float | None
    relevant_count: int
    matched_ids: list[str]


def calculate(
    hits: list[Candidate], k: int, document_ids: set[str], chunk_ids: set[str],
) -> RetrievalScores:
    if not chunk_ids and not document_ids:
        return RetrievalScores("missing_labels", k, None, None, None, None, 0, [])

    ranked: dict[str, set[str]] = {}
    for candidate in hits[:k]:
        hit = candidate.hit
        if chunk_ids:
            key = f"{hit.document_id}#{hit.chunk_index}"
            related = {key}
            if hit.parent_index is not None:
                related.add(f"{hit.document_id}#{hit.parent_index}")
        else:
            key = hit.document_id
            related = {key}
        ranked.setdefault(key, related)

    truth = chunk_ids or document_ids
    matches: set[str] = set()
    first_rank = None
    for rank, related in enumerate(ranked.values(), 1):
        current = truth & related
        if current and first_rank is None:
            first_rank = rank
        matches.update(current)
    return RetrievalScores(
        "ok", k, "chunk" if chunk_ids else "document", int(bool(matches)),
        1 / first_rank if first_rank else 0.0, len(matches) / len(truth),
        len(truth), sorted(matches),
    )
