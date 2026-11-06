from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from itertools import batched
import logging
from statistics import mean
from threading import Event, Thread
from typing import Any

from sqlalchemy.engine import Connection, RowMapping
from sqlalchemy.exc import SQLAlchemyError

from ..document.repository import DocumentBusy
from ..rag.errors import UpstreamFailure
from ..rag.rerank import BGE_MODEL_HASH
from ..rag.retrieval import CANDIDATE_LIMIT, RECALL_LIMIT, RRF_CONSTANT
from .dataset import BUILTIN_DATASET, Dataset, DatasetCase, DatasetError, parse_dataset
from .metrics import METRIC_NAMES, Rating
from .repository import EvaluationRepository
from .service import Evaluation, EvaluationCase, UnknownLabels

logger = logging.getLogger(__name__)
PARALLEL_CASES = 3
METRICS_VERSION = "context-and-answer-v1"


def summarize(rows: list[RowMapping]) -> dict[str, Any]:
    retrieval = defaultdict(list)
    judges = defaultdict(list)
    metrics = {name: [] for name in METRIC_NAMES}
    generation: Counter[str] = Counter()
    errors = missing = 0
    for row in rows:
        result = row.payload.get("result")
        for name, rating in (result or row.payload).get("metrics", {}).items():
            if name in metrics:
                metrics[name].append(rating)
        if result is None:
            errors += 1
            continue
        scores = result["retrieval_metrics"]
        if scores["status"] == "missing_labels":
            missing += 1
        else:
            for key in ("hit_at_k", "mrr", "recall_at_k"):
                retrieval[key].append(scores[key])
        answer = result["answer"]
        if answer is not None:
            generation[answer["status"]] += 1
        judgement = result["judgement"]
        if judgement is not None:
            for key, rating in judgement["metrics"].items():
                judges[key].append(rating)
    return {
        "retrieval_errors": errors, "missing_labels": missing,
        "retrieval": {
            key: {"mean": mean(retrieval[key]) if retrieval[key] else None, "count": len(retrieval[key])}
            for key in ("hit_at_k", "mrr", "recall_at_k")
        },
        "generation": dict(generation),
        "judge": {key: summarize_ratings(values) for key, values in judges.items()},
        "metrics": {key: summarize_ratings(values) for key, values in metrics.items()},
    }


def summarize_ratings(ratings: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [item["score"] for item in ratings if item["status"] == "ok"]
    counts = Counter(item["status"] for item in ratings)
    return {
        "mean": mean(scores) if scores else None, "count": len(scores),
        "skipped": counts["skipped"], "errors": counts["error"],
    }


class BatchEvaluation:
    def __init__(self, evaluation: Evaluation) -> None:
        self.evaluation = evaluation
        self.repository = EvaluationRepository(evaluation.retrieval.repository)
        self.stopping = Event()
        self.worker = Thread(target=self.work, name="rag-evaluation", daemon=True)

    def start(self) -> None:
        self.repository.initialize()
        dataset = parse_dataset(BUILTIN_DATASET.name, BUILTIN_DATASET.read_bytes())
        fingerprints = {key: value for case in dataset.cases for key, value in case.fingerprints.items()}
        self.repository.save_dataset(dataset, fingerprints, builtin=True)
        self.worker.start()

    def stop(self) -> None:
        self.stopping.set()
        self.worker.join()

    def register(self, dataset: Dataset) -> RowMapping:
        retrieval = self.evaluation.retrieval
        bindings = dataset.bind(retrieval.repository, retrieval.pipeline)
        stored = self.repository.get_dataset(dataset.id)
        if stored is not None and stored.bindings != bindings:
            raise DatasetError("相同题集的分块内容已变化，请核对标签后上传新版题集")
        return self.repository.save_dataset(dataset, bindings)

    def corpus(self) -> dict[str, str]:
        retrieval = self.evaluation.retrieval
        return {
            key: item.generation
            for key, item in retrieval.repository.active_documents(retrieval.pipeline, None).items()
        }

    def runtime(self, options: dict[str, Any]) -> dict[str, Any]:
        retrieval = self.evaluation.retrieval
        return {
            "pipeline": retrieval.pipeline, "vector_index": retrieval.index.name,
            "embedding_model": retrieval.embedder.model, "embedding_dimensions": retrieval.embedder.dimensions,
            "recall_limit": RECALL_LIMIT, "candidate_limit": CANDIDATE_LIMIT, "rrf_constant": RRF_CONSTANT,
            "rerank_min_score": retrieval.min_score, "reranker_model_hash": BGE_MODEL_HASH,
            "chat_model": self.evaluation.chat.model if options["generate"] else None,
            "metrics_version": METRICS_VERSION,
        }

    def enqueue(self, stored: RowMapping, options: dict[str, Any]) -> RowMapping:
        retrieval = self.evaluation.retrieval
        corpus = self.corpus()
        if not corpus:
            raise DatasetError("当前知识库没有已入库文档")
        dataset = parse_dataset(stored.filename, stored.content.encode())
        if dataset.bind(retrieval.repository, retrieval.pipeline) != stored.bindings:
            raise DatasetError("评测集绑定的分块已变化，请重新核对标签")
        if self.corpus() != corpus:
            raise DatasetError("语料代次正在切换，请重新发起评测")
        return self.repository.enqueue(dataset.id, len(dataset.cases), options, self.runtime(options), corpus)

    def work(self) -> None:
        while not self.stopping.is_set():
            try:
                with self.repository.documents.lock_document("evaluation-worker") as connection:
                    if not self.stopping.is_set():
                        run = self.repository.claim(connection)
                        if run is not None:
                            self.process(connection, run)
            except DocumentBusy:
                pass
            except Exception:
                logger.exception("Evaluation worker iteration failed; stored checkpoints will be reused")
            self.stopping.wait(1)

    def process(self, connection: Connection, run: RowMapping) -> None:
        def verify_corpus() -> None:
            if self.corpus() != run.corpus or self.runtime(run.options) != run.runtime:
                raise DatasetError("运行期间语料代次或模型配置发生变化，已停止本次评测")

        try:
            verify_corpus()
            stored = self.repository.get_dataset(run.dataset_id)
            if stored is None:
                raise DatasetError("评测集不存在")
            dataset = parse_dataset(stored.filename, stored.content.encode())
            saved = self.repository.saved_ordinals(connection, run.id)
            pending = ((i, case) for i, case in enumerate(dataset.cases) if i not in saved)
            with ThreadPoolExecutor(max_workers=PARALLEL_CASES) as executor:
                for group in batched(pending, PARALLEL_CASES):
                    if self.stopping.is_set():
                        return
                    verify_corpus()
                    futures = {executor.submit(self.evaluate_case, case, run.options): i for i, case in group}
                    for future in as_completed(futures):
                        result = future.result()
                        verify_corpus()
                        self.repository.checkpoint(connection, run.id, futures[future], result)
        except DatasetError as exc:
            self.repository.finish(connection, run.id, summarize(self.repository.results(run.id)), str(exc))
        except SQLAlchemyError:
            raise
        except Exception:
            logger.exception("Batch evaluation failed")
            connection.rollback()
            self.repository.finish(connection, run.id, summarize(self.repository.results(run.id)), "批量评测运行失败")
        else:
            self.repository.finish(connection, run.id, summarize(self.repository.results(run.id)), None)

    def evaluate_case(self, case: DatasetCase, options: dict[str, Any]) -> dict[str, Any]:
        request = EvaluationCase(
            query=case.question, k=options["k"], mode=options["mode"], document_id=None,
            ground_truth_document_ids=case.document_ids, ground_truth_chunk_ids=case.chunk_ids,
            reference_answer=case.reference_answer, generate=options["generate"], judge=options["judge"],
            expand_parents=options.get("expand_parents", True),
        )
        try:
            try:
                result = self.evaluation.run(request)
            except UpstreamFailure as exc:
                if exc.status != 504 or self.stopping.wait(0.25):
                    raise
                result = self.evaluation.run(request)
            payload = {"result": asdict(result), "error": None}
        except UpstreamFailure as exc:
            payload = {"result": None, "error": {"stage": exc.stage, "message": str(exc)}}
        except UnknownLabels as exc:
            payload = {"result": None, "error": {"stage": "labels", "message": str(exc)}}
        except Exception:
            logger.exception("Evaluation case failed: %s", case.id)
            payload = {"result": None, "error": {"stage": "evaluation", "message": "用例执行失败"}}
        if payload["error"] is not None:
            payload["metrics"] = {
                name: asdict(Rating("error", payload["error"]["message"])) for name in METRIC_NAMES
            }
        return {
            "case_id": case.id, "question": case.question, "reference_answer": case.reference_answer,
            "payload": {**payload, "ground_truth_document_ids": case.document_ids, "ground_truth_chunk_ids": case.chunk_ids},
        }
