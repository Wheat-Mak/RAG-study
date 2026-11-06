from pathlib import PurePosixPath
from typing import Annotated, Self
from urllib.parse import quote

from fastapi import APIRouter, File, HTTPException, Path, Query, Request, Response, UploadFile
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator
from sqlalchemy.engine import RowMapping

from ..evaluation.batch import BatchEvaluation
from ..evaluation.dataset import MAX_DATASET_BYTES, DatasetError, parse_dataset
from ..rag.retrieval import Mode

router = APIRouter(prefix="/api/eval", tags=["批量评测"])
DatasetID = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
RunID = Annotated[str, Path(pattern=r"^[0-9a-f]{32}$")]


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: DatasetID
    k: Annotated[int, Field(strict=True, ge=1, le=10)] = 5
    mode: Mode = "hybrid"
    generate: bool = False
    judge: bool = False
    expand_parents: bool = True

    @model_validator(mode="after")
    def check_judge(self) -> Self:
        if self.judge and not self.generate:
            raise ValueError("启用评分时需要生成答案")
        return self


def dataset_response(row: RowMapping) -> dict[str, object]:
    return {
        "id": row.id, "filename": row.filename, "case_count": row.case_count,
        "builtin": bool(row.builtin), "download_url": f"/api/eval/datasets/{row.id}/download",
    }


def run_response(row: RowMapping) -> dict[str, object]:
    result = dict(row)
    result["created_at"] = row.created_at.isoformat(timespec="milliseconds") + "Z"
    result["finished_at"] = row.finished_at.isoformat(timespec="milliseconds") + "Z" if row.finished_at else None
    return result


@router.get("/datasets")
def list_datasets(request: Request) -> list[dict[str, object]]:
    batch: BatchEvaluation = request.app.state.batch_evaluation
    return [dataset_response(row) for row in batch.repository.list_datasets()]


@router.post("/datasets", status_code=201)
def upload_dataset(request: Request, file: Annotated[UploadFile, File()]) -> dict[str, object]:
    filename = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    if PurePosixPath(filename).suffix.lower() != ".jsonl":
        raise HTTPException(415, "请上传 JSONL 评测集")
    if len(filename) > 255 or any(ord(char) < 32 or ord(char) == 127 for char in filename):
        raise HTTPException(422, "评测集文件名无效")
    raw = file.file.read(MAX_DATASET_BYTES + 1)
    if len(raw) > MAX_DATASET_BYTES:
        raise HTTPException(413, "评测集超过 2 MiB")
    batch: BatchEvaluation = request.app.state.batch_evaluation
    try:
        dataset = parse_dataset(filename, raw)
        return dataset_response(batch.register(dataset))
    except DatasetError as exc:
        raise HTTPException(422, {"stage": "dataset", "message": str(exc)}) from exc


@router.get("/datasets/{dataset_id}/download", response_class=Response)
def download_dataset(request: Request, dataset_id: Annotated[str, Path(pattern=r"^[0-9a-f]{64}$")]) -> Response:
    batch: BatchEvaluation = request.app.state.batch_evaluation
    row = batch.repository.get_dataset(dataset_id)
    if row is None:
        raise HTTPException(404, "评测集不存在")
    filename = quote(row.filename, safe="")
    return Response(row.content, media_type="application/jsonl", headers={
        "Content-Disposition": f"attachment; filename*=UTF-8''{filename}",
    })


@router.post("/runs", status_code=202)
def start_run(request: Request, payload: RunRequest) -> dict[str, object]:
    batch: BatchEvaluation = request.app.state.batch_evaluation
    stored = batch.repository.get_dataset(payload.dataset_id)
    if stored is None:
        raise HTTPException(404, "评测集不存在")
    try:
        return run_response(batch.enqueue(stored, payload.model_dump(exclude={"dataset_id"})))
    except DatasetError as exc:
        raise HTTPException(409, {"stage": "dataset", "message": str(exc)}) from exc


@router.get("/runs")
def list_runs(
    request: Request, offset: Annotated[int, Query(ge=0, le=2**31 - 1)] = 0,
    limit: Annotated[int, Query(ge=1, le=50)] = 15,
) -> dict[str, object]:
    batch: BatchEvaluation = request.app.state.batch_evaluation
    total, rows = batch.repository.list_runs(offset, limit)
    return {"total": total, "offset": offset, "items": [run_response(row) for row in rows]}


@router.get("/runs/{run_id}")
def get_run(request: Request, run_id: RunID) -> dict[str, object]:
    batch: BatchEvaluation = request.app.state.batch_evaluation
    run = batch.repository.get_run(run_id)
    if run is None:
        raise HTTPException(404, "评测运行不存在")
    return run_response(run)


@router.get("/runs/{run_id}/cases")
def get_cases(
    request: Request, run_id: RunID, offset: Annotated[int, Query(ge=0, le=2**31 - 1)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 10,
) -> dict[str, object]:
    batch: BatchEvaluation = request.app.state.batch_evaluation
    run = batch.repository.get_run(run_id)
    if run is None:
        raise HTTPException(404, "评测运行不存在")
    return {
        "total": run.completed_cases, "offset": offset,
        "items": [{
            "ordinal": row.ordinal, "case_id": row.case_id, "question": row.question,
            "reference_answer": row.reference_answer, **row.payload,
        } for row in batch.repository.results(run_id, offset, limit)],
    }
