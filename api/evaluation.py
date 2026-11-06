from typing import Annotated, Self

from fastapi import APIRouter, HTTPException, Request
from pydantic import Field, StringConstraints, model_validator

from ..document.identifiers import ChunkID, DocumentID
from ..evaluation.service import Evaluation, EvaluationCase, EvaluationResult, UnknownLabels
from .retrieval import SearchRequest

router = APIRouter(prefix="/api/eval", tags=["评测"])


class EvaluationRequest(SearchRequest):
    ground_truth_document_ids: list[DocumentID] = Field(default_factory=list, max_length=100)
    ground_truth_chunk_ids: list[ChunkID] = Field(default_factory=list, max_length=100)
    reference_answer: Annotated[str, StringConstraints(strip_whitespace=True, max_length=8000)] = ""
    generate: bool = False
    judge: bool = False

    @model_validator(mode="after")
    def check_judge(self) -> Self:
        if self.judge and not self.generate:
            raise ValueError("启用评分时需要生成答案")
        return self


@router.post("/evaluate")
def evaluate(request: Request, payload: EvaluationRequest) -> EvaluationResult:
    service: Evaluation = request.app.state.evaluation
    try:
        return service.run(EvaluationCase(**payload.model_dump()))
    except UnknownLabels as exc:
        raise HTTPException(422, {
            "stage": "labels", "message": str(exc), "missing": exc.labels,
        }) from exc
