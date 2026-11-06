from typing import Annotated

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from ..document.identifiers import DocumentID
from ..rag.retrieval import Mode, Retrieval, RetrievalResult

router = APIRouter(prefix="/api", tags=["检索"])


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4096)]
    k: Annotated[int, Field(strict=True, ge=1, le=10)] = 5
    mode: Mode = "hybrid"
    document_id: DocumentID | None = None


@router.post("/retrieve")
def retrieve(request: Request, payload: SearchRequest) -> RetrievalResult:
    retrieval: Retrieval = request.app.state.retrieval
    return retrieval.search(payload.query, payload.k, payload.mode, payload.document_id)
