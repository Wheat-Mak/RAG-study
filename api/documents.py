from pathlib import PurePosixPath
from typing import Annotated, Literal
from urllib.parse import quote

from fastapi import APIRouter, File, HTTPException, Query, Request, Response, UploadFile
from sqlalchemy.engine import RowMapping
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ..document.parser import MAX_FILE_BYTES, MEDIA_TYPES
from ..document.repository import DocumentRepository
from ..document.service import DocumentIngestion
from ..evaluation.dataset import MAX_DATASET_BYTES

router = APIRouter(prefix="/api/documents", tags=["文档"])
UPLOAD_BODY_LIMITS = {
    "/api/documents": MAX_FILE_BYTES + 64 * 1024,
    "/api/eval/datasets": MAX_DATASET_BYTES + 64 * 1024,
}


class LimitUploadBody:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        limit = UPLOAD_BODY_LIMITS.get(scope.get("path", ""))
        if scope["type"] != "http" or scope["method"] != "POST" or limit is None:
            await self.app(scope, receive, send)
            return

        received = 0

        async def bounded_receive() -> Message:
            nonlocal received
            message = await receive()
            received += len(message.get("body", b""))
            if received > limit:
                raise HTTPException(413, {"stage": "upload", "message": "上传请求超过大小上限"})
            return message

        await self.app(scope, bounded_receive, send)


def require_document(storage: DocumentRepository, document_id: str) -> RowMapping:
    row = storage.find(document_id)
    if row is None:
        raise HTTPException(404, "文档不存在")
    return row


def document_response(row: RowMapping) -> dict[str, object]:
    url = f"/api/documents/{row.id}"
    result = dict(row)
    del result["pipeline"]
    result["status"] = "indexed" if row.generation else "pending_index"
    result.update(original_url=f"{url}/original", chunks_url=f"{url}/chunks")
    if row.format in {"md", "pdf", "docx"}:
        result["markdown_url"] = f"{url}/markdown"
    return result


@router.post("", status_code=201)
def upload(
    request: Request, response: Response, file: Annotated[UploadFile, File()]
) -> dict[str, object]:
    filename = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    fmt = PurePosixPath(filename).suffix[1:].lower()
    if fmt not in MEDIA_TYPES:
        raise HTTPException(415, {"stage": "upload", "message": "不支持的文件扩展名"})
    if len(filename) > 255 or any(ord(char) < 32 or ord(char) == 127 for char in filename):
        raise HTTPException(422, {"stage": "upload", "message": "文件名无效"})
    data = file.file.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(413, {"stage": "upload", "message": "文件超过 10 MiB"})
    if not data:
        raise HTTPException(422, {"stage": "upload", "message": "文件内容为空"})

    ingestion: DocumentIngestion = request.app.state.ingestion
    row, created = ingestion.ingest(filename, fmt, data)
    if not created:
        response.status_code = 200
    return document_response(row)


@router.get("")
def list_documents(
    request: Request,
    offset: Annotated[int, Query(ge=0, le=2**31 - 1)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    title: Annotated[str, Query(max_length=255)] = "",
    status: Literal["indexed", "pending_index"] | None = None,
) -> dict[str, object]:
    repository: DocumentRepository = request.app.state.ingestion.repository
    total, rows = repository.list_documents(offset, limit, title.strip(), status)
    return {"total": total, "offset": offset, "items": [document_response(row) for row in rows]}


@router.get("/{document_id}")
def get_document(request: Request, document_id: str) -> dict[str, object]:
    return document_response(require_document(request.app.state.ingestion.repository, document_id))


@router.get("/{document_id}/chunks")
def get_chunks(
    request: Request,
    document_id: str,
    offset: Annotated[int, Query(ge=0, le=2**31 - 1)] = 0,
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
) -> dict[str, object]:
    repository: DocumentRepository = request.app.state.ingestion.repository
    page = repository.get_chunk_page(document_id, offset, limit)
    if page is None:
        raise HTTPException(404, "文档不存在")
    total, rows = page
    return {
        "total": total,
        "offset": offset,
        "items": [dict(row) for row in rows],
    }


@router.get("/{document_id}/markdown", response_class=Response)
def get_markdown(request: Request, document_id: str) -> Response:
    repository: DocumentRepository = request.app.state.ingestion.repository
    markdown = repository.get_markdown(document_id)
    if markdown is None:
        raise HTTPException(404, "没有 Markdown 产物")
    return Response(markdown, media_type="text/markdown")


@router.get("/{document_id}/original", response_class=Response)
def get_original(request: Request, document_id: str) -> Response:
    ingestion: DocumentIngestion = request.app.state.ingestion
    row = require_document(ingestion.repository, document_id)
    data = ingestion.originals.get(document_id, row.sha256, row.size_bytes)
    filename = quote(row.filename, safe="")
    return Response(
        data, media_type=MEDIA_TYPES[row.format],
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{filename}"},
    )
