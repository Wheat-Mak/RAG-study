from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import httpx
from minio.error import MinioException
from sqlalchemy.exc import SQLAlchemyError
from starlette.concurrency import run_in_threadpool
from urllib3.exceptions import HTTPError as ObjectHTTPError

from .ai.chat import ChatModel
from .api.batch_evaluation import router as batch_evaluation_router
from .api.documents import LimitUploadBody, router as documents_router
from .api.evaluation import router as evaluation_router
from .api.retrieval import router as retrieval_router
from .document.parser import ParseFailure
from .document.repository import DocumentBusy, DocumentRepository
from .document.service import DocumentIngestion
from .evaluation.batch import BatchEvaluation
from .evaluation.service import Evaluation
from .infra.object_store import OriginalStore
from .rag.embedding import Embedder
from .rag.errors import UpstreamFailure
from .rag.rerank import Reranker
from .rag.retrieval import Retrieval
from .rag.vector_index import VectorIndex

logger = logging.getLogger(__name__)
WEB_ROOT = Path(__file__).with_name("web")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    repository = DocumentRepository()
    originals = OriginalStore()
    try:
        with httpx.Client(timeout=httpx.Timeout(240.0, connect=10.0), trust_env=False) as client:
            embedder = Embedder(client)
            index = VectorIndex(client, embedder.model, embedder.dimensions)
            await run_in_threadpool(repository.initialize)
            await run_in_threadpool(originals.initialize)
            await run_in_threadpool(index.initialize)
            ingestion = DocumentIngestion(repository, originals, client, embedder, index)
            app.state.ingestion = ingestion
            retrieval = Retrieval(repository, ingestion.pipeline, embedder, index, Reranker(client))
            app.state.retrieval = retrieval
            evaluation = Evaluation(retrieval, ChatModel(client))
            app.state.evaluation = evaluation
            batch = BatchEvaluation(evaluation)
            await run_in_threadpool(batch.start)
            app.state.batch_evaluation = batch
            try:
                yield
            finally:
                await run_in_threadpool(batch.stop)
    finally:
        await run_in_threadpool(repository.close)
        await run_in_threadpool(originals.close)


app = FastAPI(title="RAG Python · 入库、检索与评测", lifespan=lifespan)
app.add_middleware(LimitUploadBody)
app.include_router(documents_router)
app.include_router(retrieval_router)
app.include_router(evaluation_router)
app.include_router(batch_evaluation_router)
app.mount("/assets", StaticFiles(directory=WEB_ROOT / "assets"), name="assets")


@app.get("/", include_in_schema=False)
def workbench() -> RedirectResponse:
    return RedirectResponse("/upload.html")


@app.get("/upload.html", include_in_schema=False)
@app.get("/document.html", include_in_schema=False)
@app.get("/chat.html", include_in_schema=False)
@app.get("/eval-report.html", include_in_schema=False)
def frontend_page(request: Request) -> FileResponse:
    return FileResponse(WEB_ROOT / request.url.path.rsplit("/", 1)[-1])


@app.get("/healthz")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.exception_handler(ParseFailure)
@app.exception_handler(UpstreamFailure)
async def rag_failure(_: Request, exc: ParseFailure | UpstreamFailure) -> JSONResponse:
    if exc.status >= 500:
        logger.warning("RAG operation failed at %s: %s", exc.stage, exc)
    return JSONResponse(
        {"detail": {"stage": exc.stage, "message": str(exc)}}, status_code=exc.status
    )


@app.exception_handler(DocumentBusy)
async def document_busy(_: Request, exc: DocumentBusy) -> JSONResponse:
    return JSONResponse({"detail": {"stage": "upload", "message": str(exc)}}, status_code=409)


@app.exception_handler(SQLAlchemyError)
@app.exception_handler(MinioException)
@app.exception_handler(ObjectHTTPError)
@app.exception_handler(OSError)
async def storage_failure(_: Request, exc: Exception) -> JSONResponse:
    logger.error("Document storage operation failed", exc_info=exc)
    return JSONResponse(
        {"detail": {"stage": "storage", "message": "文档存储暂不可用"}}, status_code=503
    )
