from hashlib import sha256
from itertools import batched
import json
import logging
import os
from uuid import uuid4

import httpx
from sqlalchemy.engine import RowMapping

from ..infra.object_store import OriginalStore
from ..rag.chunking import (
    TABLE_CHUNK_SIZE, TABLE_LAYOUT, TABLE_ROWS_PER_CHUNK, TEXT_CHUNK_SIZE, TEXT_OVERLAP,
)
from ..rag.embedding import Embedder
from ..rag.errors import UpstreamFailure
from ..rag.vector_index import VectorIndex
from .parser import MEDIA_TYPES, parse_document
from .repository import DocumentRepository

logger = logging.getLogger(__name__)


class DocumentIngestion:
    def __init__(
        self, repository: DocumentRepository, originals: OriginalStore,
        client: httpx.Client, embedder: Embedder, index: VectorIndex,
    ) -> None:
        self.repository = repository
        self.originals = originals
        self.client = client
        self.embedder = embedder
        self.index = index
        self.mineru_url = os.getenv("RAG_MINERU_URL", "http://127.0.0.1:8000").rstrip("/")
        config = json.dumps([
            "ingestion-v1", TEXT_CHUNK_SIZE, TEXT_OVERLAP, TABLE_CHUNK_SIZE,
            embedder.url, embedder.model, embedder.dimensions, index.url, index.name,
        ])
        self.pipeline = sha256(config.encode()).hexdigest()

    def ingest(self, filename: str, fmt: str, data: bytes) -> tuple[RowMapping, bool]:
        digest = sha256(data).hexdigest()
        document_id = f"{fmt}-{digest}"
        with self.repository.lock_document(document_id) as connection:
            existing = self.repository.find_on(connection, document_id)
            layout_is_current = fmt not in {"csv", "xlsx"} or (
                existing is not None
                and self.repository.has_table_layout(
                    connection, document_id, TABLE_LAYOUT, TABLE_ROWS_PER_CHUNK,
                )
            )
            connection.commit()
            if (
                existing is not None and existing.pipeline == self.pipeline
                and layout_is_current
                and self.index.count(document_id, existing.generation) == existing.vector_count
            ):
                self.index.delete_generations(document_id, existing.generation, keep=True)
                return existing, False

            parsed = parse_document(fmt, data, self.client, self.mineru_url)
            self.originals.put(document_id, data, MEDIA_TYPES[fmt])
            generation = uuid4().hex
            indexable = (
                (ordinal, chunk) for ordinal, chunk in enumerate(parsed.chunks)
                if chunk.kind != "parent" and chunk.text.strip()
            )
            indexed = 0
            try:
                for batch in batched(indexable, self.embedder.batch_size):
                    vectors = self.embedder.embed([chunk.text for _, chunk in batch])
                    self.index.write(document_id, generation, list(batch), vectors)
                    indexed += len(batch)
                if not indexed or self.index.count(document_id, generation) != indexed:
                    raise UpstreamFailure("index", "向量入库数量与分块数量不一致")
            except UpstreamFailure:
                try:
                    self.index.delete_generations(document_id, generation, keep=False)
                except UpstreamFailure:
                    logger.warning("Staged generation requires cleanup: %s", generation, exc_info=True)
                raise

            # Retain staged vectors on an uncertain SQL commit. A later request
            # checks the published generation before pruning unreferenced data.
            published = self.repository.publish(
                connection, document_id, filename, fmt, len(data), digest, parsed,
                {
                    "generation": generation, "pipeline": self.pipeline,
                    "embedding_model": self.embedder.model,
                    "embedding_dimensions": self.embedder.dimensions,
                    "vector_index": self.index.name, "vector_count": indexed,
                },
            )
            self.index.delete_generations(document_id, generation, keep=True)
            return published, existing is None
