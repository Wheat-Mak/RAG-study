from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import logging
import os

from sqlalchemy import (
    JSON, URL, Column, ForeignKey, Integer, MetaData, String, Table,
    create_engine, func, select, text, tuple_,
)
from sqlalchemy.dialects.mysql import LONGTEXT, insert
from sqlalchemy.engine import Connection, RowMapping
from sqlalchemy.exc import SQLAlchemyError

from .parser import ParsedDocument

logger = logging.getLogger(__name__)
schema = MetaData()
documents = Table(
    "documents",
    schema,
    Column("id", String(72), primary_key=True),
    Column("filename", String(255), nullable=False),
    Column("format", String(4), nullable=False),
    Column("size_bytes", Integer, nullable=False),
    Column("sha256", String(64), nullable=False),
    Column("chunk_count", Integer, nullable=False),
    Column("markdown", LONGTEXT),
    mysql_charset="utf8mb4",
)
chunks = Table(
    "chunks",
    schema,
    Column("document_id", ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True),
    Column("index", Integer, primary_key=True),
    Column("kind", String(8), nullable=False),
    Column("text", LONGTEXT, nullable=False),
    Column("source", JSON, nullable=False),
    Column("parent_index", Integer),
    mysql_charset="utf8mb4",
)
indexes = Table(
    "document_indexes",
    schema,
    Column("document_id", ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True),
    Column("generation", String(32), nullable=False),
    Column("pipeline", String(64), nullable=False),
    Column("embedding_model", String(128), nullable=False),
    Column("embedding_dimensions", Integer, nullable=False),
    Column("vector_index", String(128), nullable=False),
    Column("vector_count", Integer, nullable=False),
    mysql_charset="utf8mb4",
)
document_summary = select(
    documents.c.id, documents.c.filename, documents.c.format,
    documents.c.size_bytes, documents.c.sha256, documents.c.chunk_count,
    indexes.c.generation, indexes.c.pipeline, indexes.c.embedding_model,
    indexes.c.embedding_dimensions, indexes.c.vector_index, indexes.c.vector_count,
).select_from(documents.outerjoin(indexes))


class DocumentBusy(RuntimeError):
    pass


@dataclass(frozen=True)
class PublishedDocument:
    generation: str
    filename: str
    format: str


class DocumentRepository:
    def __init__(self) -> None:
        self.database = os.getenv("RAG_MYSQL_DATABASE", "rag_python")
        self.engine = create_engine(
            URL.create(
                "mysql+pymysql",
                username=os.getenv("RAG_MYSQL_USER", "rag_app"),
                password=os.environ["RAG_MYSQL_PASSWORD"],
                host=os.getenv("RAG_MYSQL_HOST", "127.0.0.1"),
                port=int(os.getenv("RAG_MYSQL_HOST_PORT", "13306")),
                database=self.database,
                query={"charset": "utf8mb4"},
            ),
            connect_args={"connect_timeout": 6, "read_timeout": 20, "write_timeout": 20},
            isolation_level="REPEATABLE READ",
            pool_pre_ping=True,
            hide_parameters=True,
        )

    def initialize(self) -> None:
        schema.create_all(self.engine)

    def close(self) -> None:
        self.engine.dispose()

    def find(self, document_id: str) -> RowMapping | None:
        with self.engine.connect() as connection:
            return self.find_on(connection, document_id)

    def find_on(self, connection: Connection, document_id: str) -> RowMapping | None:
        statement = document_summary.where(documents.c.id == document_id)
        return connection.execute(statement).mappings().one_or_none()

    def has_table_layout(
        self, connection: Connection, document_id: str, layout: str, rows_per_chunk: int,
    ) -> bool:
        source = connection.scalar(select(chunks.c.source).where(
            chunks.c.document_id == document_id, chunks.c.index == 0,
        ))
        return (
            source is not None and source.get("table_layout") == layout
            and source.get("rows_per_chunk") == rows_per_chunk
        )

    def list_documents(
        self, offset: int, limit: int, title: str, status: str | None,
    ) -> tuple[int, list[RowMapping]]:
        statement = document_summary
        if title:
            statement = statement.where(documents.c.filename.contains(title, autoescape=True))
        if status == "indexed":
            statement = statement.where(indexes.c.document_id.is_not(None))
        elif status == "pending_index":
            statement = statement.where(indexes.c.document_id.is_(None))
        page = statement.order_by(documents.c.filename, documents.c.id).offset(offset).limit(limit)
        with self.engine.begin() as connection:
            total = connection.execute(select(func.count()).select_from(statement.subquery())).scalar_one()
            rows = connection.execute(page).mappings()
            return total, list(rows)

    def existing_labels(
        self, pipeline: str, document_ids: set[str], chunk_keys: set[tuple[str, int]],
    ) -> tuple[set[str], set[tuple[str, int]]]:
        with self.engine.connect() as connection:
            doc_query = select(indexes.c.document_id).where(
                indexes.c.pipeline == pipeline, indexes.c.document_id.in_(document_ids),
            )
            chunk_query = select(chunks.c.document_id, chunks.c.index).join(
                indexes, chunks.c.document_id == indexes.c.document_id,
            ).where(
                indexes.c.pipeline == pipeline,
                tuple_(chunks.c.document_id, chunks.c.index).in_(chunk_keys),
            )
            return (
                set(connection.scalars(doc_query)) if document_ids else set(),
                {tuple(row) for row in connection.execute(chunk_query)} if chunk_keys else set(),
            )

    def bound_chunks(self, pipeline: str, keys: set[tuple[str, int]]) -> list[RowMapping]:
        if not keys:
            return []
        statement = select(chunks).join(indexes, chunks.c.document_id == indexes.c.document_id).where(
            indexes.c.pipeline == pipeline, tuple_(chunks.c.document_id, chunks.c.index).in_(keys),
        )
        with self.engine.connect() as connection:
            return list(connection.execute(statement).mappings())

    @contextmanager
    def lock_document(self, document_id: str) -> Iterator[Connection]:
        lock = sha256(f"{self.database}:{document_id}".encode()).hexdigest()
        with self.engine.connect() as connection:
            acquired = connection.execute(text("SELECT GET_LOCK(:name, 10)"), {"name": lock})
            if acquired.scalar_one() != 1:
                raise DocumentBusy("此文档正在入库，请稍后重试")
            try:
                connection.commit()
                yield connection
            finally:
                if not connection.invalidated:
                    try:
                        connection.rollback()
                        connection.execute(text("SELECT RELEASE_LOCK(:name)"), {"name": lock})
                    except SQLAlchemyError:
                        logger.warning("Unable to release document lock", exc_info=True)
                        connection.invalidate()

    def publish(
        self, connection: Connection, document_id: str, filename: str, fmt: str,
        size_bytes: int, digest: str, parsed: ParsedDocument, index: dict[str, str | int],
    ) -> RowMapping:
        with connection.begin():
            document = insert(documents).values(
                id=document_id, filename=filename, format=fmt, size_bytes=size_bytes,
                sha256=digest, chunk_count=len(parsed.chunks), markdown=parsed.markdown,
            )
            connection.execute(document.on_duplicate_key_update(
                chunk_count=document.inserted.chunk_count, markdown=document.inserted.markdown,
            ))
            connection.execute(chunks.delete().where(chunks.c.document_id == document_id))
            for start in range(0, len(parsed.chunks), 200):
                connection.execute(chunks.insert(), [
                    {
                        "document_id": document_id, "index": ordinal, "kind": chunk.kind,
                        "text": chunk.text, "source": chunk.source,
                        "parent_index": chunk.parent_index,
                    }
                    for ordinal, chunk in enumerate(parsed.chunks[start:start + 200], start)
                ])
            entry = insert(indexes).values(document_id=document_id, **index)
            connection.execute(entry.on_duplicate_key_update(**index))
            statement = document_summary.where(documents.c.id == document_id)
            return connection.execute(statement).mappings().one()

    def get_chunk_page(
        self, document_id: str, offset: int, limit: int,
    ) -> tuple[int, list[RowMapping]] | None:
        statement = (
            select(chunks.c.index, chunks.c.kind, chunks.c.text, chunks.c.source, chunks.c.parent_index)
            .where(chunks.c.document_id == document_id)
            .order_by(chunks.c.index)
            .offset(offset)
            .limit(limit)
        )
        with self.engine.begin() as connection:
            document = self.find_on(connection, document_id)
            if document is None:
                return None
            return document.chunk_count, list(connection.execute(statement).mappings())

    def get_markdown(self, document_id: str) -> str | None:
        statement = select(documents.c.markdown).where(documents.c.id == document_id)
        with self.engine.connect() as connection:
            return connection.execute(statement).scalar_one_or_none()

    def active_documents(
        self, pipeline: str, document_id: str | None,
    ) -> dict[str, PublishedDocument]:
        with self.engine.connect() as connection:
            return self._active_documents(connection, pipeline, document_id)

    def _active_documents(
        self, connection: Connection, pipeline: str, document_id: str | None,
    ) -> dict[str, PublishedDocument]:
        statement = select(
            documents.c.id, indexes.c.generation, documents.c.filename, documents.c.format,
        ).join(indexes).where(indexes.c.pipeline == pipeline)
        if document_id is not None:
            statement = statement.where(documents.c.id == document_id)
        return {
            row.id: PublishedDocument(row.generation, row.filename, row.format)
            for row in connection.execute(statement)
        }

    def read_parents(
        self, pipeline: str, document_id: str | None,
        expected: dict[str, PublishedDocument], keys: set[tuple[str, int]],
    ) -> dict[tuple[str, int], RowMapping] | None:
        with self.engine.begin() as connection:
            # Check the scope and read parent ordinals from one SQL snapshot.
            if self._active_documents(connection, pipeline, document_id) != expected:
                return None
            if not keys:
                return {}
            statement = select(chunks).where(
                chunks.c.kind == "parent",
                tuple_(chunks.c.document_id, chunks.c.index).in_(keys),
            )
            return {
                (row.document_id, row.index): row
                for row in connection.execute(statement).mappings()
            }

