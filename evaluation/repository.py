from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import JSON, Column, ForeignKey, Integer, MetaData, String, Table, func, select
from sqlalchemy.dialects.mysql import DATETIME, LONGTEXT, insert
from sqlalchemy.engine import Connection, RowMapping

from ..document.repository import DocumentRepository
from .dataset import Dataset

schema = MetaData()
datasets = Table(
    "eval_datasets", schema,
    Column("id", String(64), primary_key=True),
    Column("filename", String(255), nullable=False),
    Column("case_count", Integer, nullable=False),
    Column("content", LONGTEXT, nullable=False),
    Column("bindings", JSON, nullable=False),
    Column("builtin", Integer, nullable=False),
    mysql_charset="utf8mb4",
)
runs = Table(
    "eval_runs", schema,
    Column("id", String(32), primary_key=True),
    Column("dataset_id", ForeignKey("eval_datasets.id"), nullable=False),
    Column("status", String(16), nullable=False),
    Column("options", JSON, nullable=False),
    Column("runtime", JSON, nullable=False),
    Column("corpus", JSON, nullable=False),
    Column("total_cases", Integer, nullable=False),
    Column("completed_cases", Integer, nullable=False),
    Column("summary", JSON(none_as_null=True)),
    Column("error", String(512)),
    Column("created_at", DATETIME(fsp=6), nullable=False),
    Column("finished_at", DATETIME(fsp=6)),
    mysql_charset="utf8mb4",
)
cases = Table(
    "eval_run_cases", schema,
    Column("run_id", ForeignKey("eval_runs.id", ondelete="CASCADE"), primary_key=True),
    Column("ordinal", Integer, primary_key=True),
    Column("case_id", String(100), nullable=False),
    Column("question", String(4096), nullable=False),
    Column("reference_answer", LONGTEXT, nullable=False),
    Column("payload", JSON, nullable=False),
    mysql_charset="utf8mb4",
)
dataset_metadata = select(datasets.c.id, datasets.c.filename, datasets.c.case_count, datasets.c.builtin)
run_metadata = select(runs, datasets.c.filename.label("dataset_name")).join(datasets)


def utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class EvaluationRepository:
    def __init__(self, documents: DocumentRepository) -> None:
        self.documents = documents
        self.engine = documents.engine

    def initialize(self) -> None:
        schema.create_all(self.engine)

    def save_dataset(self, dataset: Dataset, bindings: dict[str, str], *, builtin: bool = False) -> RowMapping:
        with self.engine.begin() as connection:
            statement = insert(datasets).values(
                id=dataset.id, filename=dataset.filename, case_count=len(dataset.cases),
                content=dataset.text, bindings=bindings, builtin=int(builtin),
            )
            if builtin:
                connection.execute(datasets.update().where(
                    datasets.c.filename == dataset.filename, datasets.c.id != dataset.id,
                    datasets.c.builtin == 1,
                ).values(builtin=0))
                connection.execute(statement.on_duplicate_key_update(builtin=1))
            else:
                connection.execute(statement.on_duplicate_key_update(id=datasets.c.id))
            return connection.execute(dataset_metadata.where(datasets.c.id == dataset.id)).mappings().one()

    def list_datasets(self) -> list[RowMapping]:
        with self.engine.connect() as connection:
            statement = dataset_metadata.order_by(datasets.c.builtin.desc(), datasets.c.filename, datasets.c.id)
            return list(connection.execute(statement).mappings())

    def get_dataset(self, dataset_id: str) -> RowMapping | None:
        with self.engine.connect() as connection:
            return connection.execute(select(datasets).where(datasets.c.id == dataset_id)).mappings().one_or_none()

    def enqueue(
        self, dataset_id: str, total: int, options: dict[str, Any],
        runtime: dict[str, Any], corpus: dict[str, str],
    ) -> RowMapping:
        run_id = uuid4().hex
        with self.engine.begin() as connection:
            connection.execute(runs.insert().values(
                id=run_id, dataset_id=dataset_id, status="queued", options=options,
                runtime=runtime, corpus=corpus, total_cases=total, completed_cases=0, created_at=utc_now(),
            ))
            return connection.execute(run_metadata.where(runs.c.id == run_id)).mappings().one()

    def get_run(self, run_id: str) -> RowMapping | None:
        with self.engine.connect() as connection:
            return connection.execute(run_metadata.where(runs.c.id == run_id)).mappings().one_or_none()

    def list_runs(self, offset: int, limit: int) -> tuple[int, list[RowMapping]]:
        statement = run_metadata.order_by(runs.c.created_at.desc(), runs.c.id).offset(offset).limit(limit)
        with self.engine.begin() as connection:
            total = connection.execute(select(func.count()).select_from(runs)).scalar_one()
            return total, list(connection.execute(statement).mappings())

    def claim(self, connection: Connection) -> RowMapping | None:
        statement = run_metadata.where(runs.c.status.in_(("queued", "running"))).order_by(runs.c.created_at, runs.c.id).limit(1)
        row = connection.execute(statement).mappings().first()
        if row is not None and row.status == "queued":
            connection.execute(runs.update().where(runs.c.id == row.id).values(status="running"))
        connection.commit()
        return row

    def saved_ordinals(self, connection: Connection, run_id: str) -> set[int]:
        ordinals = set(connection.scalars(select(cases.c.ordinal).where(cases.c.run_id == run_id)))
        connection.commit()
        return ordinals

    def checkpoint(self, connection: Connection, run_id: str, ordinal: int, case: dict[str, Any]) -> None:
        connection.execute(cases.insert().values(run_id=run_id, ordinal=ordinal, **case))
        connection.execute(runs.update().where(runs.c.id == run_id).values(completed_cases=runs.c.completed_cases + 1))
        connection.commit()

    def results(self, run_id: str, offset: int = 0, limit: int = 100) -> list[RowMapping]:
        statement = select(cases).where(cases.c.run_id == run_id).order_by(cases.c.ordinal).offset(offset).limit(limit)
        with self.engine.connect() as connection:
            return list(connection.execute(statement).mappings())

    def finish(
        self, connection: Connection, run_id: str, summary: dict[str, Any], error: str | None,
    ) -> None:
        connection.execute(runs.update().where(runs.c.id == run_id).values(
            status="failed" if error else "completed", summary=summary, error=error, finished_at=utc_now(),
        ))
        connection.commit()
