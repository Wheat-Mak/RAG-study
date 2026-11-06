from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError, model_validator

from ..document.identifiers import ChunkID, DocumentID
from ..document.repository import DocumentRepository

MAX_DATASET_BYTES = 2 * 1024 * 1024
MAX_CASES = 100
BUILTIN_DATASET = Path(__file__).with_name("datasets") / "aurora-100.jsonl"
Fingerprint = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class DatasetError(ValueError):
    pass


class DatasetCase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4096)]
    reference_answer: Annotated[str, StringConstraints(strip_whitespace=True, max_length=8000)] = Field(
        default="", alias="groundTruthAnswer",
    )
    document_ids: list[DocumentID] = Field(default_factory=list, alias="groundTruthDocumentIds", max_length=100)
    chunk_ids: list[ChunkID] = Field(default_factory=list, alias="groundTruthChunkIds", max_length=100)
    fingerprints: dict[ChunkID, Fingerprint] = Field(default_factory=dict, alias="chunkFingerprints")

    @model_validator(mode="after")
    def check_fingerprints(self) -> Self:
        if not self.fingerprints.keys() <= set(self.chunk_ids):
            raise ValueError("分块指纹必须对应本题标签")
        return self


@dataclass(frozen=True)
class Dataset:
    id: str
    filename: str
    text: str
    cases: list[DatasetCase]

    def bind(self, repository: DocumentRepository, pipeline: str) -> dict[str, str]:
        documents = repository.active_documents(pipeline, None)
        doc_ids = {label for case in self.cases for label in case.document_ids}
        missing = doc_ids - documents.keys()
        if missing:
            raise DatasetError(f"标准文档不在当前语料中：{min(missing)}")
        keys = {
            (doc_id, int(index)) for case in self.cases
            for doc_id, index in (label.split("#") for label in case.chunk_ids)
        }
        rows = repository.bound_chunks(pipeline, keys)
        bindings: dict[str, str] = {}
        for row in rows:
            data = [row.kind, row.parent_index, row.source, row.text]
            encoded = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            bindings[f"{row.document_id}#{row.index}"] = sha256(encoded.encode()).hexdigest()
        for case in self.cases:
            for label in case.chunk_ids:
                actual = bindings.get(label)
                expected = case.fingerprints.get(label)
                if actual is None or (expected is not None and actual != expected):
                    raise DatasetError(f"{case.id} 的分块标签不存在或内容已变化：{label}")
        return bindings


def parse_dataset(filename: str, data: bytes) -> Dataset:
    if len(data) > MAX_DATASET_BYTES:
        raise DatasetError("评测集超过 2 MiB")
    try:
        text = data.decode("utf-8-sig").replace("\r\n", "\n")
    except UnicodeDecodeError as exc:
        raise DatasetError("评测集必须使用 UTF-8 编码") from exc

    cases: list[DatasetCase] = []
    seen: set[str] = set()
    for number, line in enumerate(text.split("\n"), 1):
        if not line.strip():
            continue
        if len(cases) >= MAX_CASES:
            raise DatasetError("评测集最多包含 100 题")
        try:
            case = DatasetCase.model_validate_json(line)
        except ValidationError as exc:
            location = ".".join(map(str, exc.errors()[0]["loc"])) or "JSON"
            raise DatasetError(f"第 {number} 行的 {location} 格式不正确") from exc
        if case.id in seen:
            raise DatasetError(f"第 {number} 行重复使用题号 {case.id}")
        seen.add(case.id)
        cases.append(case)
    if not cases:
        raise DatasetError("评测集为空")
    return Dataset(sha256(text.encode()).hexdigest(), filename, text, cases)
