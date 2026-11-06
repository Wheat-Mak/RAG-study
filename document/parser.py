from collections.abc import Iterable, Iterator
import csv
from dataclasses import dataclass
from datetime import date, time, timedelta
from io import BytesIO, StringIO
import math
from pathlib import PurePosixPath
from xml.etree.ElementTree import ParseError
from zipfile import BadZipFile, ZipFile

import httpx
from openpyxl import load_workbook
from openpyxl.utils.exceptions import InvalidFileException

from ..rag.chunking import Cell, Chunk, TableRow, split_markdown, split_table, split_text

MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_MINERU_RESPONSE = 16 * 1024 * 1024

MEDIA_TYPES = {
    "md": "text/markdown",
    "txt": "text/plain",
    "csv": "text/csv",
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


class ParseFailure(ValueError):
    def __init__(self, stage: str, message: str, status: int = 422) -> None:
        super().__init__(message)
        self.stage = stage
        self.status = status


@dataclass(frozen=True)
class ParsedDocument:
    chunks: list[Chunk]
    markdown: str | None = None


def decode_text(data: bytes, stage: str) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        status = 502 if stage == "mineru" else 422
        raise ParseFailure(stage, "正文必须使用 UTF-8 编码", status) from exc


def read_mineru_markdown(client: httpx.Client, url: str, fmt: str, data: bytes) -> str:
    params = {
        "backend": "pipeline",
        "response_format_zip": "true",
        "return_md": "true",
        "return_images": "false",
        "image_analysis": "false",
        "return_middle_json": "false",
        "return_model_output": "false",
    }
    output = BytesIO()
    try:
        with client.stream(
            "POST", f"{url}/file_parse",
            files={"files": (f"document.{fmt}", data, MEDIA_TYPES[fmt])},
            data=params,
        ) as response:
            response.raise_for_status()
            for part in response.iter_bytes():
                if output.tell() + len(part) > MAX_MINERU_RESPONSE:
                    raise ParseFailure("mineru", "转换结果超过大小上限", 502)
                output.write(part)
    except httpx.TimeoutException as exc:
        raise ParseFailure("mineru", "MinerU 转换超时", 504) from exc
    except httpx.HTTPError as exc:
        raise ParseFailure("mineru", "MinerU 转换请求失败", 502) from exc

    try:
        with ZipFile(output) as archive:
            entries = [
                entry for entry in archive.infolist()
                if not entry.is_dir() and PurePosixPath(entry.filename).suffix.lower() == ".md"
            ]
            if len(entries) != 1:
                raise ParseFailure("mineru", "转换结果必须包含一个 Markdown 文件", 502)
            with archive.open(entries[0]) as artifact:
                markdown = artifact.read(MAX_FILE_BYTES + 1)
    except (BadZipFile, NotImplementedError, RuntimeError, OSError) as exc:
        raise ParseFailure("mineru", "转换结果不是可读取的 ZIP", 502) from exc
    if len(markdown) > MAX_FILE_BYTES:
        raise ParseFailure("mineru", "转换后的 Markdown 超过大小上限", 502)
    return decode_text(markdown, "mineru")


def normalize_cell(value: object) -> Cell:
    if value is None or isinstance(value, (str, int, bool)):
        return value if value != "" else None
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, (date, time)):
        return value.isoformat()
    if isinstance(value, (timedelta, float)):
        return str(value)
    raise ParseFailure("parse", "表格包含不支持的单元格值")


def table_chunks(rows: Iterable[TableRow], sheet: str | None = None) -> Iterator[Chunk]:
    nonempty = (row for row in rows if any(value is not None for value in row.values))
    header = next(nonempty, None)
    if header is None:
        return
    width = max(index for index, value in enumerate(header.values, 1) if value is not None)
    headers = [
        str(value).strip() if value is not None and str(value).strip() else f"列{index}"
        for index, value in enumerate(header.values[:width], 1)
    ]
    if len(set(headers)) != len(headers):
        raise ParseFailure("parse", "表头包含重复字段")

    def aligned_rows() -> Iterator[TableRow]:
        for row in nonempty:
            if any(value is not None for value in row.values[width:]):
                raise ParseFailure("parse", f"第 {row.start} 行包含表头之外的数据")
            values = row.values[:width]
            yield TableRow(row.start, row.end, values + (None,) * (width - len(values)))

    yield from split_table(headers, aligned_rows(), sheet)


def csv_rows(data: bytes) -> Iterator[TableRow]:
    reader = csv.reader(StringIO(decode_text(data, "parse"), newline=""), strict=True)
    try:
        start = 1
        for cells in reader:
            yield TableRow(start, reader.line_num, tuple(map(normalize_cell, cells)))
            start = reader.line_num + 1
    except csv.Error as exc:
        raise ParseFailure("parse", "CSV 内容格式无效") from exc


def validate_office_archive(data: bytes, member: str) -> None:
    try:
        with ZipFile(BytesIO(data)) as archive:
            archive.getinfo(member)
            if sum(entry.file_size for entry in archive.infolist()) > MAX_ARCHIVE_BYTES:
                raise ParseFailure("parse", "Office 文件解压后超过大小上限")
    except (BadZipFile, KeyError) as exc:
        raise ParseFailure("parse", "Office 文件与扩展名不符或已损坏") from exc


def xlsx_chunks(data: bytes) -> list[Chunk]:
    validate_office_archive(data, "xl/workbook.xml")
    try:
        workbook = load_workbook(BytesIO(data), read_only=True, data_only=True)
        try:
            chunks: list[Chunk] = []
            for sheet in workbook.worksheets:
                if (sheet.max_row or 1) * (sheet.max_column or 1) > 1_000_000:
                    raise ParseFailure("parse", "工作表声明的单元格范围过大")
                rows = (
                    TableRow(index, index, tuple(map(normalize_cell, cells)))
                    for index, cells in enumerate(sheet.values, 1)
                )
                chunks.extend(table_chunks(rows, sheet.title))
            return chunks
        finally:
            workbook.close()
    except (InvalidFileException, BadZipFile, ParseError, KeyError, OSError) as exc:
        raise ParseFailure("parse", "XLSX 内容格式无效") from exc


def parse_document(fmt: str, data: bytes, client: httpx.Client, mineru_url: str) -> ParsedDocument:
    if fmt == "csv":
        result = ParsedDocument(list(table_chunks(csv_rows(data))))
    elif fmt == "xlsx":
        result = ParsedDocument(xlsx_chunks(data))
    elif fmt == "txt":
        text = decode_text(data, "parse")
        result = ParsedDocument(split_text(text) if text.strip() else [])
    else:
        if fmt == "pdf" and not data.startswith(b"%PDF-"):
            raise ParseFailure("parse", "PDF 文件与扩展名不符或已损坏")
        if fmt == "docx":
            validate_office_archive(data, "word/document.xml")
        markdown = (
            decode_text(data, "parse") if fmt == "md"
            else read_mineru_markdown(client, mineru_url, fmt, data)
        )
        markdown = markdown.replace("\r\n", "\n").replace("\r", "\n")
        result = ParsedDocument(split_markdown(markdown), markdown)
    if not result.chunks:
        raise ParseFailure("chunk", "文档没有可切分的正文或数据行")
    return result
