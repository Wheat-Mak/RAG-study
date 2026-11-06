from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
import json
import re
from itertools import batched
from typing import Literal

from markdown_it import MarkdownIt

TEXT_CHUNK_SIZE = 256
TEXT_OVERLAP = 50
TABLE_CHUNK_SIZE = 700
TABLE_ROWS_PER_CHUNK = 3
TABLE_LAYOUT = "rows-v1"

SourceValue = str | int | bool | list[str]
Cell = str | int | float | bool | None


@dataclass(frozen=True)
class Chunk:
    kind: Literal["text", "section", "parent", "child", "table"]
    text: str
    source: dict[str, SourceValue]
    parent_index: int | None = None


@dataclass(frozen=True)
class TableRow:
    start: int
    end: int
    values: tuple[Cell, ...]


def windows(start: int, end: int) -> Iterator[tuple[int, int]]:
    while start < end:
        stop = min(start + TEXT_CHUNK_SIZE, end)
        yield start, stop
        if stop == end:
            return
        start = stop - TEXT_OVERLAP


def split_text(text: str) -> list[Chunk]:
    return [
        Chunk("text", text[start:end], {"start_char": start, "end_char": end})
        for start, end in windows(0, len(text))
    ]


def split_markdown(text: str) -> list[Chunk]:
    tokens = MarkdownIt("commonmark").parse(text)
    headings = [
        (token.map, int(token.tag[1:]), tokens[index + 1].content)
        for index, token in enumerate(tokens)
        if token.type == "heading_open" and token.level == 0 and token.map
    ]
    line_offsets = [0, *(match.end() for match in re.finditer("\n", text))]
    if line_offsets[-1] != len(text):
        line_offsets.append(len(text))

    chunks: list[Chunk] = []
    heading_path: list[tuple[int, str]] = []
    previous_end = 0
    for (heading_start, heading_end), level, title in headings:
        append_section(
            chunks, text, line_offsets, previous_end, heading_start, heading_path
        )
        while heading_path and heading_path[-1][0] >= level:
            heading_path.pop()
        heading_path.append((level, title))
        previous_end = heading_end
    append_section(
        chunks, text, line_offsets, previous_end, len(line_offsets) - 1, heading_path
    )
    return chunks


def append_section(
    chunks: list[Chunk],
    text: str,
    line_offsets: list[int],
    first_line: int,
    stop_line: int,
    heading_path: list[tuple[int, str]],
) -> None:
    start, end = line_offsets[first_line], line_offsets[stop_line]
    body = text[start:end]
    if not body.strip():
        return
    start += len(body) - len(body.lstrip("\n"))
    end -= len(body) - len(body.rstrip("\n"))
    path = [title for _, title in heading_path]

    def source(left: int, right: int) -> dict[str, SourceValue]:
        return {
            "start_char": left,
            "end_char": right,
            "start_line": bisect_right(line_offsets, left),
            "end_line": bisect_left(line_offsets, right),
            "heading_path": path,
        }

    if end - start <= TEXT_CHUNK_SIZE:
        chunks.append(Chunk("section", text[start:end], source(start, end)))
        return

    # The list index is the persisted ordinal used by child chunks.
    parent = len(chunks)
    chunks.append(Chunk("parent", text[start:end], source(start, end)))
    for left, right in windows(start, end):
        chunks.append(Chunk("child", text[left:right], source(left, right), parent))


def table_cell_text(value: Cell) -> str:
    if value is None:
        return ""
    # Escape embedded tabs and line breaks so each record occupies one line.
    encoded = json.dumps(value, ensure_ascii=False)
    return encoded[1:-1] if isinstance(value, str) else encoded


def split_table(
    headers: list[str], rows: Iterable[TableRow], sheet: str | None = None
) -> Iterator[Chunk]:
    header = "\t".join(map(table_cell_text, headers))
    for group in batched(rows, TABLE_ROWS_PER_CHUNK):
        lines = ["\t".join(map(table_cell_text, row.values)) for row in group]
        text = "\n".join([header, *lines])
        source: dict[str, SourceValue] = {
            "row_start": group[0].start,
            "row_end": group[-1].end,
            "table_layout": TABLE_LAYOUT,
            "rows_per_chunk": TABLE_ROWS_PER_CHUNK,
        }
        if sheet is not None:
            source["sheet"] = sheet
        if len(text) > TABLE_CHUNK_SIZE:
            source["oversized"] = True
        yield Chunk("table", text, source)


def table_rows_for_rerank(text: str, *, row_layout: bool = False) -> list[str]:
    if row_layout:
        headers, *rows = (line.split("\t") for line in text.split("\n"))
        if any(len(row) != len(headers) for row in rows):
            return []
        return named_table_rows(headers, rows)

    # Legacy chunks remain readable while table documents are re-ingested.
    decoder = json.JSONDecoder()
    headers: list[str] = []
    columns: list[list[Cell]] = []
    start = 0
    for separator in re.finditer("：\\[", text):
        if separator.start() <= start:
            continue
        try:
            values, end = decoder.raw_decode(text, separator.start() + 1)
        except json.JSONDecodeError:
            continue
        if end < len(text) and text[end] != "\n":
            continue
        headers.append(text[start:separator.start()])
        columns.append(values)
        start = end + 1

    if start != len(text) + 1 or len({len(column) for column in columns}) != 1:
        return []
    return named_table_rows(headers, zip(*columns, strict=True))


def named_table_rows(headers: list[str], rows: Iterable[Iterable[Cell]]) -> list[str]:
    return [
        "，".join(f"{header}：{value}" for header, value in zip(headers, row, strict=True)
                 if value is not None and value != "")
        for row in rows
    ]
