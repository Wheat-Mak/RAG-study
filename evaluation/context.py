from collections.abc import Iterator
import json
import re

from ..rag.answer import AnswerContext
from ..rag.chunking import TABLE_LAYOUT
from ..rag.retrieval import Evidence

SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？!?])\s*|(?<=\.)\s+|\n+")
TABLE_COLUMN = re.compile(r"：\[")


def table_rows(text: str, visible_end: int) -> Iterator[str]:
    decoder = json.JSONDecoder()
    columns: list[tuple[str, list[object]]] = []
    column_start = 0
    for marker in TABLE_COLUMN.finditer(text):
        if marker.start() < column_start:
            continue
        try:
            values, column_end = decoder.raw_decode(text, marker.end() - 1)
        except json.JSONDecodeError:
            continue
        if column_end < len(text) and text[column_end] != "\n":
            continue
        if column_end > visible_end:
            visible_values = []
            cursor = marker.end()
            for value in values:
                _, value_end = decoder.raw_decode(text, cursor)
                if value_end > visible_end:
                    break
                visible_values.append(value)
                cursor = value_end + 2  # Values are serialized with ", " between them.
            values = visible_values
        columns.append((text[column_start:marker.start()], values))
        column_start = column_end + 1

    # Each row must have a visible value in every column, including visible nulls.
    headers = [header for header, _ in columns]
    for row in zip(*(values for _, values in columns)):
        yield "；".join(
            f"{header}：{json.dumps(value, ensure_ascii=False)}"
            for header, value in zip(headers, row, strict=True)
        )


def tabular_rows(text: str, visible_end: int) -> Iterator[str]:
    lines = text[:visible_end].split("\n")
    if visible_end < len(text) and text[visible_end] != "\n":
        lines.pop()
    if not lines:
        return
    headers, *rows = (line.split("\t") for line in lines)
    for row in rows:
        if len(row) == len(headers):
            yield "；".join(
                f"{header}：{value or 'null'}" for header, value in zip(headers, row, strict=True)
            )


def context_sentences(contexts: list[AnswerContext], evidence: list[Evidence]) -> dict[str, str]:
    sentences: dict[str, str] = {}
    for context, chunk in zip(contexts, evidence):
        if chunk.kind != "table":
            units = (text.strip() for text in SENTENCE_BOUNDARY.split(context.text))
        elif chunk.source.get("table_layout") == TABLE_LAYOUT:
            units = tabular_rows(chunk.text, len(context.text))
        else:
            units = table_rows(chunk.text, len(context.text))
        content = (unit for unit in units if any(character.isalnum() for character in unit))
        sentences.update((f"{context.id}.S{index}", unit) for index, unit in enumerate(content, 1))
    return sentences
