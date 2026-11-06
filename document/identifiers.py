from typing import Annotated

from pydantic import StringConstraints

DOCUMENT_ID_PATTERN = r"(md|txt|pdf|docx|csv|xlsx)-[0-9a-f]{64}"
DocumentID = Annotated[str, StringConstraints(pattern=rf"^{DOCUMENT_ID_PATTERN}$")]
ChunkID = Annotated[str, StringConstraints(pattern=rf"^{DOCUMENT_ID_PATTERN}#(0|[1-9][0-9]{{0,7}})$")]
