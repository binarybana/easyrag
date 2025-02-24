from enum import Enum
from typing import Literal, Union
from pydantic import BaseModel, Field
from pathlib import Path
from urllib.parse import ParseResult

class SourceType(str, Enum):
    CODE = "code"
    URL = "url"
    PDF = "pdf"

class Document(BaseModel):
    content: str
    metadata: dict
    source_type: SourceType
    source_path: Union[Path, ParseResult, None] = None

class SearchResult(BaseModel):
    content: str
    metadata: dict
    score: float
    source: str
