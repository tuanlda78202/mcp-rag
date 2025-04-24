from typing import List, Optional

from pydantic import BaseModel, Field


# Request model
class SearchRequest(BaseModel):
    query: str
    match_threshold: Optional[float] = Field(0.5, description="Similarity threshold")
    match_count: Optional[int] = Field(20, description="Maximum number of results")


# Response model
class SearchResult(BaseModel):
    id: str
    file_id: str
    content: str
    similarity: float


class SearchResponse(BaseModel):
    results: List[SearchResult]
