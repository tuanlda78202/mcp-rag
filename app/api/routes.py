from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader

from app.config import settings
from app.models.search import SearchRequest, SearchResponse
from app.services.search import search_documents

router = APIRouter()

api_key_header = APIKeyHeader(name="Search-Key")


# Validate API key
async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key != settings.SEARCH_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key"
        )
    return api_key


@router.post("/search", response_model=SearchResponse, tags=["search"])
async def search(
    request: SearchRequest, api_key: str = Depends(get_api_key)
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Search for documents using semantic search

    This endpoint performs semantic search using Gemini embeddings and Supabase pgvector.
    It returns documents matching the query ranked by similarity.
    """
    results = await search_documents(
        query=request.query,
        match_threshold=request.match_threshold,
        match_count=request.match_count,
    )

    return {"results": results}
