import logging
from typing import Any, Dict, List

from app.client.gemini import embed_content
from app.client.supabase import supabase
from app.config import settings
from app.models.embed import EmbeddingTaskTypeEnum

logger = logging.getLogger(__name__)


async def search_documents(
    query: str, match_threshold: float = None, match_count: int = None
) -> List[Dict[str, Any]]:
    """
    Search documents using semantic search with Gemini embeddings and Supabase pgvector

    Args:
        query: The search query text
        match_threshold: Similarity threshold (optional, uses config default if not provided)
        match_count: Maximum number of results to return (optional, uses config default if not provided)

    Returns:
        List of matching documents with id, file_id, content, and similarity score
    """
    logger.info(f"Searching documents with query: '{query}'")

    # Use provided values or fall back to config defaults
    threshold = (
        match_threshold if match_threshold is not None else settings.RAG_MATCH_THRESHOLD
    )
    count = match_count if match_count is not None else settings.RAG_MATCH_COUNT

    # Generate embeddings for the query
    embeds = embed_content(query, task_type=EmbeddingTaskTypeEnum.RETRIEVAL_QUERY)

    # If embedding failed or returned empty, return empty result
    if not embeds or not embeds[0].values:
        logger.warning(f"Failed to generate embeddings for query: '{query}'")
        return []

    # Search for matches using Supabase RPC function
    result = supabase.rpc(
        "match_embed",
        params={
            "p_query_embedding": embeds[0].values,
            "p_match_threshold": threshold,
            "p_match_count": count,
        },
    ).execute()

    # Return the matching documents
    if result.data:
        # Process results directly without pandas
        formatted_results = []
        for item in result.data:
            # Format similarity score to 3 decimal places
            if "similarity" in item:
                item["similarity"] = float(f"{item['similarity']:.3f}")

            formatted_results.append(item)

        logger.info(f"Found {len(formatted_results)} results for query: '{query}'")
        return formatted_results

    logger.info(f"No results found for query: '{query}'")
    return []
