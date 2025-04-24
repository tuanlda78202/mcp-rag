from typing import List, Union

from google import genai
from google.genai.types import ContentEmbedding, EmbedContentConfig

from app.config import settings
from app.models.embed import EmbeddingTaskTypeEnum

client = genai.Client(api_key=settings.GEMINI_API_KEY)


def embed_content(
    contents: Union[str, List[str]],
    task_type: EmbeddingTaskTypeEnum = EmbeddingTaskTypeEnum.RETRIEVAL_QUERY,
) -> List[ContentEmbedding]:
    """
    Generate embeddings for content using Gemini model
    Args:
        contents : The content to embed, either a string or list of strings
        task_type: The type of embedding task
    Returns:
        List of content embeddings
    """
    try:
        response = client.models.embed_content(
            model=settings.GEMINI_EMBEDDING_ID,
            contents=contents,
            config=EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=settings.RAG_EMBEDDING_SIZE,
            ),
        )
        return response.embeddings
    except Exception:
        return [ContentEmbedding(values=[])]
