from typing import Dict


def format_search_results(search_response: Dict) -> str:
    """
    Format search results into a readable string format.

    Args:
        search_response: The response from search_vector_database

    Returns:
        Formatted string with search results
    """
    if search_response.get("status") != "success":
        return f"Error: {search_response.get('message', 'Unknown error')}"

    results = search_response.get("results", [])
    if not results:
        return "No matching documents found."

    formatted = f"Found {len(results)} matching documents:\n\n"

    for i, result in enumerate(results, 1):
        similarity = result.get("similarity", 0)
        formatted += f"Result #{i} [Similarity: {similarity:.3f}]\n"
        formatted += f"ID: {result.get('id')}\n"
        formatted += f"File ID: {result.get('file_id')}\n"

        # Truncate content if it's too long
        content = result.get("content", "")
        if len(content) > 1000:
            content = content[:997] + "..."

        formatted += f"Content: {content}\n\n"

    return formatted
