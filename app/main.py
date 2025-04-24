from fastapi import FastAPI

from app.api.routes import router as vector_search

app = FastAPI(
    title="Vector Search API",
    description="API for vector search using Supabase's pgvector.",
    version="1.0.0",
)

# Include the router for the vector search API
app.include_router(vector_search, prefix="/api/v1", tags=["vector-search"])
