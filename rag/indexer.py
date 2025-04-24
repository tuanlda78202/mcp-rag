"""
Markdown File Indexer for Supabase Vector Search

1. Finds all *.md files in the specified directory
2. Splits them into chunks using RecursiveCharacterTextSplitter
3. Generates embeddings using Gemini
4. Uploads the embeddings to Supabase for vector search
"""

import argparse
import glob
import os
import sys
import time
from typing import Any, Dict, List

from dotenv import load_dotenv
from google import genai
from google.genai.types import EmbedContentConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase import Client, create_client
from supabase.lib.client_options import ClientOptions

# Load environment variables
load_dotenv()

# ENV
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_SIZE = 768
BATCH_SIZE = 20  # Process files in batches to avoid rate limits
GEMINI_BATCH_LIMIT = 100  # Maximum batch size for Gemini embedding API

# Initialize Supabase client
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY"),
    options=ClientOptions(postgrest_client_timeout=30),
)

# Initialize Gemini client
model_id = os.getenv("GEMINI_EMBEDDING_ID")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def read_markdown_file(file_path: str) -> str:
    """
    Read a markdown file and return its contents

    Args:
        file_path: Path to the markdown file

    Returns:
        String content of the markdown file
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""


def split_text(text: str, file_path: str) -> List[Dict[str, Any]]:
    """
    Split text into chunks and prepare for embedding

    Args:
        text: Text content to split
        file_path: Original file path for tracking

    Returns:
        List of dictionaries with chunk information
    """
    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    # Split the text into chunks
    chunks = text_splitter.split_text(text)

    # Get the relative file path for the file_id
    file_id = os.path.relpath(file_path).replace("\\", "/")

    # Create a list of chunk dictionaries
    chunk_dicts = []
    for i, chunk_text in enumerate(chunks):
        # Get start position (approximate)
        start_pos = i * (CHUNK_SIZE - CHUNK_OVERLAP) if i > 0 else 0
        end_pos = start_pos + len(chunk_text)

        chunk_id = f"{file_id}_{start_pos}-{end_pos}"

        chunk_dicts.append(
            {
                "id": chunk_id,
                "file_id": file_id,
                "content": chunk_text,
                "start_pos": start_pos,
                "end_pos": end_pos,
            }
        )

    return chunk_dicts


def embed_content(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate embeddings for a list of text chunks

    Args:
        chunks: List of chunk dictionaries with content

    Returns:
        Chunks with embeddings added
    """
    # Process in batches of GEMINI_BATCH_LIMIT to avoid API limitations
    all_chunks_with_embeddings = []

    for i in range(0, len(chunks), GEMINI_BATCH_LIMIT):
        batch = chunks[i : i + GEMINI_BATCH_LIMIT]
        print(
            f"Processing embedding batch {i // GEMINI_BATCH_LIMIT + 1} ({len(batch)} chunks)..."
        )

        # Extract text content from chunks
        texts = [chunk["content"] for chunk in batch]

        try:
            # Generate embeddings
            response = client.models.embed_content(
                model=model_id,
                contents=texts,
                config=EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT", output_dimensionality=EMBEDDING_SIZE
                ),
            )

            # Add embeddings to chunks
            for j, embedding in enumerate(response.embeddings):
                batch[j]["embedding"] = embedding.values

            all_chunks_with_embeddings.extend(batch)

        except Exception as e:
            print(
                f"Error generating embeddings for batch {i // GEMINI_BATCH_LIMIT + 1}: {e}"
            )
            # Add chunks without embeddings
            for chunk in batch:
                chunk["embedding"] = []
                all_chunks_with_embeddings.append(chunk)

        # Add a small delay between batches to avoid rate limits
        if i + GEMINI_BATCH_LIMIT < len(chunks):
            time.sleep(0.5)

    return all_chunks_with_embeddings


def index_markdown_files(directory: str, dry_run: bool = False) -> Dict[str, Any]:
    """
    Find all markdown files in a directory and index them

    Args:
        directory: Path to directory to scan for .md files
        dry_run: If True, don't actually upload to Supabase

    Returns:
        Dictionary with statistics about the indexing process
    """
    start_time = time.time()

    # Find all markdown files
    md_files = glob.glob(f"{directory}/**/*.md", recursive=True)

    if not md_files:
        return {"status": "error", "message": f"No markdown files found in {directory}"}

    print(f"Found {len(md_files)} markdown files in {directory}")

    # Statistics
    stats = {
        "files_processed": 0,
        "files_failed": 0,
        "chunks_created": 0,
        "chunks_indexed": 0,
        "processing_time": 0,
    }

    # Process files in batches
    for i in range(0, len(md_files), BATCH_SIZE):
        batch = md_files[i : i + BATCH_SIZE]

        all_chunks = []

        # Process each file in the batch
        for file_path in batch:
            try:
                # Read the markdown file
                content = read_markdown_file(file_path)
                if not content:
                    stats["files_failed"] += 1
                    continue

                # Split the content into chunks
                chunks = split_text(content, file_path)

                # Add to the list of all chunks
                all_chunks.extend(chunks)

                # Update statistics
                stats["files_processed"] += 1
                stats["chunks_created"] += len(chunks)

                print(f"Processed {file_path} - {len(chunks)} chunks")

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                stats["files_failed"] += 1

        # Generate embeddings for all chunks in the batch
        if all_chunks:
            chunks_with_embeddings = embed_content(all_chunks)

            # Filter out chunks without embeddings
            valid_chunks = [
                chunk for chunk in chunks_with_embeddings if chunk.get("embedding")
            ]

            # If not a dry run, upload to Supabase
            if not dry_run and valid_chunks:
                try:
                    # Prepare data for upsert
                    upsert_data = []
                    for chunk in valid_chunks:
                        upsert_data.append(
                            {
                                "id": chunk["id"],
                                "file_id": chunk["file_id"],
                                "content": chunk["content"],
                                "embedding": chunk["embedding"],
                            }
                        )

                    # Upsert to Supabase in batches if needed
                    max_batch = 500  # Maximum batch size for Supabase upsert
                    for j in range(0, len(upsert_data), max_batch):
                        sub_batch = upsert_data[j : j + max_batch]
                        response = (
                            supabase.table("rag_embed").upsert(sub_batch).execute()
                        )
                        print(
                            f"Indexed batch {j // max_batch + 1} ({len(sub_batch)} chunks) to Supabase"
                        )

                    # Update statistics
                    stats["chunks_indexed"] += len(upsert_data)

                except Exception as e:
                    print(f"Error upserting to Supabase: {e}")
            else:
                # In dry run mode, just count valid chunks
                stats["chunks_indexed"] += len(valid_chunks)
                print(
                    f"Dry run: Would have indexed {len(valid_chunks)} chunks to Supabase"
                )

        # Add a small delay between batches to avoid rate limits
        if i + BATCH_SIZE < len(md_files):
            time.sleep(1)

    # Calculate total processing time
    stats["processing_time"] = round(time.time() - start_time, 2)

    return {
        "status": "success",
        "message": f"Processed {stats['files_processed']} files and indexed {stats['chunks_indexed']} chunks in {stats['processing_time']} seconds",
        "stats": stats,
    }


def main():
    """Main function to parse arguments and run the indexer"""
    parser = argparse.ArgumentParser(
        description="Index markdown files for vector search"
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directory to scan for markdown files (default: current directory)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without uploading to Supabase",
    )

    args = parser.parse_args()

    # Validate environment variables
    if not os.getenv("SUPABASE_URL") or not os.getenv("SUPABASE_KEY"):
        print("Error: SUPABASE_URL and SUPABASE_KEY environment variables must be set")
        sys.exit(1)

    if not os.getenv("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable must be set")
        sys.exit(1)

    # Run the indexer
    result = index_markdown_files(args.directory, args.dry_run)

    # Print the result
    if result["status"] == "success":
        print(f"\nSuccess: {result['message']}")
        print("Statistics:")
        for key, value in result["stats"].items():
            print(f"  {key}: {value}")
    else:
        print(f"\nError: {result['message']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
