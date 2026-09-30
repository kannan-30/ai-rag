"""
ingest.py - PDF ingestion pipeline.

Steps performed:
  1. Load the PDF and extract text page-by-page
  2. Split text into overlapping chunks
  3. Generate an OpenAI embedding per chunk
  4. Upsert vectors + chunk text metadata into Pinecone

Run with:
    python ingest.py
  or optionally pass a custom PDF path:
    python ingest.py --pdf path/to/file.pdf
"""

import argparse
import os
import sys
import time
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

import config


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_and_split_pdf(pdf_path: str) -> List:
    """
    Load a PDF with PyPDFLoader (page-by-page) and split into chunks.

    Why RecursiveCharacterTextSplitter?
    - It tries to split on paragraph breaks first, then sentence breaks,
      then word breaks — preserving semantic units as much as possible.
    - chunk_size controls the maximum characters per chunk.
    - chunk_overlap ensures continuity between consecutive chunks so that
      a sentence split across a boundary is still retrievable.
    """
    if not os.path.exists(pdf_path):
        sys.exit(f"[ERROR] PDF not found: {pdf_path}")

    print(f"[1/4] Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"      → {len(pages)} pages loaded")

    print(f"[2/4] Splitting into chunks "
          f"(size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    print(f"      → {len(chunks)} chunks created")
    return chunks


def get_or_create_index(pc: Pinecone, index_name: str, dimension: int) -> None:
    """
    Create a Pinecone serverless index if it does not already exist.

    dimension must match the embedding model output:
      - text-embedding-3-small  → 1536
      - text-embedding-3-large  → 3072
      - text-embedding-ada-002  → 1536
    """
    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        print(f"      → Creating Pinecone index '{index_name}' (dim={dimension})")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Wait until the index is ready
        while not pc.describe_index(index_name).status["ready"]:
            print("      → Waiting for index to be ready…")
            time.sleep(2)
        print("      → Index ready ✓")
    else:
        print(f"      → Index '{index_name}' already exists ✓")


def ingest(pdf_path: str) -> None:
    """
    Full ingestion pipeline: PDF → chunks → embeddings → Pinecone.
    """
    # Step 1 & 2: Load PDF and split
    chunks = load_and_split_pdf(pdf_path)

    # Step 3: Set up embedding model
    print(f"[3/4] Initialising embeddings model: {config.EMBEDDING_MODEL}")
    embeddings = GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        google_api_key=config.GEMINI_API_KEY,
    )

    # Determine embedding dimension by encoding a test string
    sample_vec = embeddings.embed_query("dimension check")
    dimension = len(sample_vec)
    print(f"      → Embedding dimension: {dimension}")

    # Step 4: Pinecone setup and upsert
    print(f"[4/4] Upserting to Pinecone index '{config.PINECONE_INDEX_NAME}'")
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    get_or_create_index(pc, config.PINECONE_INDEX_NAME, dimension)

    # Free tier limit: 100 requests per minute. Let's do batches of 90 with a 60s sleep.
    batch_size = 90
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        print(f"      → Upserting batch {i // batch_size + 1}/{(len(chunks) - 1) // batch_size + 1} ({len(batch)} chunks)")
        PineconeVectorStore.from_documents(
            documents=batch,
            embedding=embeddings,
            index_name=config.PINECONE_INDEX_NAME,
            pinecone_api_key=config.PINECONE_API_KEY,
        )
        if i + batch_size < len(chunks):
            print("      → Waiting 60s for Gemini free tier rate limit to reset...")
            time.sleep(60)

    print(f"\n✅ Ingestion complete — {len(chunks)} chunks stored in Pinecone.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDF into Pinecone")
    parser.add_argument(
        "--pdf",
        default=config.PDF_PATH,
        help=f"Path to the PDF file (default: {config.PDF_PATH})",
    )
    args = parser.parse_args()
    ingest(args.pdf)
