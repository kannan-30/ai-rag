"""
api.py - FastAPI Chat API.

Endpoints:
    POST /chat          → Main chat endpoint
    GET  /health        → Liveness probe
    GET  /              → Basic info

Request:  { "question": "..." }
Response:    { "answer": "...", "retrieved_chunks": [...], "confidence": 0.86 }

Run with:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import time
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import config
from rag_pipeline import run_rag_pipeline


# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Agentic AI eBook Chatbot",
    description=(
        "A RAG-based chatbot that answers questions strictly from the "
        "Agentic AI eBook using LangGraph + Pinecone + OpenAI."
    ),
    version="1.0.0",
    docs_url="/docs",       # Swagger UI
    redoc_url="/redoc",     # ReDoc UI
)

# Allow all origins in development (tighten in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Request body for POST /chat."""
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        example="What is Agentic AI?",
    )


class RetrievedChunk(BaseModel):
    """A single context chunk returned by the retrieval step."""
    text: str = Field(..., description="The chunk text from the PDF")
    score: float = Field(..., description="Cosine similarity score [0, 1]")


class ChatResponse(BaseModel):
    """Response body for POST /chat."""
    answer: str = Field(..., description="LLM-generated answer grounded in the PDF")
    retrieved_chunks: List[RetrievedChunk] = Field(
        ..., description="Context chunks used to generate the answer"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Confidence score = cosine similarity of the top retrieved chunk. "
            "Higher values indicate the question is well-covered by the PDF."
        ),
    )
    latency_ms: float = Field(..., description="End-to-end response time in milliseconds")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
async def root():
    """Basic API info endpoint."""
    return {
        "name": "Agentic AI eBook Chatbot",
        "version": "1.0.0",
        "docs": "/docs",
        "chat_endpoint": "POST /chat",
    }


@app.get("/health", tags=["Info"])
async def health():
    """Liveness probe — returns 200 if the service is up."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Main RAG chat endpoint.

    Pipeline:
        1. Validate the incoming question (Pydantic does this automatically).
        2. Invoke the LangGraph RAG pipeline (retrieve → generate).
        3. Return the structured response with answer, chunks, and confidence.

    Error handling:
        - 422 Unprocessable Entity if the request body is malformed.
        - 500 Internal Server Error if the pipeline raises an unexpected exception.
    """
    try:
        t0 = time.perf_counter()
        result = run_rag_pipeline(request.question)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"RAG pipeline error: {str(exc)}",
        ) from exc

    return ChatResponse(
        answer=result["answer"],
        retrieved_chunks=[
            RetrievedChunk(text=c["text"], score=c["score"])
            for c in result["retrieved_chunks"]
        ],
        confidence=result["confidence"],
        latency_ms=latency_ms,
    )
