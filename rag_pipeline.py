"""
rag_pipeline.py - LangGraph RAG pipeline.

Graph topology:
    START → retrieve → generate → END

Node: retrieve
    - Embeds the user question with the same model used during ingestion.
    - Queries Pinecone for the top-k most semantically similar chunks.
    - Attaches them to the shared graph state.

Node: generate
    - Builds a prompt that includes ONLY the retrieved chunks as context.
    - Calls the LLM and returns the grounded answer.
    - Prompt explicitly forbids the model from using outside knowledge.

State object:
    A TypedDict that flows through every node, carrying the question,
    retrieved chunks with scores, the final answer, and a confidence score.
"""

from __future__ import annotations

from typing import List, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

import config


# ── Shared state ──────────────────────────────────────────────────────────────

class RAGState(TypedDict):
    """
    The state object that is passed between LangGraph nodes.

    Fields:
        question        : The user's raw question.
        retrieved_chunks: List of dicts with 'text' and 'score' keys.
        answer          : The LLM-generated answer (empty until generate node).
        confidence      : Similarity score of the top chunk (0.0 until generate).
    """
    question: str
    retrieved_chunks: List[dict]
    answer: str
    confidence: float


# ── Node: retrieve ─────────────────────────────────────────────────────────────

def retrieve(state: RAGState) -> RAGState:
    """
    Retrieve node.

    Workflow:
        1. Initialise the same OpenAI embedding model used at ingestion time.
        2. Connect to the existing Pinecone index via LangChain's
           PineconeVectorStore wrapper.
        3. Call similarity_search_with_score() which:
               a. Embeds the question into a vector.
               b. Runs an approximate nearest-neighbour (ANN) search in Pinecone
                  using cosine similarity.
               c. Returns the top-k (document, score) pairs.
        4. Normalise scores: Pinecone cosine similarity returns values in [0, 1]
           where 1 = identical. We keep them as-is.
        5. Update state with the retrieved chunks.
    """
    embeddings = GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        google_api_key=config.GEMINI_API_KEY,
    )

    vector_store = PineconeVectorStore(
        index_name=config.PINECONE_INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=config.PINECONE_API_KEY,
    )

    # similarity_search_with_score returns List[Tuple[Document, float]]
    results = vector_store.similarity_search_with_score(
        query=state["question"],
        k=config.TOP_K,
    )

    retrieved_chunks = [
        {
            "text": doc.page_content,
            "score": round(float(score), 4),
            "metadata": doc.metadata,   # page number, source file, etc.
        }
        for doc, score in results
    ]

    return {**state, "retrieved_chunks": retrieved_chunks}


# ── Node: generate ─────────────────────────────────────────────────────────────

# System prompt ────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a helpful and knowledgeable guide focused solely on the Agentic AI eBook.
Your goal is to answer questions using only the information provided in the context passages below.

Guidelines:
- Base your answers strictly on the provided context passages. Do not use outside knowledge.
- If the context doesn't contain the answer, politely let the user know that the information isn't covered in the eBook.
- Keep your answers clear, concise, and easy to read.
- Feel free to mention the page number if it's relevant and helpful.
- Maintain a helpful, professional tone.
"""


def _build_context_block(chunks: List[dict]) -> str:
    """Format retrieved chunks into a readable context string for the LLM."""
    lines = []
    for i, chunk in enumerate(chunks, 1):
        page = chunk.get("metadata", {}).get("page", "?")
        lines.append(
            f"--- Chunk {i} (page {page}, score {chunk['score']}) ---\n"
            f"{chunk['text']}"
        )
    return "\n\n".join(lines)


def generate(state: RAGState) -> RAGState:
    """
    Generate node.

    Workflow:
        1. Format the retrieved chunks into a context block.
        2. Build a two-message prompt:
               SystemMessage  → rules + context (grounding instructions)
               HumanMessage   → the user's question
        3. Call the LLM.
        4. Derive confidence as the score of the top-ranked chunk.
           Rationale: the top chunk score is the strongest signal of whether
           the question is answerable from the PDF. If the best match has a
           low score the bot is likely drifting outside the document.
        5. Return the updated state.
    """
    llm = ChatGoogleGenerativeAI(
        model=config.LLM_MODEL,
        temperature=0,          # deterministic – grounding is paramount
        google_api_key=config.GEMINI_API_KEY,
    )

    context_block = _build_context_block(state["retrieved_chunks"])

    messages = [
        SystemMessage(content=f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context_block}"),
        HumanMessage(content=state["question"]),
    ]

    response = llm.invoke(messages)
    # In newer versions of langchain-google-genai, response.content may be a
    # list of content parts rather than a plain string — handle both.
    raw = response.content
    if isinstance(raw, list):
        answer = " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in raw
        ).strip()
    else:
        answer = raw.strip()

    # Confidence = top chunk similarity score (documented in README)
    confidence = (
        state["retrieved_chunks"][0]["score"]
        if state["retrieved_chunks"]
        else 0.0
    )

    return {**state, "answer": answer, "confidence": confidence}


# ── Build the graph ────────────────────────────────────────────────────────────

def build_rag_graph() -> StateGraph:
    """
    Assemble and compile the LangGraph pipeline.

    Graph:  START → retrieve → generate → END

    Using LangGraph (not a simple chain) gives us:
        • Explicit, inspectable node boundaries.
        • Easy extensibility (e.g. add a reranker or fallback node later).
        • Built-in streaming support for each node.
    """
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


# ── Public helper ─────────────────────────────────────────────────────────────

def run_rag_pipeline(question: str) -> dict:
    """
    Convenience wrapper: run the full pipeline for a given question.

    Returns a dict matching the response schema:
        {
            "answer": str,
            "retrieved_chunks": [{"text": str, "score": float}, ...],
            "confidence": float,
        }
    """
    app = build_rag_graph()
    initial_state: RAGState = {
        "question": question,
        "retrieved_chunks": [],
        "answer": "",
        "confidence": 0.0,
    }
    final_state = app.invoke(initial_state)

    # Strip internal metadata before returning to the API layer
    clean_chunks = [
        {"text": c["text"], "score": c["score"]}
        for c in final_state["retrieved_chunks"]
    ]

    return {
        "answer": final_state["answer"],
        "retrieved_chunks": clean_chunks,
        "confidence": final_state["confidence"],
    }
