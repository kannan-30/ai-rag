# Agentic AI eBook RAG Chatbot

A production-ready **Retrieval-Augmented Generation (RAG)** chatbot built with **LangGraph**, **Pinecone**, and **Google Gemini**. It answers questions *strictly* from the Agentic AI eBook — no hallucinations, no outside knowledge.

---

## Architecture Overview

```
Ingestion Pipeline:
  PDF  -->  Text Extraction  -->  Chunking  -->  Gemini Embeddings  -->  Pinecone Index

Query Pipeline:
  User Question
       |
  FastAPI POST /chat
       |
  LangGraph Graph
  [ retrieve node ]  -->  [ generate node ]
  (Pinecone ANN)         (LLM + grounding prompt)
       |
  { answer, retrieved_chunks, confidence }
```

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Orchestration | LangGraph |
| Vector Store | Pinecone (serverless) |
| Embeddings | Gemini models/text-embedding-004 |
| LLM | Gemini gemini-1.5-flash |
| API | FastAPI + Uvicorn |
| UI (optional) | Streamlit |

---

## Quick Start

### 1. Clone and Install

```bash
git clone <your-repo-url>
cd rag-ai
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your API keys:

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Gemini API key (embeddings + LLM) |
| `PINECONE_API_KEY` | Pinecone API key |
| `PINECONE_INDEX_NAME` | Name for the Pinecone index (default: agentic-ai-ebook) |
| `EMBEDDING_MODEL` | Gemini embedding model (default: models/text-embedding-004) |
| `LLM_MODEL` | Gemini LLM model (default: gemini-1.5-flash) |
| `CHUNK_SIZE` | Characters per chunk (default: 800) |
| `CHUNK_OVERLAP` | Overlap between consecutive chunks (default: 150) |
| `TOP_K` | Number of chunks to retrieve (default: 5) |

### 3. Ingest the PDF

```bash
python ingest.py
```

Optional — use a custom PDF path:

```bash
python ingest.py --pdf path/to/your-file.pdf
```

This will:
1. Load `data/Ebook-Agentic-AI.pdf`
2. Split it into ~800-character chunks with 150-character overlap
3. Generate a Gemini embedding for each chunk
4. Upsert all vectors + text metadata into Pinecone

### 4a. Run the API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API docs available at: http://localhost:8000/docs

### 4b. Run the Chat UI (optional)

```bash
streamlit run app_ui.py
```

---

## API Reference

### POST /chat

**Request:**
```json
{
  "question": "What is Agentic AI?"
}
```

**Response:**
```json
{
  "answer": "Agentic AI refers to AI systems that can autonomously...",
  "retrieved_chunks": [
    {
      "text": "Agentic AI systems are designed to perceive, reason...",
      "score": 0.8732
    }
  ],
  "confidence": 0.8732,
  "latency_ms": 1423.5
}
```

**Response Fields:**

| Field | Description |
|-------|-------------|
| `answer` | The LLM's grounded answer |
| `retrieved_chunks` | Top-k chunks used as context, with cosine similarity scores |
| `confidence` | Score of the top-ranked retrieved chunk (see below) |
| `latency_ms` | Total end-to-end latency |

### Confidence Score Methodology

`confidence` = **cosine similarity score of the top-ranked Pinecone result**.

- Range: [0, 1] where 1 = identical vectors.
- Interpretation:
  - >= 0.75: High confidence — the question is well-covered by the PDF.
  - 0.50 to 0.74: Medium confidence — partial coverage.
  - < 0.50: Low confidence — the question may be out of scope.
- When confidence is low, the pipeline will return a standard fallback response indicating the answer was not found in the eBook.

---

## Project Structure

```
rag-ai/
├── data/
│   └── Ebook-Agentic-AI.pdf   # Source PDF
├── config.py                  # Centralised env-var config
├── ingest.py                  # PDF --> Pinecone pipeline
├── rag_pipeline.py            # LangGraph RAG (retrieve + generate)
├── api.py                     # FastAPI chat endpoint
├── app_ui.py                  # Streamlit UI
├── sample_queries.py          # Runs 6 sample queries --> sample_outputs.json
├── sample_outputs.json        # Pre-run query results (generated)
├── requirements.txt
├── .env.example
├── README.md
└── details.md                 # Deep-dive architecture explanation
```

---

## Sample Queries

Run all 6 sample queries at once:

```bash
python sample_queries.py
```

See `sample_outputs.json` for the full output.

| # | Question | Expected Behaviour |
|---|----------|--------------------|
| 1 | What is Agentic AI? | Answered from PDF |
| 2 | How does Agentic AI differ from traditional AI? | Answered from PDF |
| 3 | What are the key components of an AI agent? | Answered from PDF |
| 4 | What are the use cases of Agentic AI? | Answered from PDF |
| 5 | What challenges or risks are mentioned? | Answered from PDF |
| 6 | Who is the prime minister of India? | Handled by fallback (out of scope) |

---

## Architecture (Expanded)

See `details.md` for a full explanation of:
- How the ingestion pipeline works
- How LangGraph nodes are structured
- The grounding strategy (prompt engineering)
- The confidence scoring methodology
- Design decisions and trade-offs

---

## Getting API Keys

- Gemini: https://aistudio.google.com/app/apikey
- Pinecone: https://app.pinecone.io/ (Create a free serverless index)

---

## License

MIT
