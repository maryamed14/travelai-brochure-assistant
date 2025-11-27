🧭 TravelAI Brochure Assistant

A Retrieval-Augmented Generation (RAG) system + ReAct Agent for answering travel questions using real PDF brochures.

This project demonstrates a full, production-style AI backend:

PDF ingestion → JSONL dataset

Semantic search retriever (filtering + reranking)

RAG QA pipeline (OpenAI)

ReAct agent with tool calling

Custom tools: brochure_search and city_summary

FastAPI backend (/qa, /agent)

Dockerized deployment

GitHub Actions CI

Offline retrieval evaluation

LangSmith tracing for deep visibility

✨ Features
🔍 PDF → RAG Dataset

Brochures are converted into structured chunks using a clean ingestion pipeline:

python -m travelai.data_ingestion

Output:
data/processed/brochures.jsonl

🧠 Semantic Search Retriever

Built using sentence-transformers embeddings with:

Min-score filtering

Reranking

Deterministic offline evaluation

💬 RAG QA Pipeline

Answers grounded questions such as:

“Which hotel in New York has views of Central Park?”

Includes:

Retrieval

Context assembly

LLM answer generation

Irrelevant-chunk filtering

🤖 ReAct Agent with Tool Calling

Two tools:

brochure_search → factual questions

city_summary → general city descriptions

The agent chooses the correct tool automatically using ReAct reasoning.

🌐 FastAPI Endpoints
POST /qa

RAG question answering.

POST /agent

ReAct agent with tool calling and thought/action/observation traces.

📦 Docker Support

Backend packaged in a single Dockerfile.

🚦 GitHub Actions CI

The CI pipeline executes:

Install dependencies

Run unit tests

Build Docker image

Run ingestion

Run retrieval evaluation (offline, deterministic)

🏗 Architecture
PDFs → Ingestion → brochures.jsonl → Retriever → QA Pipeline → FastAPI
↘ ReAct Agent + Tools
