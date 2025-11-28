## 🚀 Live Demo

Try the TravelAI Brochure Assistant live on Hugging Face Spaces:

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/Rema2/travelai-brochure-assistant)

The Space runs the complete system:
- FastAPI backend (search, QA, and agent endpoints)
- Semantic search over brochure chunks
- RAG question answering using OpenAI models
- Agent with tool-calling (brochure search + city summary)
- Frontend UI for easy interaction

Click the badge above to try it!




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

🔧 Setup

Clone:

git clone https://github.com/maryamed14/travelai-brochure-assistant
cd travelai-brochure-assistant

Create virtual environment:

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Environment variables (.env):

OPENAI_API_KEY=sk-...

▶️ Running Locally

Ingest brochures:

python -m travelai.data_ingestion

Run API:
uvicorn travelai.api.main:app --reload
Swagger UI:
http://localhost:8000/docs

🧪 Retrieval Evaluation
Offline evaluation

python -m travelai.eval.qa_eval
Metrics shown:

City Hit Rate

Answer Hit Rate

Runs automatically in CI.

🚦 CI/CD
GitHub Actions:

Install deps

Run pytest

Build Docker image

Run ingestion

Run evaluation

All green before merging.

🐳 Docker
Build:
docker build -t travelai-brochure-assistant .

