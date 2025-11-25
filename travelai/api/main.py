from __future__ import annotations

from functools import lru_cache
from typing import Optional, List

from fastapi import FastAPI
from pydantic import BaseModel

# Load environment variables (OPENAI_API_KEY) from .env
from dotenv import load_dotenv
load_dotenv()

from travelai.config import BROCHURES_JSONL
from travelai.nlp import BrochureRetriever
from travelai.qa import BrochureQAPipeline
from travelai.agent import build_travel_agent


class SearchRequest(BaseModel):
    query: str
    k: Optional[int] = 5


class SearchResult(BaseModel):
    city: str
    source_file: str
    chunk_id: int
    text: str
    score: float


class QARequest(BaseModel):
    question: str
    k: Optional[int] = 5


class QAResponse(BaseModel):
    answer: str
    context: List[SearchResult]


class AgentRequest(BaseModel):
    question: str


class AgentResponse(BaseModel):
    answer: str


app = FastAPI(
    title="TravelAI Brochure Assistant",
    description="Semantic search, RAG QA, and agentic reasoning over travel brochures.",
    version="0.3.0",
)


@lru_cache(maxsize=1)
def get_retriever() -> BrochureRetriever:
    retriever = BrochureRetriever(BROCHURES_JSONL)
    retriever.load()
    return retriever


@lru_cache(maxsize=1)
def get_qa_pipeline() -> BrochureQAPipeline:
    return BrochureQAPipeline(model_name="gpt-4o-mini")


@lru_cache(maxsize=1)
def get_travel_agent():
    return build_travel_agent(model_name="gpt-4o-mini")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/search", response_model=List[SearchResult])
def search(req: SearchRequest) -> List[SearchResult]:
    retriever = get_retriever()
    k = req.k or 5
    chunks = retriever.search(req.query, k=k)

    return [
        SearchResult(
            city=c.city,
            source_file=c.source_file,
            chunk_id=c.chunk_id,
            text=c.text,
            score=c.score,
        )
        for c in chunks
    ]


@app.post("/qa", response_model=QAResponse)
def qa(req: QARequest) -> QAResponse:
    pipeline = get_qa_pipeline()
    k = req.k or 5
    result = pipeline.answer(req.question, k=k)

    context_results: List[SearchResult] = []
    for c in result["context"]:
        context_results.append(
            SearchResult(
                city=c["city"],
                source_file=c["source_file"],
                chunk_id=c["chunk_id"],
                text=c["text"],
                score=c["score"],
            )
        )

    return QAResponse(answer=result["answer"], context=context_results)


@app.post("/agent", response_model=AgentResponse)
def agent_endpoint(req: AgentRequest) -> AgentResponse:
    agent = get_travel_agent()
    response = agent.run(req.question)
    return AgentResponse(answer=response)
