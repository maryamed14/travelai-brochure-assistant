from __future__ import annotations

from typing import List, Dict

from langchain_openai import ChatOpenAI

from travelai.nlp import BrochureRetriever, RetrievedChunk
from travelai.config import BROCHURES_JSONL


class BrochureQAPipeline:
    """
    RAG-style QA pipeline:
    - Uses existing BrochureRetriever (TF-IDF or whatever you have now).
    - Adds:
        * City-aware filtering
        * Simple reranking on top of existing similarity scores
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.retriever = BrochureRetriever(BROCHURES_JSONL)
        self.retriever.load()
        self.llm = ChatOpenAI(model=model_name, temperature=0.2)

    # ---------- Filtering helpers ----------

    def _detect_city_from_question(self, question: str) -> str | None:
        """
        Heuristic: detect explicit city mentions in the question text.
        This uses the same 'city' names you have in your dataset
        (e.g. 'New York Brochure', 'London Brochure', etc.).
        """
        q = question.lower()
        mapping: Dict[str, str] = {
            "new york": "New York Brochure",
            "london": "London Brochure",
            "las vegas": "Las Vegas Brochure",
            "dubai": "Dubai Brochure",
            "san francisco": "San Francisco Brochure",
        }
        for alias, city_name in mapping.items():
            if alias in q:
                return city_name
        return None

    def _filter_by_city(self, question: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """
        Prefer chunks from:
        - the city explicitly mentioned in the question, if detected; otherwise
        - the city of the top-scoring chunk.
        """
        if not chunks:
            return []

        # 1) Try to detect from question text
        city_from_question = self._detect_city_from_question(question)
        if city_from_question:
            filtered = [c for c in chunks if c.city == city_from_question]
            if filtered:
                return filtered

        # 2) Fallback to the city of the top chunk
        main_city = chunks[0].city
        filtered = [c for c in chunks if c.city == main_city]
        return filtered or chunks

    # ---------- Reranking helper ----------

    def _rerank(self, question: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """
        Simple reranker on top of the retriever's score:
        combine existing score with a small bonus for token overlap.
        Does not change the underlying similarity model.
        """
        q_terms = set(question.lower().split())

        def combined_score(c: RetrievedChunk) -> float:
            text_terms = set(c.text.lower().split())
            overlap = len(q_terms & text_terms)
            # Use the original similarity score + a small overlap bonus
            return c.score + 0.1 * overlap

        return sorted(chunks, key=combined_score, reverse=True)

    # ---------- Main retrieval used by QA ----------

    def retrieve(self, question: str, k: int = 5) -> List[RetrievedChunk]:
        """
        1) Ask the existing retriever for more candidates (e.g. 4 * k).
        2) Filter by city.
        3) Rerank.
        4) Return final top-k.
        """
        initial_k = max(k * 4, 10)
        # This uses your current similarity model (e.g. TF-IDF)
        candidates = self.retriever.search(question, k=initial_k)

        # City filtering
        filtered = self._filter_by_city(question, candidates)

        # Rerank
        reranked = self._rerank(question, filtered)

        # Final top-k
        return reranked[:k]

    # ---------- LLM answering ----------

    def answer(self, question: str, k: int = 5) -> dict:
        chunks = self.retrieve(question, k=k)

        if not chunks:
            context_text = "No context."
        else:
            context_blocks = []
            for idx, c in enumerate(chunks):
                block = (
                    f"[{idx+1}] City: {c.city} | Source: {c.source_file} | Chunk ID: {c.chunk_id}\n"
                    f"{c.text}"
                )
                context_blocks.append(block)
            context_text = "\n\n".join(context_blocks)

        prompt = (
            "You are an AI travel assistant answering questions using ONLY the brochure excerpts given below.\n\n"
            "### Brochure Excerpts (already filtered and reranked)\n"
            f"{context_text}\n\n"
            "### Question\n"
            f"{question}\n\n"
            "### Instructions\n"
            "- Use only the excerpts that clearly answer the question.\n"
            "- Do not mix details from different cities unless the question explicitly asks to compare cities.\n"
            "- If the brochures do not contain the answer, say: 'The brochures do not contain information to answer this question.'\n"
            "- Keep the answer short (2–4 sentences).\n\n"
            "### Final Answer:\n"
        )

        response = self.llm.invoke(prompt)

        return {
            "answer": response.content,
            "context": [
                {
                    "city": c.city,
                    "source_file": c.source_file,
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "score": c.score,
                }
                for c in chunks
            ],
        }
