from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievedChunk:
    city: str
    source_file: str
    chunk_id: int
    text: str
    score: float


class BrochureRetriever:
    """Semantic-ish search over brochure chunks using TF-IDF (no torch needed)."""

    def __init__(self, jsonl_path: Path):
        self.jsonl_path = jsonl_path
        self._texts: List[str] = []
        self._meta: List[Dict[str, Any]] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None

    def load(self) -> None:
        """Load dataset and build TF-IDF matrix."""
        self._texts.clear()
        self._meta.clear()

        with self.jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                self._texts.append(rec["text"])
                self._meta.append(rec)

        if not self._texts:
            raise RuntimeError(f"No records found in {self.jsonl_path}")

        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform(self._texts)

    def search(self, query: str, k: int = 5) -> List[RetrievedChunk]:
        if self._vectorizer is None or self._matrix is None:
            raise RuntimeError("Retriever not loaded. Call .load() first.")

        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix)[0]

        top_indices = scores.argsort()[::-1][:k]

        results: List[RetrievedChunk] = []
        for idx in top_indices:
            meta = self._meta[idx]
            results.append(
                RetrievedChunk(
                    city=meta["city"],
                    source_file=meta["source_file"],
                    chunk_id=meta["chunk_id"],
                    text=self._texts[idx],
                    score=float(scores[idx]),
                )
            )
        return results
