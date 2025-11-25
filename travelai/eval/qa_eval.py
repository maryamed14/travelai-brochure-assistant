from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

from travelai.nlp import BrochureRetriever


EVAL_FILE = Path("data/eval/qa_eval_examples.jsonl")


@dataclass
class EvalExample:
    id: int
    question: str
    expected_city: str
    expected_contains: str


@dataclass
class EvalResult:
    example: EvalExample
    predicted_text: str
    predicted_cities: List[str]
    city_hit: bool
    answer_hit: bool


def load_examples(path: Path = EVAL_FILE) -> List[EvalExample]:
    examples: List[EvalExample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data: Dict[str, Any] = json.loads(line)
            examples.append(
                EvalExample(
                    id=data["id"],
                    question=data["question"],
                    expected_city=data["expected_city"],
                    expected_contains=data["expected_contains"],
                )
            )
    return examples


def evaluate_qa(max_k: int = 5) -> List[EvalResult]:
    """
    Offline evaluation of the retrieval step only (no LLM calls).

    Metrics:
    - city_hit: expected city appears among retrieved chunks
    - answer_hit: expected phrase appears in the retrieved text (case-insensitive)
    """
    retriever = BrochureRetriever()
    examples = load_examples()

    results: List[EvalResult] = []

    for ex in examples:
        chunks = retriever.search(ex.question, k=max_k)
        if not chunks:
            results.append(
                EvalResult(
                    example=ex,
                    predicted_text="",
                    predicted_cities=[],
                    city_hit=False,
                    answer_hit=False,
                )
            )
            continue

        predicted_cities = list({c.city for c in chunks})
        city_hit = ex.expected_city in predicted_cities

        combined_text = " ".join(c.text for c in chunks)
        combined_lower = combined_text.lower()
        expected_lower = ex.expected_contains.lower()
        answer_hit = expected_lower in combined_lower

        results.append(
            EvalResult(
                example=ex,
                predicted_text=combined_text,
                predicted_cities=predicted_cities,
                city_hit=city_hit,
                answer_hit=answer_hit,
            )
        )

    return results


def main() -> None:
    results = evaluate_qa()

    total = len(results)
    city_hits = sum(1 for r in results if r.city_hit)
    answer_hits = sum(1 for r in results if r.answer_hit)

    print(f"Total examples: {total}")
    print(f"City hit rate: {city_hits}/{total} = {city_hits/total:.2f}")
    print(f"Answer hit rate (retrieval-based): {answer_hits}/{total} = {answer_hits/total:.2f}")
    print()

    for r in results:
        print(f"--- Example {r.example.id} ---")
        print(f"Q: {r.example.question}")
        print(f"Expected city: {r.example.expected_city}")
        print(f"Predicted cities: {r.predicted_cities}")
        print(f"Expected in retrieved text: {r.example.expected_contains}")
        print(f"Retrieved snippet (truncated): {r.predicted_text[:300]}...")
        print(f"city_hit={r.city_hit}, answer_hit={r.answer_hit}")
        print()


if __name__ == "__main__":
    main()
