from __future__ import annotations

from typing import Any

from langchain.tools import BaseTool
from pydantic.v1 import PrivateAttr

from travelai.qa import BrochureQAPipeline


class BrochureSearchTool(BaseTool):
    """
    Tool that searches the travel brochures for relevant chunks of text.
    Uses the same retrieval logic as the QA pipeline (filtering + reranking).
    """

    name: str = "brochure_search"
    description: str = (
        "Search within the travel brochures for relevant information. "
        "Use this when you need concrete details about cities, hotels, "
        "attractions, or descriptions from the brochures."
    )

    _pipeline: BrochureQAPipeline = PrivateAttr()

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._pipeline = BrochureQAPipeline(model_name="gpt-4o-mini")

    def _run(self, query: str) -> str:
        chunks = self._pipeline.retrieve(question=query, k=5)
        if not chunks:
            return "No relevant brochure text found."

        blocks = []
        for idx, c in enumerate(chunks, start=1):
            block = (
                f"[{idx}] City: {c.city} | Source: {c.source_file} | Chunk ID: {c.chunk_id} | Score: {c.score:.3f}\n"
                f"{c.text}"
            )
            blocks.append(block)

        return "\n\n".join(blocks)

    async def _arun(self, query: str) -> Any:
        raise NotImplementedError("Async not implemented for BrochureSearchTool")
