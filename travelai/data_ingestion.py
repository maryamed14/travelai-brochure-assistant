from __future__ import annotations

import json
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import RAW_PDF_DIR, PROCESSED_DIR, BROCHURES_JSONL


def infer_city_name(pdf_path: Path) -> str:
    """Infer city name from file name."""
    stem = pdf_path.stem.lower().replace("_", " ")
    return stem.title()


def load_brochure_documents() -> List[dict]:
    """
    Use LangChain's PyPDFLoader + RecursiveCharacterTextSplitter
    to turn all PDFs into chunked documents with metadata.
    """
    docs_with_meta: List[dict] = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " "],
    )

    for pdf_path in sorted(RAW_PDF_DIR.glob("*.pdf")):
        city = infer_city_name(pdf_path)

        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()  # list[Document]

        # Pages -> smaller chunks
        chunks = splitter.split_documents(pages)

        for idx, doc in enumerate(chunks):
            text = doc.page_content.strip()
            if not text:
                continue

            meta = doc.metadata or {}
            docs_with_meta.append(
                {
                    "city": city,
                    "source_file": pdf_path.name,
                    "chunk_id": idx,
                    "page": meta.get("page", meta.get("page_number")),
                    "text": text,
                }
            )

    return docs_with_meta


def build_brochure_dataset() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    records = load_brochure_documents()

    with BROCHURES_JSONL.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} chunks to {BROCHURES_JSONL}")


if __name__ == "__main__":
    build_brochure_dataset()
