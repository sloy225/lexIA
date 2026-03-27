"""
RAG pipeline — in-memory implementation (no Azure AI Search required).

Chunks are stored in process memory keyed by document_id.
Retrieval uses cosine similarity on OpenAI embeddings.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from config.settings import settings
from services.azure_openai import azure_openai_service
from utils.text_utils import chunk_by_sections

logger = logging.getLogger(__name__)

# Module-level store: document_id -> list of chunk dicts (with embeddings)
_CHUNK_STORE: dict[str, list[dict]] = {}


@dataclass
class SearchResult:
    document_id: str
    filename: str
    content: str
    section_title: str
    chunk_index: int
    score: float
    page_number: int | None = None


@dataclass
class IndexStats:
    document_id: str
    filename: str
    chunks_indexed: int


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class RAGService:

    def index_document(
        self,
        document_id: str,
        filename: str,
        full_text: str,
        contract_type: str = "Autre",
    ) -> IndexStats:
        """Chunk, embed, and store document in memory."""
        sections = chunk_by_sections(full_text)
        if not sections:
            logger.warning("No sections found in document %s", document_id)
            return IndexStats(document_id=document_id, filename=filename, chunks_indexed=0)

        texts = [s["content"] for s in sections]
        embeddings = azure_openai_service.embed_batch(texts)

        chunks = []
        for i, (section, embedding) in enumerate(zip(sections, embeddings)):
            chunks.append(
                {
                    "document_id": document_id,
                    "filename": filename,
                    "contract_type": contract_type,
                    "content": section["content"],
                    "section_title": section.get("section_title", ""),
                    "chunk_index": i,
                    "embedding": embedding,
                }
            )

        _CHUNK_STORE[document_id] = chunks
        logger.info("Indexed %d chunks for document %s (in-memory)", len(chunks), document_id)
        return IndexStats(document_id=document_id, filename=filename, chunks_indexed=len(chunks))

    def delete_document(self, document_id: str) -> int:
        chunks = _CHUNK_STORE.pop(document_id, [])
        return len(chunks)

    def search(
        self,
        query: str,
        document_id: str | None = None,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Return the top-k most relevant chunks using cosine similarity."""
        top_k = top_k or settings.TOP_K_RESULTS

        if document_id:
            candidates = _CHUNK_STORE.get(document_id, [])
        else:
            candidates = [c for chunks in _CHUNK_STORE.values() for c in chunks]

        if not candidates:
            logger.warning("No chunks in memory for document_id=%s", document_id)
            return []

        query_embedding = azure_openai_service.embed(query)

        scored = [
            (_cosine_similarity(query_embedding, c["embedding"]), c)
            for c in candidates
        ]
        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            SearchResult(
                document_id=c["document_id"],
                filename=c["filename"],
                content=c["content"],
                section_title=c.get("section_title", ""),
                chunk_index=c.get("chunk_index", 0),
                score=score,
            )
            for score, c in scored[:top_k]
        ]

    def is_indexed(self, document_id: str) -> bool:
        return document_id in _CHUNK_STORE and len(_CHUNK_STORE[document_id]) > 0


rag_service = RAGService()
