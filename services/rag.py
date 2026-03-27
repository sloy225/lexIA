"""
RAG pipeline: index contract chunks into Azure AI Search and query them.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    SearchableField,
    VectorSearch,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery

from config.settings import settings
from services.azure_openai import azure_openai_service
from utils.text_utils import chunk_by_sections

logger = logging.getLogger(__name__)


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


class RAGService:
    _index_created: bool = False

    def __init__(self) -> None:
        credential = AzureKeyCredential(settings.AZURE_SEARCH_API_KEY)
        self._index_client = SearchIndexClient(
            endpoint=settings.AZURE_SEARCH_ENDPOINT,
            credential=credential,
        )
        self._search_client = SearchClient(
            endpoint=settings.AZURE_SEARCH_ENDPOINT,
            index_name=settings.AZURE_SEARCH_INDEX_NAME,
            credential=credential,
        )

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def ensure_index(self) -> None:
        """Create the search index if it does not exist."""
        if RAGService._index_created:
            return

        existing = [idx.name for idx in self._index_client.list_indexes()]
        if settings.AZURE_SEARCH_INDEX_NAME not in existing:
            self._create_index()
            logger.info("Created search index: %s", settings.AZURE_SEARCH_INDEX_NAME)
        else:
            logger.info("Search index already exists: %s", settings.AZURE_SEARCH_INDEX_NAME)

        RAGService._index_created = True

    def _create_index(self) -> None:
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SimpleField(
                name="document_id",
                type=SearchFieldDataType.String,
                filterable=True,
            ),
            SimpleField(
                name="filename",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True,
            ),
            SimpleField(
                name="contract_type",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True,
            ),
            SearchableField(name="content", type=SearchFieldDataType.String, language_analyzer_name="fr.microsoft"),
            SearchableField(name="section_title", type=SearchFieldDataType.String, language_analyzer_name="fr.microsoft"),
            SimpleField(name="chunk_index", type=SearchFieldDataType.Int32),
            SimpleField(name="page_number", type=SearchFieldDataType.Int32),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=settings.EMBEDDING_DIMENSIONS,
                vector_search_profile_name="hnsw-profile",
            ),
        ]

        vector_search = VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="hnsw-algo")],
            profiles=[VectorSearchProfile(name="hnsw-profile", algorithm_configuration_name="hnsw-algo")],
        )

        semantic_search = SemanticSearch(
            configurations=[
                SemanticConfiguration(
                    name="semantic-config",
                    prioritized_fields=SemanticPrioritizedFields(
                        content_fields=[SemanticField(field_name="content")],
                        keywords_fields=[SemanticField(field_name="section_title")],
                    ),
                )
            ]
        )

        index = SearchIndex(
            name=settings.AZURE_SEARCH_INDEX_NAME,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic_search,
        )
        self._index_client.create_index(index)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def index_document(
        self,
        document_id: str,
        filename: str,
        full_text: str,
        contract_type: str = "Autre",
    ) -> IndexStats:
        """Chunk, embed and upload a document to the search index."""
        self.ensure_index()

        sections = chunk_by_sections(full_text)
        if not sections:
            logger.warning("No sections found in document %s", document_id)
            return IndexStats(document_id=document_id, filename=filename, chunks_indexed=0)

        # Embed all chunks in one batch
        texts = [s["content"] for s in sections]
        embeddings = azure_openai_service.embed_batch(texts)

        documents = []
        for i, (section, embedding) in enumerate(zip(sections, embeddings)):
            doc = {
                "id": str(uuid.uuid4()),
                "document_id": document_id,
                "filename": filename,
                "contract_type": contract_type,
                "content": section["content"],
                "section_title": section.get("section_title", ""),
                "chunk_index": i,
                "page_number": section.get("page_number"),
                "embedding": embedding,
            }
            documents.append(doc)

        # Upload in batches of 100 (Search API limit)
        batch_size = 100
        for start in range(0, len(documents), batch_size):
            self._search_client.upload_documents(documents[start : start + batch_size])

        logger.info("Indexed %d chunks for document %s", len(documents), document_id)
        return IndexStats(
            document_id=document_id,
            filename=filename,
            chunks_indexed=len(documents),
        )

    def delete_document(self, document_id: str) -> int:
        """Remove all chunks for a document from the index."""
        results = self._search_client.search(
            search_text="*",
            filter=f"document_id eq '{document_id}'",
            select=["id"],
            top=1000,
        )
        ids = [{"id": r["id"]} for r in results]
        if ids:
            self._search_client.delete_documents(ids)
        logger.info("Deleted %d chunks for document %s", len(ids), document_id)
        return len(ids)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        document_id: str | None = None,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """
        Hybrid search: combine vector similarity with BM25 full-text search.
        Optionally filter by document_id.
        """
        self.ensure_index()
        top_k = top_k or settings.TOP_K_RESULTS

        query_embedding = azure_openai_service.embed(query)
        vector_query = VectorizedQuery(
            vector=query_embedding,
            k_nearest_neighbors=top_k,
            fields="embedding",
        )

        filter_expr = f"document_id eq '{document_id}'" if document_id else None

        raw_results = self._search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            filter=filter_expr,
            select=["id", "document_id", "filename", "content", "section_title", "chunk_index", "page_number"],
            top=top_k,
            query_type="semantic",
            semantic_configuration_name="semantic-config",
        )

        results = []
        for r in raw_results:
            results.append(
                SearchResult(
                    document_id=r["document_id"],
                    filename=r["filename"],
                    content=r["content"],
                    section_title=r.get("section_title", ""),
                    chunk_index=r.get("chunk_index", 0),
                    score=r.get("@search.reranker_score") or r.get("@search.score", 0.0),
                    page_number=r.get("page_number"),
                )
            )
        return results


rag_service = RAGService()
