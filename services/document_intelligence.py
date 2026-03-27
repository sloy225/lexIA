from __future__ import annotations

import logging
from dataclasses import dataclass, field

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, DocumentAnalysisFeature
from azure.core.credentials import AzureKeyCredential

from config.settings import settings
from utils.text_utils import clean_extracted_text

logger = logging.getLogger(__name__)


@dataclass
class ExtractedDocument:
    full_text: str
    pages: list[dict] = field(default_factory=list)   # [{page_number, content}]
    tables: list[dict] = field(default_factory=list)  # extracted table data
    key_value_pairs: dict = field(default_factory=dict)
    page_count: int = 0


class DocumentIntelligenceService:
    def __init__(self) -> None:
        self._client: DocumentIntelligenceClient | None = None

    @property
    def client(self) -> DocumentIntelligenceClient:
        if self._client is None:
            self._client = DocumentIntelligenceClient(
                endpoint=settings.AZURE_DOC_INTELLIGENCE_ENDPOINT,
                credential=AzureKeyCredential(settings.AZURE_DOC_INTELLIGENCE_API_KEY),
            )
        return self._client

    def extract_from_bytes(self, file_bytes: bytes, filename: str) -> ExtractedDocument:
        """Extract text and structure from a document using the prebuilt-layout model."""
        logger.info("Extracting document: %s (%d bytes)", filename, len(file_bytes))

        poller = self.client.begin_analyze_document(
            "prebuilt-layout",
            analyze_request=AnalyzeDocumentRequest(bytes_source=file_bytes),
            features=[DocumentAnalysisFeature.KEY_VALUE_PAIRS],
            output_content_format="markdown",
        )
        result = poller.result()

        pages: list[dict] = []
        tables: list[dict] = []

        # Per-page text
        if result.pages:
            for page in result.pages:
                page_lines = []
                if page.lines:
                    for line in page.lines:
                        page_lines.append(line.content)
                pages.append(
                    {
                        "page_number": page.page_number,
                        "content": "\n".join(page_lines),
                    }
                )

        # Tables
        if result.tables:
            for table in result.tables:
                rows: dict[int, dict[int, str]] = {}
                for cell in table.cells:
                    rows.setdefault(cell.row_index, {})[cell.column_index] = cell.content
                table_data = [
                    [rows[r].get(c, "") for c in range(table.column_count)]
                    for r in range(table.row_count)
                ]
                tables.append({"row_count": table.row_count, "data": table_data})

        # Key-value pairs
        kv_pairs: dict[str, str] = {}
        if result.key_value_pairs:
            for pair in result.key_value_pairs:
                if pair.key and pair.value:
                    kv_pairs[pair.key.content] = pair.value.content

        # Full text: use markdown content if available, otherwise reconstruct
        full_text = result.content if result.content else "\n\n".join(p["content"] for p in pages)
        full_text = clean_extracted_text(full_text)

        return ExtractedDocument(
            full_text=full_text,
            pages=pages,
            tables=tables,
            key_value_pairs=kv_pairs,
            page_count=len(pages),
        )

    def extract_from_url(self, blob_url: str) -> ExtractedDocument:
        """Extract document from an Azure Blob URL."""
        logger.info("Extracting document from URL: %s", blob_url)

        poller = self.client.begin_analyze_document(
            "prebuilt-layout",
            analyze_request=AnalyzeDocumentRequest(url_source=blob_url),
            features=[DocumentAnalysisFeature.KEY_VALUE_PAIRS],
            output_content_format="markdown",
        )
        result = poller.result()

        pages: list[dict] = []
        if result.pages:
            for page in result.pages:
                page_lines = [line.content for line in (page.lines or [])]
                pages.append(
                    {"page_number": page.page_number, "content": "\n".join(page_lines)}
                )

        full_text = result.content if result.content else "\n\n".join(p["content"] for p in pages)
        full_text = clean_extracted_text(full_text)

        return ExtractedDocument(
            full_text=full_text,
            pages=pages,
            page_count=len(pages),
        )


document_intelligence_service = DocumentIntelligenceService()
