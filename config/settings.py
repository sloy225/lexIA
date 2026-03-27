import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Azure AI Foundry — inference endpoint (replaces bare Azure OpenAI endpoint)
    AZURE_AI_FOUNDRY_ENDPOINT: str = os.getenv("AZURE_AI_FOUNDRY_ENDPOINT", "")
    AZURE_AI_FOUNDRY_API_KEY: str = os.getenv("AZURE_AI_FOUNDRY_API_KEY", "")
    AZURE_AI_FOUNDRY_CHAT_DEPLOYMENT: str = os.getenv("AZURE_AI_FOUNDRY_CHAT_DEPLOYMENT", "gpt-4o")
    AZURE_AI_FOUNDRY_EMBEDDING_DEPLOYMENT: str = os.getenv(
        "AZURE_AI_FOUNDRY_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"
    )
    EMBEDDING_DIMENSIONS: int = 3072  # text-embedding-3-large default

    # Azure AI Search
    AZURE_SEARCH_ENDPOINT: str = os.getenv("AZURE_SEARCH_ENDPOINT", "")
    AZURE_SEARCH_API_KEY: str = os.getenv("AZURE_SEARCH_API_KEY", "")
    AZURE_SEARCH_INDEX_NAME: str = os.getenv("AZURE_SEARCH_INDEX_NAME", "lexia-contracts")

    # Azure Document Intelligence
    AZURE_DOC_INTELLIGENCE_ENDPOINT: str = os.getenv("AZURE_DOC_INTELLIGENCE_ENDPOINT", "")
    AZURE_DOC_INTELLIGENCE_API_KEY: str = os.getenv("AZURE_DOC_INTELLIGENCE_API_KEY", "")

    # Azure Storage
    AZURE_STORAGE_CONNECTION_STRING: str = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
    AZURE_STORAGE_CONTAINER_NAME: str = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "contracts")

    # RAG settings
    CHUNK_SIZE: int = 1000          # tokens per chunk
    CHUNK_OVERLAP: int = 150        # overlap between chunks
    TOP_K_RESULTS: int = 6          # number of chunks to retrieve
    VECTOR_WEIGHT: float = 0.7      # weight for vector search in hybrid mode

    # Supported contract types
    CONTRACT_TYPES: list[str] = [
        "Contrat de prestation de services",
        "NDA / Accord de confidentialité",
        "Contrat de travail",
        "CGV / CGU",
        "Autre",
    ]


settings = Settings()
