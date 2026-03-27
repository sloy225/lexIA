import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Azure AI Foundry — project OpenAI-compatible endpoint
    AZURE_AI_FOUNDRY_ENDPOINT: str = os.getenv("AZURE_AI_FOUNDRY_ENDPOINT", "")
    AZURE_AI_FOUNDRY_API_KEY: str = os.getenv("AZURE_AI_FOUNDRY_API_KEY", "")
    AZURE_AI_FOUNDRY_CHAT_DEPLOYMENT: str = os.getenv("AZURE_AI_FOUNDRY_CHAT_DEPLOYMENT", "gpt-4o")
    AZURE_AI_FOUNDRY_EMBEDDING_DEPLOYMENT: str = os.getenv(
        "AZURE_AI_FOUNDRY_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"
    )
    EMBEDDING_DIMENSIONS: int = 3072  # text-embedding-3-large default

    # Azure Document Intelligence
    AZURE_DOC_INTELLIGENCE_ENDPOINT: str = os.getenv("AZURE_DOC_INTELLIGENCE_ENDPOINT", "")
    AZURE_DOC_INTELLIGENCE_API_KEY: str = os.getenv("AZURE_DOC_INTELLIGENCE_API_KEY", "")

    # RAG settings (in-memory)
    CHUNK_SIZE: int = 1000   # tokens per chunk
    CHUNK_OVERLAP: int = 150
    TOP_K_RESULTS: int = 6

    # Supported contract types
    CONTRACT_TYPES: list[str] = [
        "Contrat de prestation de services",
        "NDA / Accord de confidentialité",
        "Contrat de travail",
        "CGV / CGU",
        "Autre",
    ]


settings = Settings()
