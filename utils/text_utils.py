import re
import tiktoken
from config.settings import settings


def get_encoder() -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model("gpt-4o")
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    encoder = get_encoder()
    return len(encoder.encode(text))


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> list[str]:
    """Split text into overlapping chunks based on token count."""
    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = overlap or settings.CHUNK_OVERLAP

    encoder = get_encoder()
    tokens = encoder.encode(text)
    chunks = []

    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = encoder.decode(chunk_tokens)
        chunks.append(chunk_text.strip())
        if end == len(tokens):
            break
        start += chunk_size - overlap

    return [c for c in chunks if c]


def chunk_by_sections(text: str, chunk_size: int = None, overlap: int = None) -> list[dict]:
    """
    Chunk text while preserving section context.
    Returns list of dicts with 'content' and 'section_title'.
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE

    # Detect section headers: numbered (1. / 1.1 / Article 1) or ALL CAPS lines
    section_pattern = re.compile(
        r"^(?:(?:Article|ARTICLE|Clause|CLAUSE|Section|SECTION)\s+\d[\d.]*"
        r"|\d{1,2}(?:\.\d{1,2})*\.?\s+[A-ZÀÂÇÉÈÊËÎÏÔÛÙÜŸŒÆ]"
        r"|[A-ZÀÂÇÉÈÊËÎÏÔÛÙÜŸŒÆ][A-ZÀÂÇÉÈÊËÎÏÔÛÙÜŸŒÆ\s]{4,})",
        re.MULTILINE,
    )

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    sections: list[dict] = []
    current_section = "Introduction"
    current_paragraphs: list[str] = []

    for para in paragraphs:
        if section_pattern.match(para):
            if current_paragraphs:
                sections.append(
                    {"section_title": current_section, "content": "\n\n".join(current_paragraphs)}
                )
            current_section = para[:120]
            current_paragraphs = []
        else:
            current_paragraphs.append(para)

    if current_paragraphs:
        sections.append(
            {"section_title": current_section, "content": "\n\n".join(current_paragraphs)}
        )

    # Sub-chunk sections that exceed chunk_size
    result: list[dict] = []
    for section in sections:
        if count_tokens(section["content"]) <= chunk_size:
            result.append(section)
        else:
            sub_chunks = chunk_text(section["content"], chunk_size=chunk_size, overlap=overlap)
            for i, sub in enumerate(sub_chunks):
                result.append(
                    {
                        "section_title": section["section_title"],
                        "content": sub,
                        "sub_chunk_index": i,
                    }
                )

    return result


def clean_extracted_text(text: str) -> str:
    """Remove artifacts from OCR / Document Intelligence extraction."""
    # Collapse runs of whitespace (but preserve paragraph breaks)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove page number lines like "- 3 -" or "Page 3"
    text = re.sub(r"(?m)^[-–—]?\s*[Pp]age\s*\d+\s*[-–—]?\s*$", "", text)
    text = re.sub(r"(?m)^\s*[-–—]\s*\d+\s*[-–—]\s*$", "", text)
    return text.strip()


def truncate_text(text: str, max_tokens: int) -> str:
    """Truncate text to a maximum number of tokens."""
    encoder = get_encoder()
    tokens = encoder.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoder.decode(tokens[:max_tokens])
