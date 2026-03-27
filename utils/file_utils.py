import hashlib
import mimetypes
import os
import tempfile
from pathlib import Path

ALLOWED_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
    "text/plain",
}

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt"}
MAX_FILE_SIZE_MB = 50


def validate_file(file_bytes: bytes, filename: str) -> tuple[bool, str]:
    """Return (is_valid, error_message)."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Extension non supportée : {ext}. Formats acceptés : PDF, DOCX, TXT."

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return False, f"Fichier trop volumineux ({size_mb:.1f} Mo). Maximum : {MAX_FILE_SIZE_MB} Mo."

    return True, ""


def compute_file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def save_temp_file(file_bytes: bytes, suffix: str) -> str:
    """Save bytes to a named temp file and return the path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(file_bytes)
    tmp.flush()
    tmp.close()
    return tmp.name


def cleanup_temp_file(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def get_blob_name(document_id: str, filename: str) -> str:
    ext = Path(filename).suffix.lower()
    return f"{document_id}{ext}"


def human_readable_size(size_bytes: int) -> str:
    for unit in ["o", "Ko", "Mo", "Go"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} To"
