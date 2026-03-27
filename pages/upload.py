"""Page 1 – Upload & ingestion d'un contrat."""
from __future__ import annotations

import uuid
import streamlit as st

from config.settings import settings
from services.document_intelligence import document_intelligence_service
from services.rag import rag_service
from utils.file_utils import compute_file_hash, human_readable_size, validate_file


def render() -> None:
    st.header("Importer un contrat")
    st.write(
        "Importez votre contrat (PDF, DOCX ou TXT). "
        "Le document sera analysé et indexé pour vous permettre de l'interroger."
    )

    # ------------------------------------------------------------------ Upload
    uploaded = st.file_uploader(
        "Sélectionner un fichier",
        type=["pdf", "docx", "doc", "txt"],
        accept_multiple_files=False,
    )

    if not uploaded:
        st.info("Aucun fichier sélectionné. Formats acceptés : PDF, DOCX, TXT (max 50 Mo).")
        return

    file_bytes = uploaded.read()
    is_valid, error_msg = validate_file(file_bytes, uploaded.name)
    if not is_valid:
        st.error(f"Fichier invalide : {error_msg}")
        return

    st.success(
        f"**{uploaded.name}** — {human_readable_size(len(file_bytes))} prêt à l'import."
    )

    contract_type = st.selectbox("Type de contrat", settings.CONTRACT_TYPES)

    # ------------------------------------------------------------------ Action
    col1, col2 = st.columns([1, 3])
    with col1:
        start = st.button("Analyser le contrat", type="primary", use_container_width=True)

    if not start:
        return

    document_id = str(uuid.uuid4())
    file_hash = compute_file_hash(file_bytes)

    with st.status("Traitement en cours…", expanded=True) as status:

        # 1. Document Intelligence — extract text
        st.write("Extraction du texte avec Azure Document Intelligence…")
        try:
            extracted = document_intelligence_service.extract_from_bytes(
                file_bytes, uploaded.name
            )
            st.write(f"Extraction : ✅ ({extracted.page_count} page(s))")
        except Exception as exc:
            status.update(label="Échec de l'extraction", state="error")
            st.error(f"Erreur Document Intelligence : {exc}")
            return

        # 2. RAG indexing — chunk + embed + store in memory
        st.write("Indexation en mémoire (embeddings)…")
        try:
            stats = rag_service.index_document(
                document_id=document_id,
                filename=uploaded.name,
                full_text=extracted.full_text,
                contract_type=contract_type,
            )
            st.write(f"Indexation : ✅ ({stats.chunks_indexed} segments)")
        except Exception as exc:
            status.update(label="Échec de l'indexation", state="error")
            st.error(f"Erreur lors de l'indexation : {exc}")
            return

        status.update(label="Traitement terminé !", state="complete")

    # ------------------------------------------------------------------ Store in session
    doc_meta = {
        "document_id": document_id,
        "filename": uploaded.name,
        "contract_type": contract_type,
        "full_text": extracted.full_text,
        "page_count": extracted.page_count,
        "chunks_indexed": stats.chunks_indexed,
        "file_hash": file_hash,
    }

    if "documents" not in st.session_state:
        st.session_state["documents"] = {}
    st.session_state["documents"][document_id] = doc_meta
    st.session_state["active_document_id"] = document_id

    st.success(
        f"**{uploaded.name}** importé avec succès ! "
        "Rendez-vous sur **Analyse** ou **Chat** pour explorer ce contrat."
    )

    with st.expander("Aperçu du texte extrait", expanded=False):
        st.text(extracted.full_text[:3000] + ("…" if len(extracted.full_text) > 3000 else ""))
