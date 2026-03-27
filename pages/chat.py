"""Page 3 – Chat RAG avec le contrat."""
from __future__ import annotations

import streamlit as st

from services.azure_openai import azure_openai_service
from services.rag import rag_service

_SYSTEM_PROMPT = """\
Tu es lexIA, un assistant juridique expert qui aide des équipes non juridiques \
(RH, commerciaux, achats, dirigeants de PME/startups) à comprendre leurs contrats.

Règles :
- Réponds TOUJOURS en français, de façon claire et accessible.
- Base tes réponses UNIQUEMENT sur les extraits du contrat fournis dans le contexte.
- Si l'information n'est pas dans le contrat, dis-le clairement.
- Cite les sections pertinentes quand c'est utile.
- Rappelle régulièrement que tu es un outil d'aide à la compréhension, \
  pas un substitut à un conseil juridique professionnel.
- Sois concis mais complet.
"""

_SUGGESTED_QUESTIONS = [
    "Quelles sont les obligations principales de chaque partie ?",
    "Quelle est la durée du contrat et les conditions de renouvellement ?",
    "Quelles sont les conditions de résiliation ?",
    "Y a-t-il des pénalités ou des clauses de dédit ?",
    "Qui détient la propriété intellectuelle ?",
    "Quelles sont les conditions de confidentialité ?",
    "Quels sont les délais et conditions de paiement ?",
    "Quelle juridiction est compétente en cas de litige ?",
]


def _build_rag_context(query: str, search_results: list) -> str:
    """Build the user message with retrieved context injected."""
    if not search_results:
        return (
            f"Question : {query}\n\n"
            "Note : aucun extrait pertinent n'a été trouvé dans le contrat. "
            "Indique à l'utilisateur que tu ne peux pas répondre sans contexte contractuel."
        )
    parts = []
    for i, r in enumerate(search_results, 1):
        section = f" — {r.section_title}" if r.section_title and r.section_title.strip() else ""
        parts.append(f"[Extrait {i}{section}]\n{r.content}")
    context = "\n\n---\n\n".join(parts)
    return f"Extraits du contrat :\n\n{context}\n\n---\n\nQuestion : {query}"


def _build_api_messages(history: list[dict], rag_user_message: str) -> list[dict]:
    """
    Build the messages list for the API call.
    Uses the original history for context (last 10 turns),
    but replaces the final user turn with the RAG-augmented version.
    History is NEVER modified in-place.
    """
    api_messages = [{"role": "system", "content": _SYSTEM_PROMPT}]

    # Add previous turns (everything before the current user message we just added)
    previous_turns = history[:-1]
    for h in previous_turns[-10:]:
        if h["role"] in ("user", "assistant") and h.get("content"):
            api_messages.append({"role": h["role"], "content": h["content"]})

    # Add current user turn with RAG context — not modifying history
    api_messages.append({"role": "user", "content": rag_user_message})
    return api_messages


def render() -> None:
    st.header("Chat avec votre contrat")

    docs = st.session_state.get("documents", {})
    if not docs:
        st.warning(
            "Aucun contrat importé. Rendez-vous sur la page **Import** pour charger un document."
        )
        return

    # ------------------------------------------------------------------ Sidebar
    with st.sidebar:
        st.markdown("### Contrat actif")
        names = {v["filename"]: k for k, v in docs.items()}
        active_id = st.session_state.get("active_document_id")
        active_name = next(
            (v["filename"] for k, v in docs.items() if k == active_id), None
        )
        default_idx = list(names.keys()).index(active_name) if active_name in names else 0
        selected_name = st.selectbox(
            "Sélectionner",
            list(names.keys()),
            index=default_idx,
            label_visibility="collapsed",
        )
        selected_id = names[selected_name]
        st.session_state["active_document_id"] = selected_id
        doc = docs[selected_id]

        st.caption(f"Type : {doc['contract_type']}")
        st.caption(f"{doc['page_count']} page(s) · {doc['chunks_indexed']} segments")

        st.markdown("---")
        st.markdown("### Questions suggérées")
        for i, q in enumerate(_SUGGESTED_QUESTIONS):
            if st.button(q, key=f"sq_{selected_id}_{i}", use_container_width=True):
                st.session_state["pending_question"] = q
                st.rerun()

        st.markdown("---")
        if st.button("Effacer l'historique", key=f"clear_{selected_id}", use_container_width=True):
            st.session_state[f"chat_history_{selected_id}"] = []
            st.rerun()

    # ------------------------------------------------------------------ Memory check
    if not rag_service.is_indexed(selected_id):
        st.warning(
            "Ce contrat n'est plus en mémoire (le serveur a redémarré). "
            "Veuillez le **réimporter** depuis la page Import."
        )
        return

    # ------------------------------------------------------------------ Chat history
    chat_key = f"chat_history_{selected_id}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []
    history: list[dict] = st.session_state[chat_key]

    # Welcome message when history is empty
    if not history:
        with st.chat_message("assistant"):
            st.markdown(
                f"Bonjour ! Je suis **lexIA**, votre assistant pour le contrat "
                f"**{doc['filename']}**.\n\n"
                "Posez-moi vos questions sur ce document. Je m'appuierai uniquement sur son contenu.\n\n"
                "*Rappel : je suis un outil d'aide à la compréhension, pas un avocat.*"
            )

    # Render existing history
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources utilisées", expanded=False):
                    for src in msg["sources"]:
                        if src.get("section"):
                            st.markdown(f"**{src['section']}**")
                        st.caption(src["excerpt"])

    # ------------------------------------------------------------------ User input
    pending = st.session_state.pop("pending_question", None)
    user_input: str | None = st.chat_input("Posez votre question sur le contrat…") or pending

    if not user_input:
        return

    # Add user message to history with the ORIGINAL text (never overwritten)
    history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # ------------------------------------------------------------------ RAG retrieval
    with st.chat_message("assistant"):
        search_results = []
        try:
            with st.spinner("Recherche dans le contrat…"):
                search_results = rag_service.search(
                    query=user_input,
                    document_id=selected_id,
                )
        except Exception as exc:
            error_msg = f"Erreur lors de la recherche dans le contrat : {exc}"
            st.error(error_msg)
            # Remove the user message we just added so the history stays clean
            history.pop()
            return

        # Build API call messages — history is NOT modified
        rag_user_message = _build_rag_context(user_input, search_results)
        api_messages = _build_api_messages(history, rag_user_message)

        # Stream the response
        full_response = ""
        response_placeholder = st.empty()
        try:
            for chunk in azure_openai_service.chat(
                api_messages, temperature=0.1, max_tokens=1500, stream=True
            ):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)
        except Exception as exc:
            # Persist whatever was generated before the error
            if full_response:
                response_placeholder.markdown(full_response)
            else:
                st.error(f"Erreur lors de la génération : {exc}")
                history.pop()  # remove the user message to keep history clean
                return

        # Show sources
        sources = []
        if search_results:
            with st.expander("Sources utilisées", expanded=False):
                for r in search_results:
                    label = r.section_title.strip() if r.section_title and r.section_title.strip() else "Extrait"
                    st.markdown(f"**{label}**")
                    excerpt = r.content[:300] + ("…" if len(r.content) > 300 else "")
                    st.caption(excerpt)
                    sources.append({"section": label, "excerpt": excerpt})

        # Add assistant response to history with original user text preserved
        history.append({"role": "assistant", "content": full_response, "sources": sources})
