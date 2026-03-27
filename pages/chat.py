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
    "Quelle est la durée du contrat et comment se renouvelle-t-il ?",
    "Quelles sont les conditions de résiliation ?",
    "Y a-t-il des pénalités ou des clauses de dédit ?",
    "Qui détient la propriété intellectuelle ?",
    "Quelles sont les conditions de confidentialité ?",
    "Quels sont les délais de paiement ?",
    "Quelle juridiction est compétente en cas de litige ?",
]


def _build_rag_prompt(query: str, search_results: list) -> str:
    if not search_results:
        return (
            f"Question de l'utilisateur : {query}\n\n"
            "Aucun extrait pertinent n'a été trouvé dans le contrat. "
            "Indique que tu ne peux pas répondre sans contexte."
        )

    context_parts = []
    for i, r in enumerate(search_results, 1):
        section = f" [{r.section_title}]" if r.section_title else ""
        context_parts.append(f"Extrait {i}{section} :\n{r.content}")

    context = "\n\n---\n\n".join(context_parts)
    return (
        f"Voici les extraits pertinents du contrat :\n\n{context}\n\n"
        f"---\n\nQuestion : {query}"
    )


def _get_active_document() -> dict | None:
    docs = st.session_state.get("documents", {})
    active_id = st.session_state.get("active_document_id")
    if not docs:
        return None
    if active_id and active_id in docs:
        return docs[active_id]
    return next(iter(docs.values()), None)


def render() -> None:
    st.header("Chat avec votre contrat")

    docs = st.session_state.get("documents", {})
    if not docs:
        st.warning(
            "Aucun contrat importé. Rendez-vous sur la page **Import** pour charger un document."
        )
        return

    # ------------------------------------------------------------------ Document selector
    with st.sidebar:
        st.markdown("### Contrat actif")
        names = {v["filename"]: k for k, v in docs.items()}
        active_id = st.session_state.get("active_document_id")
        active_name = next((v["filename"] for k, v in docs.items() if k == active_id), None)
        selected_name = st.selectbox(
            "Sélectionner",
            list(names.keys()),
            index=list(names.keys()).index(active_name) if active_name in names else 0,
            label_visibility="collapsed",
        )
        selected_id = names[selected_name]
        st.session_state["active_document_id"] = selected_id
        doc = docs[selected_id]

        st.caption(f"Type : {doc['contract_type']}")
        st.caption(f"{doc['page_count']} page(s) · {doc['chunks_indexed']} segments")

        st.markdown("---")
        st.markdown("### Questions suggérées")
        short_id = selected_id[:8]
        for i, q in enumerate(_SUGGESTED_QUESTIONS):
            if st.button(q, key=f"sq_{short_id}_{i}", use_container_width=True):
                st.session_state["pending_question"] = q
                st.rerun()

        if st.button("Effacer l'historique", key=f"clear_{short_id}", use_container_width=True):
            chat_key = f"chat_history_{selected_id}"
            st.session_state[chat_key] = []
            st.rerun()

    # ------------------------------------------------------------------ Chat history
    chat_key = f"chat_history_{selected_id}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []

    history: list[dict] = st.session_state[chat_key]

    # Welcome message
    if not history:
        with st.chat_message("assistant"):
            st.markdown(
                f"Bonjour ! Je suis **lexIA**, votre assistant pour analyser le contrat "
                f"**{doc['filename']}**.\n\n"
                "Posez-moi vos questions sur ce document. "
                "Je vous répondrai en m'appuyant uniquement sur son contenu.\n\n"
                "*Rappel : je suis un outil d'aide à la compréhension, pas un avocat.*"
            )

    # Display history
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources utilisées", expanded=False):
                    for src in msg["sources"]:
                        label = f"Section : {src['section']}" if src.get("section") else "Extrait"
                        st.markdown(f"**{label}**")
                        st.caption(src["excerpt"])

    # ------------------------------------------------------------------ Input
    pending = st.session_state.pop("pending_question", None)
    user_input = st.chat_input("Posez votre question sur le contrat…") or pending

    if not user_input:
        return

    # Add user message
    history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # ------------------------------------------------------------------ RAG retrieval
    with st.chat_message("assistant"):
        with st.spinner("Recherche dans le contrat…"):
            try:
                search_results = rag_service.search(
                    query=user_input,
                    document_id=selected_id,
                )
            except Exception as exc:
                st.error(f"Erreur lors de la recherche : {exc}")
                return

        prompt = _build_rag_prompt(user_input, search_results)
        messages = [{"role": "system", "content": _SYSTEM_PROMPT}]

        # Include last 6 turns for context
        for h in history[-6:]:
            if h["role"] in ("user", "assistant") and "content" in h:
                messages.append({"role": h["role"], "content": h["content"]})

        # Replace last user message with the RAG-augmented prompt
        messages[-1]["content"] = prompt

        # Stream the answer
        response_placeholder = st.empty()
        full_response = ""
        try:
            for chunk in azure_openai_service.chat(
                messages, temperature=0.1, max_tokens=1500, stream=True
            ):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)
        except Exception as exc:
            st.error(f"Erreur lors de la génération : {exc}")
            return

        # Sources
        sources = []
        if search_results:
            with st.expander("Sources utilisées", expanded=False):
                for r in search_results:
                    label = r.section_title or "Extrait"
                    st.markdown(f"**{label}**")
                    st.caption(r.content[:300] + ("…" if len(r.content) > 300 else ""))
                    sources.append({"section": r.section_title, "excerpt": r.content[:300]})

    history.append({"role": "assistant", "content": full_response, "sources": sources})
