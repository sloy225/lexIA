"""Page 2 – Analyse complète d'un contrat (résumé + risques + anonymisation)."""
from __future__ import annotations

import streamlit as st

from services.anonymizer import anonymizer_service
from services.risk_detector import RiskLevel, risk_detector_service
from services.summarizer import summarizer_service


_RISK_COLORS = {
    RiskLevel.HIGH: "🔴",
    RiskLevel.MEDIUM: "🟠",
    RiskLevel.LOW: "🟢",
}

_RISK_BADGE = {
    RiskLevel.HIGH: "danger",
    RiskLevel.MEDIUM: "warning",
    RiskLevel.LOW: "success",
}


def _get_active_document() -> dict | None:
    docs = st.session_state.get("documents", {})
    active_id = st.session_state.get("active_document_id")
    if not docs:
        return None
    if active_id and active_id in docs:
        return docs[active_id]
    return next(iter(docs.values()))


def _render_document_selector(docs: dict) -> dict:
    names = {v["filename"]: k for k, v in docs.items()}
    active_id = st.session_state.get("active_document_id")
    active_name = next((v["filename"] for k, v in docs.items() if k == active_id), None)
    selected_name = st.selectbox(
        "Contrat actif",
        list(names.keys()),
        index=list(names.keys()).index(active_name) if active_name in names else 0,
    )
    selected_id = names[selected_name]
    st.session_state["active_document_id"] = selected_id
    return docs[selected_id]


def _render_summary_tab(doc: dict) -> None:
    st.subheader("Résumé du contrat")

    cache_key = f"summary_{doc['document_id']}"

    if cache_key not in st.session_state:
        if st.button("Générer le résumé", type="primary"):
            with st.spinner("Analyse en cours…"):
                summary = summarizer_service.summarize(doc["full_text"])
                st.session_state[cache_key] = summary
    else:
        summary = st.session_state[cache_key]

        # Short summary card
        st.info(f"**En résumé :** {summary.short}")

        # Detected contract type
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("### Analyse détaillée")
        with col2:
            st.caption(f"Type détecté : **{summary.contract_type_detected}**")

        st.markdown(summary.detailed)

        if st.button("Régénérer le résumé"):
            del st.session_state[cache_key]
            st.rerun()

    if cache_key not in st.session_state:
        st.info("Cliquez sur **Générer le résumé** pour analyser ce contrat.")


def _render_risks_tab(doc: dict) -> None:
    st.subheader("Détection des risques")

    cache_key = f"risks_{doc['document_id']}"

    if cache_key not in st.session_state:
        if st.button("Analyser les risques", type="primary"):
            with st.spinner("Analyse des risques en cours…"):
                report = risk_detector_service.detect(doc["full_text"])
                st.session_state[cache_key] = report
    else:
        report = st.session_state[cache_key]

        # Overall risk badge
        icon = _RISK_COLORS.get(report.overall_risk, "⚪")
        st.markdown(f"### Niveau de risque global : {icon} **{report.overall_risk.value}**")
        st.write(report.overall_comment)

        # Stats
        col1, col2, col3 = st.columns(3)
        col1.metric("🔴 Risques élevés", report.high_count)
        col2.metric("🟠 Risques moyens", report.medium_count)
        col3.metric("🟢 Risques faibles", report.low_count)

        if not report.risks:
            st.success("Aucune clause à risque détectée.")
        else:
            st.markdown("---")
            st.markdown("### Clauses identifiées")
            for risk in sorted(report.risks, key=lambda r: [RiskLevel.HIGH, RiskLevel.MEDIUM, RiskLevel.LOW].index(r.risk_level)):
                icon = _RISK_COLORS.get(risk.risk_level, "⚪")
                with st.expander(f"{icon} {risk.title} — {risk.risk_level.value}", expanded=(risk.risk_level == RiskLevel.HIGH)):
                    if risk.excerpt:
                        st.markdown(f"> *{risk.excerpt}*")
                    st.markdown(f"**Pourquoi c'est risqué :** {risk.explanation}")
                    st.markdown(f"**Recommandation :** {risk.recommendation}")

        if st.button("Réanalyser les risques"):
            del st.session_state[cache_key]
            st.rerun()

    if cache_key not in st.session_state:
        st.info("Cliquez sur **Analyser les risques** pour détecter les clauses problématiques.")
        st.caption(
            "L'analyse identifie les clauses pénales excessives, limitations de responsabilité, "
            "non-concurrence abusive, propriété intellectuelle, résiliation défavorable, etc."
        )


def _render_anonymize_tab(doc: dict) -> None:
    st.subheader("Anonymisation des données personnelles")

    st.write(
        "Supprimez les données personnelles (noms, emails, téléphones, SIRET…) "
        "avant de partager ce contrat."
    )

    mode = st.radio(
        "Mode d'anonymisation",
        ["Rapide (regex)", "Complète (IA + regex)"],
        horizontal=True,
        help="Le mode rapide utilise des expressions régulières. "
             "Le mode complet ajoute une détection par IA des noms et adresses.",
    )

    cache_key = f"anon_{doc['document_id']}_{mode}"

    if cache_key not in st.session_state:
        if st.button("Anonymiser", type="primary"):
            with st.spinner("Anonymisation en cours…"):
                if "Rapide" in mode:
                    anonymized, count = anonymizer_service.anonymize_regex(doc["full_text"])
                    result = type("R", (), {"anonymized_text": anonymized, "total_replaced": count, "entities_found": []})()
                else:
                    result = anonymizer_service.anonymize_full(doc["full_text"])
                st.session_state[cache_key] = result
    else:
        result = st.session_state[cache_key]

        st.success(f"**{result.total_replaced}** éléments anonymisés.")

        if result.entities_found:
            with st.expander(f"Données détectées ({len(result.entities_found)})", expanded=False):
                for e in result.entities_found:
                    st.markdown(f"- **{e.category}** : `{e.value}` → `{e.replacement}`")

        st.text_area(
            "Texte anonymisé",
            value=result.anonymized_text[:5000],
            height=400,
            disabled=True,
        )

        st.download_button(
            "Télécharger le texte anonymisé",
            data=result.anonymized_text.encode("utf-8"),
            file_name=f"{doc['filename'].rsplit('.', 1)[0]}_anonymise.txt",
            mime="text/plain",
        )

        if st.button("Réanonymiser"):
            del st.session_state[cache_key]
            st.rerun()

    if cache_key not in st.session_state:
        st.info("Cliquez sur **Anonymiser** pour traiter le document.")


def render() -> None:
    st.header("Analyse du contrat")

    docs = st.session_state.get("documents", {})
    if not docs:
        st.warning(
            "Aucun contrat importé. Rendez-vous sur la page **Import** pour charger un document."
        )
        return

    doc = _render_document_selector(docs)

    st.caption(
        f"Contrat : **{doc['filename']}** | Type : {doc['contract_type']} | "
        f"{doc['page_count']} page(s) | {doc['chunks_indexed']} segments indexés"
    )
    st.divider()

    tab_summary, tab_risks, tab_anon = st.tabs(["Résumé", "Risques", "Anonymisation"])

    with tab_summary:
        _render_summary_tab(doc)

    with tab_risks:
        _render_risks_tab(doc)

    with tab_anon:
        _render_anonymize_tab(doc)
