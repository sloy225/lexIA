"""lexIA – Point d'entrée Streamlit."""
import streamlit as st

st.set_page_config(
    page_title="lexIA – Analyse de contrats",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------- CSS
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] { min-width: 260px; }
        .risk-high   { color: #dc3545; font-weight: bold; }
        .risk-medium { color: #fd7e14; font-weight: bold; }
        .risk-low    { color: #198754; font-weight: bold; }
        .stAlert > div { border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------- Navigation
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/scales.png",
        width=64,
    )
    st.title("lexIA")
    st.caption("Analyse de contrats par IA")
    st.divider()

    page = st.radio(
        "Navigation",
        ["Import", "Analyse", "Chat"],
        label_visibility="collapsed",
        format_func=lambda x: {
            "Import": "📄  Importer un contrat",
            "Analyse": "🔍  Analyser",
            "Chat": "💬  Chat avec le contrat",
        }[x],
    )

    st.divider()
    docs = st.session_state.get("documents", {})
    if docs:
        st.caption(f"**{len(docs)}** contrat(s) chargé(s)")
        for d in docs.values():
            st.caption(f"• {d['filename']}")
    else:
        st.caption("Aucun contrat chargé")

    st.divider()
    st.caption(
        "⚠️ lexIA est un outil d'aide à la compréhension. "
        "Il ne remplace pas un avocat ou un juriste."
    )

# ---------------------------------------------------------------------- Page routing
from pages import upload, analyze, chat  # noqa: E402 – after set_page_config

if page == "Import":
    upload.render()
elif page == "Analyse":
    analyze.render()
elif page == "Chat":
    chat.render()
