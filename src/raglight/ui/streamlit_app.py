import json
import os
from pathlib import Path
import requests
import streamlit as st

API_URL = os.environ.get("RAGLIGHT_API_URL", "http://localhost:8000")
API_TIMEOUT = int(os.environ.get("RAGLIGHT_API_TIMEOUT", 300))

st.set_page_config(
    page_title="RAGLight",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
#MainMenu, footer, header { visibility: hidden; }

section[data-testid="stSidebar"] {
    background-color: #0f172a;
    border-right: 1px solid #1e293b;
}
section[data-testid="stSidebar"] * { color: #cbd5e1; }
section[data-testid="stSidebar"] h1 { color: #f1f5f9 !important; font-size: 1.3rem !important; }
section[data-testid="stSidebar"] .stButton button {
    background: #1e293b;
    color: #e2e8f0;
    border: 1px solid #334155;
    border-radius: 8px;
}
section[data-testid="stSidebar"] .stButton button:hover {
    background: #334155;
    border-color: #475569;
}
section[data-testid="stSidebar"] hr { border-color: #1e293b; }

.source-pill {
    display: inline-block;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    margin: 3px 2px;
    color: #94a3b8;
}
</style>
""",
    unsafe_allow_html=True,
)


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(ttl=10)
def api_health() -> bool:
    try:
        return requests.get(f"{API_URL}/health", timeout=3).status_code == 200
    except Exception:
        return False


LLM_PROVIDERS = [
    "Ollama",
    "OpenAI",
    "LmStudio",
    "Mistral",
    "MiniMax",
    "GoogleGemini",
    "AWSBedrock",
]
# Providers that require an API base URL
PROVIDERS_WITH_API_BASE = {"Ollama", "OpenAI", "LmStudio", "Mistral", "MiniMax"}


@st.cache_data(ttl=60)
def fetch_llm_config() -> dict:
    try:
        r = requests.get(f"{API_URL}/config", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {"llm_provider": "Ollama", "llm_model": "", "llm_api_base": ""}


@st.cache_data(ttl=30)
def fetch_collections() -> list:
    try:
        r = requests.get(f"{API_URL}/collections", timeout=5)
        r.raise_for_status()
        return r.json().get("collections", [])
    except Exception:
        return []


def stream_response(prompt: str):
    """Consume the SSE /generate/stream endpoint and yield text chunks."""
    with requests.post(
        f"{API_URL}/generate/stream",
        json={"question": prompt},
        stream=True,
        timeout=API_TIMEOUT,
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    payload = json.loads(data)
                    if "error" in payload:
                        yield f"⚠️ {payload['error']}"
                        break
                    chunk = payload.get("chunk", "")
                    if chunk:
                        yield chunk
                except Exception:
                    pass


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []
if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0
if "generating" not in st.session_state:
    st.session_state.generating = False
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# ⚡ RAGLight")

    healthy = api_health()
    st.caption(
        f"{'🟢' if healthy else '🔴'} {'Connected' if healthy else 'Unreachable'} · {API_URL}"
    )

    st.divider()

    st.markdown("**📥 Import knowledge**")

    with st.expander("Upload files", expanded=True):
        uploaded = st.file_uploader(
            "files",
            accept_multiple_files=True,
            label_visibility="collapsed",
            key=f"uploader_{st.session_state.upload_key}",
        )
        if st.button("Ingest files", use_container_width=True, disabled=not uploaded):
            with st.spinner("Ingesting…"):
                files = [("files", (f.name, f.getvalue())) for f in uploaded]
                try:
                    r = requests.post(
                        f"{API_URL}/ingest/upload", files=files, timeout=300
                    )
                    r.raise_for_status()
                    st.session_state.ingested_files.extend([f.name for f in uploaded])
                    st.session_state.upload_key += 1
                    fetch_collections.clear()
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

    if st.session_state.ingested_files:
        st.markdown("**Ingested this session**")
        for name in st.session_state.ingested_files:
            st.caption(f"✓ {name}")

    with st.expander("Add a directory", expanded=False):
        st.caption("Path on the machine running the server")
        dir_path = st.text_input(
            "dir", placeholder="/path/to/my/docs", label_visibility="collapsed"
        )
        if st.button(
            "Ingest directory", use_container_width=True, disabled=not dir_path
        ):
            with st.spinner("Ingesting…"):
                try:
                    r = requests.post(
                        f"{API_URL}/ingest", json={"data_path": dir_path}, timeout=300
                    )
                    r.raise_for_status()
                    st.session_state.ingested_files.append(f"📁 {Path(dir_path).name}")
                    fetch_collections.clear()
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

    st.divider()

    if st.button("🗑 Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    with st.expander("⚙️ Model settings", expanded=False):
        current_cfg = fetch_llm_config()
        provider = st.selectbox(
            "Provider",
            LLM_PROVIDERS,
            index=(
                LLM_PROVIDERS.index(current_cfg.get("llm_provider", "Ollama"))
                if current_cfg.get("llm_provider") in LLM_PROVIDERS
                else 0
            ),
        )
        model = st.text_input("Model", value=current_cfg.get("llm_model", ""))
        if provider in PROVIDERS_WITH_API_BASE:
            api_base = st.text_input(
                "API base URL", value=current_cfg.get("llm_api_base") or ""
            )
        else:
            api_base = None
        if st.button("Apply", use_container_width=True, type="primary"):
            with st.spinner("Switching model…"):
                try:
                    r = requests.post(
                        f"{API_URL}/config",
                        json={
                            "llm_provider": provider,
                            "llm_model": model,
                            "llm_api_base": api_base or None,
                        },
                        timeout=30,
                    )
                    r.raise_for_status()
                    fetch_llm_config.clear()
                    st.success(f"{provider} / {model}")
                except Exception as e:
                    st.error(str(e))


# ── Chat ──────────────────────────────────────────────────────────────────────
if not st.session_state.messages and not st.session_state.generating:
    st.markdown(
        """
    <div style="text-align:center; padding-top:12vh; color:#6b7280">
        <div style="font-size:2.5rem">⚡</div>
        <div style="font-size:1.8rem; font-weight:700; color:#111827; margin:0.5rem 0">
            What can I help you with?
        </div>
        <div style="font-size:1rem">
            Ask anything about your documents,<br>or import knowledge from the sidebar.
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# Input is disabled while a response is being generated
prompt_input = st.chat_input(
    "Message RAGLight…",
    disabled=not healthy or st.session_state.generating,
)

if prompt_input:
    st.session_state.pending_prompt = prompt_input
    st.session_state.generating = True
    st.rerun()

if st.session_state.generating and st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            full_response = st.write_stream(stream_response(prompt))
        except Exception as e:
            full_response = f"⚠️ {e}"
            st.markdown(full_response)

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state.generating = False
    st.session_state.pending_prompt = None
    st.rerun()
