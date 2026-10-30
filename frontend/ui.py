import html
import os
import re
from pathlib import Path
from urllib.parse import quote

import requests
import streamlit as st
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Assistant", page_icon="\U0001F4DA", layout="wide")

# Colors are translucent or inherited so the same styles work in light and dark mode
st.markdown(
    """
<style>
:root { --accent: #2563EB; --line: rgba(128, 128, 128, 0.22); --soft: rgba(37, 99, 235, 0.08); }

/* Streamlit's fixed top bar (the ⋮ menu) is ~3.75rem tall; start the content below it */
[data-testid="stMainBlockContainer"] { max-width: 920px; padding-top: 4.5rem; }
[data-testid="stSidebarContent"] { padding-top: 0.5rem; }

.brand { display: flex; align-items: center; gap: 0.7rem; margin: 0 0 1.4rem; }
.brand svg { flex: none; }
.brand-name { font-size: 1.15rem; font-weight: 700; line-height: 1.2; }
.brand-tag { font-size: 0.78rem; opacity: 0.65; }

.section-label { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;
  opacity: 0.6; margin: 1.2rem 0 0.5rem; }

.doc { display: flex; align-items: center; gap: 0.6rem; min-width: 0; }
.doc-text { min-width: 0; }
.doc-name { font-size: 0.85rem; font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.doc-meta { font-size: 0.72rem; opacity: 0.6; }
.badge { flex: none; width: 30px; height: 30px; border-radius: 7px; color: #fff; font-size: 0.55rem; font-weight: 700;
  display: flex; align-items: center; justify-content: center; letter-spacing: 0.02em; }
.badge.pdf { background: #EF4444; } .badge.docx { background: #2563EB; } .badge.txt { background: #64748B; }

.status { display: flex; align-items: center; gap: 0.55rem; font-size: 0.78rem; border-top: 1px solid var(--line);
  padding-top: 0.9rem; margin-top: 1.4rem; }
.status b { font-weight: 600; }
.dot { width: 8px; height: 8px; border-radius: 50%; background: #22C55E; flex: none; }
.dot.off { background: #EF4444; }

.page-head { display: flex; align-items: center; gap: 0.7rem; padding-bottom: 0.9rem;
  border-bottom: 1px solid var(--line); margin-bottom: 1.2rem; }
.page-icon { width: 36px; height: 36px; border-radius: 9px; background: var(--soft); color: var(--accent);
  display: flex; align-items: center; justify-content: center; }
.page-title { font-size: 1.1rem; font-weight: 700; line-height: 1.2; }
.page-sub { font-size: 0.78rem; opacity: 0.6; }

.empty { text-align: center; padding: 4rem 1rem 2rem; }
.empty .page-icon { width: 52px; height: 52px; margin: 0 auto 1rem; border-radius: 14px; }
.empty h3 { margin: 0 0 0.3rem; font-size: 1.25rem; }
.empty p { opacity: 0.65; font-size: 0.9rem; margin: 0; }

/* User questions: right-aligned blue bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  flex-direction: row-reverse; background: transparent; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
  flex: 0 1 auto; max-width: 75%; background: var(--soft); border: 1px solid rgba(37, 99, 235, 0.25);
  border-radius: 14px 14px 4px 14px; padding: 0.55rem 0.9rem; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p { margin: 0; }

/* Answers: bordered card with a blue assistant avatar */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) { background: transparent; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {
  border: 1px solid var(--line); border-radius: 4px 14px 14px 14px; padding: 0.9rem 1.1rem; }
[data-testid="stChatMessageAvatarAssistant"] { background: var(--accent); color: #fff; border-radius: 9px; }
[data-testid="stChatMessageAvatarUser"] { border-radius: 50%; }

.sources-label { font-size: 0.8rem; font-weight: 600; margin: 0.9rem 0 0.4rem; opacity: 0.85; }
.sources { border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
.source { display: flex; align-items: center; gap: 0.65rem; padding: 0.55rem 0.75rem; }
.source + .source { border-top: 1px solid var(--line); }

/* Input: card on the page background, same width as the chat, solid blue send button */
[data-testid="stBottomBlockContainer"] { max-width: 920px; padding-bottom: 1.5rem; }
/* Streamlit nests a bordered text field inside a grey box; the outer box becomes the card
   and the inner field loses its own border, so there is a single outline */
[data-testid="stChatInput"] {
  background: transparent !important; border: 1px solid var(--line); border-radius: 12px !important;
  box-shadow: 0 2px 10px rgba(15, 23, 42, 0.06); overflow: hidden; }
[data-testid="stChatInput"]:focus-within { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15); }
[data-testid="stChatInput"] [data-baseweb="textarea"] {
  border: none !important; background: transparent !important; width: 100% !important; }
[data-testid="stChatInputTextArea"] {
  font-size: 0.92rem; padding: 0.85rem 3.5rem 0.85rem 1rem !important; }
[data-testid="stChatInputSubmitButton"],
[data-testid="stChatInputSubmitButton"]:hover,
[data-testid="stChatInputSubmitButton"]:disabled {
  background: var(--accent) !important; color: #fff !important; border-radius: 8px !important;
  width: 34px; height: 34px; padding: 0; margin: 0 8px 8px 0; }
[data-testid="stChatInputSubmitButton"]:disabled { opacity: 0.45; }
</style>
""",
    unsafe_allow_html=True,
)

CHAT_ICON = (
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>'
)
LOGO = (
    '<svg width="34" height="34" viewBox="0 0 24 24"><path fill="#2563EB" d="M12 1l2.6 7.4L22 11l-7.4 2.6L12 21'
    'l-2.6-7.4L2 11l7.4-2.6z"/><path fill="#60A5FA" d="M19 15l.9 2.1L22 18l-2.1.9L19 21l-.9-2.1L16 18l2.1-.9z"/></svg>'
)

if "messages" not in st.session_state:
    st.session_state.messages = []


def badge(filename: str) -> str:
    ext = Path(filename).suffix.lower().lstrip(".")
    ext = ext if ext in ("pdf", "docx", "txt") else "txt"
    return f'<div class="badge {ext}">{ext.upper()}</div>'


def doc_row(name: str, meta: str) -> str:
    return (
        f'<div class="doc">{badge(name)}<div class="doc-text">'
        f'<div class="doc-name" title="{html.escape(name)}">{html.escape(name)}</div>'
        f'<div class="doc-meta">{html.escape(meta)}</div></div></div>'
    )


def render_sources(sources: list[str]) -> None:
    # The backend labels sources "file.pdf p. 3" or "file.pdf pp. 4-6"
    rows = []
    for label in sources:
        match = re.match(r"^(.*?)\s+(pp?)\.\s*(\S+)$", label)
        name, pages = (match.group(1), f"{'Pages' if match.group(2) == 'pp' else 'Page'} {match.group(3)}") if match else (label, "")
        rows.append(f'<div class="source">{doc_row(name, pages)}</div>')
    st.markdown(
        f'<div class="sources-label">Sources ({len(sources)})</div><div class="sources">{"".join(rows)}</div>',
        unsafe_allow_html=True,
    )


def render_answer(content: str, sources: list[str]) -> None:
    # "$375 million ... $400 million" would otherwise render as a LaTeX formula
    st.markdown(content.replace("$", "\\$"))
    if sources:
        render_sources(sources)


@st.dialog("Remove document")
def confirm_remove(filename: str) -> None:
    st.markdown(f"Remove **{filename}**? You can ingest it again later.")
    cancel_col, remove_col = st.columns(2)
    if cancel_col.button("Cancel", use_container_width=True):
        st.rerun()
    if remove_col.button("Yes, remove", type="primary", use_container_width=True):
        try:
            response = requests.delete(f"{BACKEND_URL}/documents/{quote(filename, safe='')}", timeout=30)
        except requests.RequestException as exc:
            st.error(f"Failed to remove {filename}: {exc}")
            return
        if response.ok:
            st.rerun()
        st.error(f"Failed to remove {filename}: {response.text}")


with st.sidebar:
    st.markdown(
        f'<div class="brand">{LOGO}<div><div class="brand-name">RAG Assistant</div>'
        '<div class="brand-tag">Search. Ask. Get verified answers.</div></div></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">Knowledge base</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or DOCX files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if st.button("Ingest", type="primary", disabled=not uploaded_files, use_container_width=True):
        for uploaded_file in uploaded_files:
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type,
                )
            }
            with st.spinner(f"Ingesting {uploaded_file.name}..."):
                try:
                    response = requests.post(f"{BACKEND_URL}/ingest", files=files, timeout=600)
                    if not response.ok:
                        detail = response.json().get("detail", response.text) if response.headers.get(
                            "content-type", ""
                        ).startswith("application/json") else response.text
                        st.error(f"Failed to ingest {uploaded_file.name}: {detail}")
                        continue
                    data = response.json()
                    st.success(f"{data['filename']}: {data['chunks_ingested']} chunks added")
                except requests.RequestException as exc:
                    st.error(f"Failed to ingest {uploaded_file.name}: {exc}")

    # Ingested files live in the backend's vector store, so list them from there: the
    # uploader and success messages above reset on every rerun, but the documents stay
    st.markdown('<div class="section-label">Documents</div>', unsafe_allow_html=True)
    try:
        docs = requests.get(f"{BACKEND_URL}/documents", timeout=10).json().get("documents", [])
    except requests.RequestException:
        docs = None
    if docs:
        for doc in docs:
            name_col, remove_col = st.columns([6, 1], vertical_alignment="center")
            name_col.markdown(doc_row(doc["filename"], f"{doc['chunks']} chunks"), unsafe_allow_html=True)
            # The ✕ only opens a confirmation, so a stray click can't drop a document
            if remove_col.button("✕", key=f"remove-{doc['filename']}", help="Remove this document", type="tertiary"):
                confirm_remove(doc["filename"])
    elif docs is not None:
        st.caption("None yet. Upload a file and click Ingest.")

    if docs is None:
        status = '<div class="dot off"></div><div><b>Backend not reachable</b><br>Start the backend server</div>'
    else:
        chunks = sum(d["chunks"] for d in docs)
        status = f'<div class="dot"></div><div><b>Knowledge base</b><br>{len(docs)} documents · {chunks} chunks</div>'
    st.markdown(f'<div class="status">{status}</div>', unsafe_allow_html=True)

doc_count = len(docs or [])
st.markdown(
    f'<div class="page-head"><div class="page-icon">{CHAT_ICON}</div><div>'
    f'<div class="page-title">Chat</div><div class="page-sub">{doc_count} documents · answers cite their sources</div>'
    "</div></div>",
    unsafe_allow_html=True,
)

if not st.session_state.messages:
    st.markdown(
        f'<div class="empty"><div class="page-icon">{CHAT_ICON}</div><h3>Ask anything about your documents</h3>'
        "<p>Upload files in the sidebar, then ask a question. Every answer lists the pages it came from.</p></div>",
        unsafe_allow_html=True,
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_answer(message["content"], message.get("sources", []))
        else:
            st.markdown(message["content"])

placeholder = "Ask a follow-up question..." if st.session_state.messages else "Ask a question about your documents..."
if question := st.chat_input(placeholder):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    sources: list[str] = []
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/query", json={"question": question}, timeout=120
                )
                if response.ok:
                    data = response.json()
                    answer = data["answer"]
                    sources = data.get("sources", [])
                elif response.headers.get("content-type", "").startswith("application/json"):
                    answer = f"Error: {response.json().get('detail', response.text)}"
                else:
                    answer = f"Error: {response.status_code} {response.text}"
            except requests.RequestException as exc:
                answer = f"Error contacting backend: {exc}"
        render_answer(answer, sources)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
    # Rerun so the welcome panel above disappears once the first question is asked
    st.rerun()
