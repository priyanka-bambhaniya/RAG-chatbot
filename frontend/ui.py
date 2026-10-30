import os
from pathlib import Path
from urllib.parse import quote

import requests
import streamlit as st
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Chatbot", page_icon="\U0001F4DA")
st.title("RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []


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
    st.header("Ingest documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or DOCX files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
    )
    if st.button("Ingest", disabled=not uploaded_files):
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
    st.subheader("Ingested documents")
    try:
        docs = requests.get(f"{BACKEND_URL}/documents", timeout=10).json().get("documents", [])
        if docs:
            for doc in docs:
                name_col, remove_col = st.columns([5, 1], vertical_alignment="center")
                name_col.markdown(f"- {doc['filename']} ({doc['chunks']} chunks)")
                # The ✕ only opens a confirmation, so a stray click can't drop a document
                if remove_col.button("✕", key=f"remove-{doc['filename']}", help="Remove this document"):
                    confirm_remove(doc["filename"])
        else:
            st.caption("None yet. Upload a file and click Ingest.")
    except requests.RequestException:
        st.caption("Backend not reachable.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/query", json={"question": question}, timeout=120
                )
                if response.ok:
                    data = response.json()
                    answer = data["answer"]
                    if data.get("sources"):
                        answer += "\n\n**Sources:** " + ", ".join(data["sources"])
                elif response.headers.get("content-type", "").startswith("application/json"):
                    answer = f"Error: {response.json().get('detail', response.text)}"
                else:
                    answer = f"Error: {response.status_code} {response.text}"
            except requests.RequestException as exc:
                answer = f"Error contacting backend: {exc}"
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
