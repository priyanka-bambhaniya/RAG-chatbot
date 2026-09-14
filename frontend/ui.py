import os
from pathlib import Path

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
                    response = requests.post(f"{BACKEND_URL}/ingest", files=files, timeout=120)
                    response.raise_for_status()
                    data = response.json()
                    st.success(f"{data['filename']}: {data['chunks_ingested']} chunks added")
                except requests.RequestException as exc:
                    st.error(f"Failed to ingest {uploaded_file.name}: {exc}")

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
                response.raise_for_status()
                data = response.json()
                answer = data["answer"]
                if data.get("sources"):
                    answer += "\n\n**Sources:** " + ", ".join(data["sources"])
            except requests.RequestException as exc:
                answer = f"Error contacting backend: {exc}"
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
