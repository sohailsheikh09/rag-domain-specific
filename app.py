import os
import sqlite3
import streamlit as st
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from pipeline import rag_app, RAGState

load_dotenv()

DB_PATH  = "chat_history.db"
DATA_DIR = "data"

# ── SQLite setup ─────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("""
        create table if not exists chat_history (
            id        integer primary key autoincrement,
            domain    text not null,
            role      text not null,
            content   text not null,
            timestamp text not null
        )
    """)
    conn.commit()
    conn.close()

def save_message(domain: str, role: str, content: str):
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute(
        "insert into chat_history (domain, role, content, timestamp) values (?, ?, ?, ?)",
        (domain, role, content, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

def load_history(domain: str) -> list:
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute(
        "select role, content, timestamp from chat_history where domain=? order by id desc limit 10",
        (domain,)
    )
    rows = c.fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1], "timestamp": r[2]} for r in reversed(rows)]

def clear_history(domain: str):
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("delete from chat_history where domain=?", (domain,))
    conn.commit()
    conn.close()

# ── Domain helpers ───────────────────────────────────────
def get_domains() -> list:
    data_path = Path(DATA_DIR)
    if not data_path.exists():
        return []
    return [f.name for f in data_path.iterdir() if f.is_dir()]

def create_domain(name: str):
    Path(DATA_DIR, name.lower().strip()).mkdir(parents=True, exist_ok=True)

def get_doc_count(domain: str) -> int:
    try:
        from supabase import create_client
        supabase = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_KEY")
        )
        result = supabase.table("documents") \
            .select("id", count="exact") \
            .eq("domain", domain) \
            .execute()
        return result.count or 0
    except:
        return 0

# ── App ──────────────────────────────────────────────────
init_db()

st.set_page_config(
    page_title="Domain RAG",
    page_icon="🗂",
    layout="wide"
)

st.title("🗂 Domain-Specific RAG")

# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.header("Domain Management")

    # create new domain
    with st.expander("➕ Create new domain"):
        new_domain = st.text_input("Domain name", placeholder="e.g. delhi")
        if st.button("Create domain"):
            if new_domain.strip():
                create_domain(new_domain)
                st.success(f"Domain '{new_domain}' created!")
                st.rerun()
            else:
                st.error("Please enter a domain name")

    # upload documents
    with st.expander("📄 Upload documents"):
        domains = get_domains()
        if domains:
            upload_domain = st.selectbox("Select domain", domains, key="upload_domain")
            uploaded_file = st.file_uploader("Upload PDF", type=["pdf", "txt"])
            if uploaded_file and st.button("Save file"):
                save_path = Path(DATA_DIR) / upload_domain / uploaded_file.name
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"Saved to {upload_domain}/")
                st.info("Run ingest.py to embed the new file")
        else:
            st.info("Create a domain first")

    st.divider()

    # domain selector
    st.header("Start chatting")
    domains = get_domains()

    if not domains:
        st.warning("No domains found. Create one above.")
        st.stop()

    selected_domain = st.selectbox("Select domain", domains, key="selected_domain")

    if selected_domain:
        doc_count = get_doc_count(selected_domain)
        st.metric("Chunks in database", doc_count)

    st.divider()

    if st.button("🗑 Clear chat history"):
        clear_history(selected_domain)
        st.session_state.messages = []
        st.rerun()

# ── Chat area ────────────────────────────────────────────
if "current_domain" not in st.session_state:
    st.session_state.current_domain = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# reload history when domain changes
if st.session_state.current_domain != selected_domain:
    st.session_state.current_domain = selected_domain
    history = load_history(selected_domain)
    st.session_state.messages = history

st.subheader(f"Chatting in: {selected_domain.upper()}")

# display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# chat input
if prompt := st.chat_input(f"Ask something about {selected_domain}..."):

    # show user message
    with st.chat_message("user"):
        st.markdown(prompt)

    save_message(selected_domain, "user", prompt)
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "timestamp": datetime.now().isoformat()
    })

    # run pipeline
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            history = load_history(selected_domain)

            result = rag_app.invoke({
                "question":           prompt,
                "domain":             selected_domain,
                "rewritten_question": "",
                "retrieved_chunks":   [],
                "grade":              "",
                "retry_count":        0,
                "web_search_results": "",
                "final_answer":       "",
                "is_on_topic":        False,
                "chat_history":       history,
                "similarity_scores":  [],
                "answer_source":      ""
            })

        answer       = result["final_answer"]
        source       = result["answer_source"]
        scores       = result["similarity_scores"]
        rewritten    = result["rewritten_question"]

        st.markdown(answer)

        # show metadata
        with st.expander("🔍 retrieval details"):
            st.write(f"**Answer source:** {source}")
            if rewritten:
                st.write(f"**Rewritten query:** {rewritten}")
            if scores:
                st.write("**Similarity scores:**")
                for i, score in enumerate(scores):
                    st.progress(score, text=f"Chunk {i+1}: {score}")

    save_message(selected_domain, "assistant", answer)
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "timestamp": datetime.now().isoformat()
    })