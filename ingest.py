import os
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

euri_client = OpenAI(
    api_key=os.getenv("EURI_API_KEY"),
    base_url="https://api.euron.one/api/v1/euri",
)

DATA_DIR = "data"

def get_embedding(text: str) -> list:
    response = euri_client.embeddings.create(
        model="gemini-embedding-2-preview",
        input=[text],
    )
    return response.data[0].embedding[:1536]

def get_ingested_files(domain: str) -> set:
    response = supabase.table("documents") \
        .select("source") \
        .eq("domain", domain) \
        .execute()
    return {row["source"] for row in response.data}

def load_file(path: str) -> list:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(path).load()
    elif ext == ".txt":
        return TextLoader(path).load()
    else:
        print(f"  ⚠  Skipping unsupported file: {Path(path).name}")
        return []

def ingest_domain(domain: str):
    domain_path = Path(DATA_DIR) / domain
    if not domain_path.exists():
        print(f"✗  Folder not found: {domain_path}")
        return

    already_ingested = get_ingested_files(domain)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    all_files = list(domain_path.rglob("*.*"))
    new_files = [
        f for f in all_files
        if f.name not in already_ingested
    ]

    if not new_files:
        print(f"✓  No new files to ingest for '{domain}'")
        return

    print(f"\n📂  Domain: '{domain}' — {len(new_files)} new file(s) found")

    for file_path in new_files:
        print(f"\n📄  File: {file_path.name}")
        docs = load_file(str(file_path))
        if not docs:
            continue

        chunks = splitter.split_documents(docs)
        total = len(chunks)
        print(f"    {total} chunks to ingest")

        for i, chunk in enumerate(chunks):
            print(f"    ⏳ Chunk {i+1}/{total}...", end="\r")

            embedding = get_embedding(chunk.page_content)

            supabase.table("documents").insert({
                "domain":    domain,
                "source":    file_path.name,
                "chunk_id":  i,
                "content":   chunk.page_content,
                "embedding": embedding,
                "metadata":  {
                    "page":      chunk.metadata.get("page", 0),
                    "file_path": str(file_path),
                    "file_type": file_path.suffix.lower(),
                }
            }).execute()

            time.sleep(0.5)

        print(f"    ✅ {file_path.name} — {total} chunks ingested")

def ingest_all():
    data_path = Path(DATA_DIR)
    if not data_path.exists():
        print("✗  data/ folder not found")
        return

    domains = [f.name for f in data_path.iterdir() if f.is_dir()]
    if not domains:
        print("✗  No domain folders found inside data/")
        return

    print(f"Found domains: {domains}")
    for domain in domains:
        ingest_domain(domain)

    print("\n✅  All domains ingested successfully!")

if __name__ == "__main__":
    ingest_all()