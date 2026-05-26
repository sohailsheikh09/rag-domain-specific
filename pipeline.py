import os
from typing import TypedDict
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from supabase import create_client
from langgraph.graph import StateGraph, END
from serpapi import GoogleSearch

load_dotenv()

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

euri_client = OpenAI(
    api_key=os.getenv("EURI_API_KEY"),
    base_url="https://api.euron.one/api/v1/euri",
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)

# ── Embedding ────────────────────────────────────────────
def get_embedding(text: str) -> list:
    response = euri_client.embeddings.create(
        model="gemini-embedding-2-preview",
        input=[text],
    )
    return response.data[0].embedding

# ── State ────────────────────────────────────────────────
class RAGState(TypedDict):
    question:           str
    domain:             str
    rewritten_question: str
    retrieved_chunks:   list
    grade:              str
    retry_count:        int
    web_search_results: str
    final_answer:       str
    is_on_topic:        bool
    chat_history:       list
    similarity_scores:  list
    answer_source:      str

# ── Nodes ────────────────────────────────────────────────
def classifier_node(state: RAGState) -> RAGState:
    question = state["question"]
    domain   = state["domain"]
    history  = state.get("chat_history", [])

    history_text = "\n".join([
        f"{m['role']}: {m['content']}" for m in history[-10:]
    ]) if history else "No previous conversation"

    prompt = f"""You are a strict domain classifier.
The user has selected the domain: {domain}
The domain contains official documents, city development plans,
infrastructure, economy, tourism and geographic information about {domain}.

Previous conversation:
{history_text}

Current question: {question}

Is this question genuinely relevant to the '{domain}' domain?
Reply with only YES or NO."""

    response     = llm.invoke(prompt)
    is_on_topic  = response.content.strip().upper() == "YES"
    return {**state, "is_on_topic": is_on_topic}


def rewriter_node(state: RAGState) -> RAGState:
    question = state["question"]
    domain   = state["domain"]
    history  = state.get("chat_history", [])

    history_text = "\n".join([
        f"{m['role']}: {m['content']}" for m in history[-10:]
    ]) if history else ""

    prompt = f"""You are a query rewriter for a RAG system.
Domain: {domain}
Previous conversation:
{history_text}

Current question: {question}

Rewrite this question considering the conversation history to make
it more specific for document retrieval.
Return only the rewritten question, nothing else."""

    response  = llm.invoke(prompt)
    rewritten = response.content.strip()
    return {**state, "rewritten_question": rewritten}


def retriever_node(state: RAGState) -> RAGState:
    question = state["rewritten_question"] or state["question"]
    domain   = state["domain"]

    embedding = get_embedding(question)

    results = supabase.rpc("match_documents", {
        "query_embedding": embedding,
        "match_domain":    domain,
        "match_count":     5
    }).execute()

    chunks = results.data
    scores = [round(c["similarity"], 3) for c in chunks]
    return {**state, "retrieved_chunks": chunks, "similarity_scores": scores}


def grader_node(state: RAGState) -> RAGState:
    question      = state["question"]
    chunks        = state["retrieved_chunks"]
    retry_count   = state["retry_count"]
    relevant_count = 0

    for chunk in chunks:
        prompt = f"""You are a retrieval grader.
Question: {question}
Retrieved chunk: {chunk['content']}

Does this chunk contain useful information to answer the question?
Reply with only YES or NO."""
        response = llm.invoke(prompt)
        if response.content.strip().upper() == "YES":
            relevant_count += 1

    total            = len(chunks)
    relevance_ratio  = relevant_count / total if total > 0 else 0
    grade            = "good" if relevance_ratio >= 0.4 else "bad"
    return {**state, "grade": grade, "retry_count": retry_count}


def refine_node(state: RAGState) -> RAGState:
    question    = state["rewritten_question"] or state["question"]
    domain      = state["domain"]
    retry_count = state["retry_count"] + 1

    prompt = f"""You are a query refinement expert.
The following question did not retrieve relevant results from the {domain} domain:
{question}

Rewrite it differently using different keywords to improve retrieval.
Return only the refined question, nothing else."""

    response = llm.invoke(prompt)
    refined  = response.content.strip()
    return {**state, "rewritten_question": refined, "retry_count": retry_count}


def web_search_node(state: RAGState) -> RAGState:
    question = state["rewritten_question"] or state["question"]
    domain   = state["domain"]

    search = GoogleSearch({
        "q":       f"{domain} {question}",
        "api_key": os.getenv("SERPAPI_KEY"),
        "num":     5
    })

    results  = search.get_dict()
    organic  = results.get("organic_results", [])
    web_results = "\n".join([
        f"- {r.get('title', '')}: {r.get('snippet', '')}"
        for r in organic
    ])

    return {**state, "web_search_results": web_results, "answer_source": "web search"}


def generator_node(state: RAGState) -> RAGState:
    question   = state["question"]
    chunks     = state["retrieved_chunks"]
    web_results = state["web_search_results"]
    history    = state.get("chat_history", [])

    history_text = "\n".join([
        f"{m['role']}: {m['content']}" for m in history[-10:]
    ]) if history else ""

    if chunks:
        context     = "\n\n".join([
            f"Source: {c['source']}\n{c['content']}" for c in chunks
        ])
        source_type = "domain documents"
    else:
        context     = web_results
        source_type = "web search"

    prompt = f"""You are a helpful assistant answering questions about {state['domain']}.

Previous conversation:
{history_text}

Context from {source_type}:
{context}

Current question: {question}

Answer based on the context and conversation history.
Be clear and concise."""

    response = llm.invoke(prompt)
    answer   = response.content.strip()
    return {**state, "final_answer": answer, "answer_source": source_type}


def off_topic_node(state: RAGState) -> RAGState:
    message = f"Your question does not seem related to the '{state['domain']}' domain. Please ask a domain relevant question."
    return {**state, "final_answer": message, "answer_source": "off-topic"}


# ── Edges ────────────────────────────────────────────────
def route_after_classifier(state: RAGState) -> str:
    return "rewriter" if state["is_on_topic"] else "off_topic"

def route_after_grader(state: RAGState) -> str:
    if state["grade"] == "good":
        return "generator"
    elif state["retry_count"] >= 3:
        return "web_search"
    else:
        return "refine"

# ── Graph ────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("classifier", classifier_node)
    graph.add_node("rewriter",   rewriter_node)
    graph.add_node("retriever",  retriever_node)
    graph.add_node("grader",     grader_node)
    graph.add_node("refine",     refine_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("generator",  generator_node)
    graph.add_node("off_topic",  off_topic_node)

    graph.set_entry_point("classifier")

    graph.add_conditional_edges("classifier", route_after_classifier, {
        "rewriter":  "rewriter",
        "off_topic": "off_topic"
    })
    graph.add_edge("rewriter",  "retriever")
    graph.add_edge("retriever", "grader")
    graph.add_conditional_edges("grader", route_after_grader, {
        "generator":  "generator",
        "refine":     "refine",
        "web_search": "web_search"
    })
    graph.add_edge("refine",     "retriever")
    graph.add_edge("web_search", "generator")
    graph.add_edge("generator",  END)
    graph.add_edge("off_topic",  END)

    return graph.compile()

rag_app = build_graph()