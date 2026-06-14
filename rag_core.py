"""
Shared core for the HR Assistant: Groq client, embeddings, the in-memory vector
store, and a few small LLM helpers. Kept deliberately minimal — the same chunk /
embed / cosine-retrieve pipeline the original mini RAG chatbot used.
"""
import os
import glob
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL = "llama-3.1-8b-instant"
CHUNK_SIZE = 200
OVERLAP = 40
TOP_K = 4
COMPANY_DOCS_DIR = "company_docs"


# --------------------------------------------------------------- resources
@st.cache_resource
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def get_client():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))


# --------------------------------------------------------------- file / text
def read_pdf(path_or_file):
    """Extract text from a PDF given a path or an uploaded file-like object."""
    from pypdf import PdfReader
    return "\n".join(page.extract_text() or "" for page in PdfReader(path_or_file).pages)


def read_uploaded(uploaded_file):
    """Read a Streamlit UploadedFile (pdf / txt / md)."""
    if uploaded_file.name.lower().endswith(".pdf"):
        return read_pdf(uploaded_file)
    return uploaded_file.read().decode("utf-8", errors="ignore")


def chunk_text(text):
    words = text.split()
    chunks = []
    for i in range(0, len(words), CHUNK_SIZE - OVERLAP):
        chunk = " ".join(words[i:i + CHUNK_SIZE]).strip()
        if chunk:
            chunks.append(chunk)
        if i + CHUNK_SIZE >= len(words):
            break
    return chunks


# --------------------------------------------------------------- vector store
@st.cache_resource(show_spinner="Indexing company documents...")
def build_company_index():
    """
    Load every PDF in company_docs/, chunk + embed them once, and cache the
    result for the whole app session. Returns (chunks, embeddings, sources).
    """
    paths = sorted(glob.glob(os.path.join(COMPANY_DOCS_DIR, "*.pdf")))
    chunks, sources = [], []
    for path in paths:
        text = read_pdf(path)
        for chunk in chunk_text(text):
            chunks.append(chunk)
            sources.append(os.path.basename(path))
    if not chunks:
        return [], None, []
    embeddings = get_embedder().encode(chunks, normalize_embeddings=True)
    return chunks, embeddings, sources


def list_company_docs():
    return [os.path.basename(p)
            for p in sorted(glob.glob(os.path.join(COMPANY_DOCS_DIR, "*.pdf")))]


def retrieve(query, chunks, embeddings, sources=None, top_k=TOP_K):
    """Return the top-k most similar chunks (and their sources, if provided)."""
    query_emb = get_embedder().encode([query], normalize_embeddings=True)[0]
    similarities = embeddings @ query_emb
    top_idx = np.argsort(similarities)[::-1][:top_k]
    hits = [chunks[i] for i in top_idx]
    if sources is not None:
        return hits, [sources[i] for i in top_idx]
    return hits, [None] * len(hits)


# --------------------------------------------------------------- LLM helpers
def chat(messages, temperature=0.3, max_tokens=None):
    """Thin wrapper around a Groq chat completion that returns the text."""
    resp = get_client().chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


def answer_with_context(query, context_chunks, role_hint="a helpful HR assistant"):
    """RAG answer: respond using only the retrieved context."""
    context = "\n\n---\n\n".join(context_chunks)
    return chat(
        [
            {"role": "system", "content": (
                f"You are {role_hint}. Answer the user's question using ONLY the "
                "company context below. If the answer is not in the context, say "
                "you don't have that information and suggest contacting HR.\n\n"
                f"CONTEXT:\n{context}"
            )},
            {"role": "user", "content": query},
        ],
        temperature=0.2,
    )
