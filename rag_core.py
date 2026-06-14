import os
import glob
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL="llama-3.1-8b-instant"
CHUNK_SIZE=200
OVERLAP=40
TOP_K=4
COMPANY_DOCS_DIR="company_docs"

@st.cache_resource
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def get_client():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))

def read_pdf(path_or_file):
    from pypdf import PdfReader
    return "\n".join(page.extract_text() or "" for page in PdfReader(path_or_file).pages)

def read_uploaded(uploaded_file):
    if uploaded_file.name.lower().endswith(".pdf"):
        return read_pdf(uploaded_file)
    return uploaded_file.read().decode("utf-8", errors="ignore")

def chunk_text(text):
    words=text.split()
    chunks=[]
    for i in range(0, len(words), CHUNK_SIZE-OVERLAP):
        chunk=" ".join(words[i:i+CHUNK_SIZE]).strip()
        if chunk:
            chunks.append(chunk)
        if i+CHUNK_SIZE>=len(words):
            break
    return chunks

@st.cache_resource(show_spinner="Indexing company documents...")
def build_company_index():
    paths=sorted(glob.glob(os.path.join(COMPANY_DOCS_DIR, "*.pdf")))
    chunks, sources=[], []
    for path in paths:
        for chunk in chunk_text(read_pdf(path)):
            chunks.append(chunk)
            sources.append(os.path.basename(path))
    if not chunks:
        return [], None, []
    return chunks, get_embedder().encode(chunks, normalize_embeddings=True), sources

def list_company_docs():
    return [os.path.basename(p) for p in sorted(glob.glob(os.path.join(COMPANY_DOCS_DIR, "*.pdf")))]

def retrieve(query, chunks, embeddings, sources=None, top_k=TOP_K):
    query_emb=get_embedder().encode([query], normalize_embeddings=True)[0]
    top_idx=np.argsort(embeddings@query_emb)[::-1][:top_k]
    hits=[chunks[i] for i in top_idx]
    return hits, ([sources[i] for i in top_idx] if sources is not None else [None]*len(hits))

def chat(messages, temperature=0.3, max_tokens=None):
    resp=get_client().chat.completions.create(
        model=MODEL, messages=messages, temperature=temperature, max_tokens=max_tokens)
    return resp.choices[0].message.content

def answer_with_context(query, context_chunks, role_hint="a helpful HR assistant"):
    context="\n\n---\n\n".join(context_chunks)
    return chat(
        [
            {"role":"system","content":(
                f"You are {role_hint}. Answer the user's question using ONLY the "
                "company context below. If the answer is not in the context, say "
                "you don't have that information and suggest contacting HR.\n\n"
                f"CONTEXT:\n{context}"
            )},
            {"role":"user","content":query},
        ],
        temperature=0.2,
    )
