import os
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

@st.cache_resource
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def get_client():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))

def read_file(uploaded_file):
    if uploaded_file.name.lower().endswith(".pdf"):
        from pypdf import PdfReader
        return "\n".join(page.extract_text() or "" for page in PdfReader(uploaded_file).pages)
    return uploaded_file.read().decode("utf-8",errors="ignore")

def chunk_text(text):
    words=text.split()
    chunks=[]
    for i in range(0,len(words),CHUNK_SIZE-OVERLAP):
        chunk=" ".join(words[i:i+CHUNK_SIZE]).strip()
        if chunk:
            chunks.append(chunk)
        if i+CHUNK_SIZE>=len(words):
            break
    return chunks

def retrieve(query,chunks,embeddings):
    query_emb=get_embedder().encode([query],normalize_embeddings=True)[0]
    similarities=embeddings@query_emb 
    top_idx=np.argsort(similarities)[::-1][:TOP_K]
    return [chunks[i] for i in top_idx]

def generate_answer(query,context_chunks):
    context="\n\n---\n\n".join(context_chunks)
    response=get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role":"system","content":(
                "You are a helpful assistant. Answer the user's question using ONLY the "
                "context below. If the answer is not in the context, say you don't know.\n\n"
                f"CONTEXT:\n{context}"
            )},
            {"role":"user","content":query},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content

st.set_page_config(page_title="Mini RAG Chatbot",page_icon="📄")
st.title("📄 Mini RAG Chatbot")
st.caption(f"Upload a document, then ask questions about it. Powered by {MODEL} via Groq.")
if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY not found. Add it to the .env file and restart.")
    st.stop()
st.session_state.setdefault("messages",[])
st.session_state.setdefault("chunks",None)
st.session_state.setdefault("embeddings",None)

with st.sidebar:
    st.header("Document")
    uploaded=st.file_uploader("Upload a file",type=["txt","md","pdf"])
    if uploaded and st.session_state.get("doc_name")!=uploaded.name:
        with st.spinner("Reading, chunking and embedding..."):
            text=read_file(uploaded)
            chunks=chunk_text(text)
            if not chunks:
                st.error("Couldn't extract any text from that file.")
            else:
                st.session_state.chunks=chunks
                st.session_state.embeddings=get_embedder().encode(chunks,normalize_embeddings=True)
                st.session_state.doc_name=uploaded.name
                st.session_state.messages=[]
    if st.session_state.chunks:
        st.success(f"Indexed **{st.session_state.doc_name}** ({len(st.session_state.chunks)} chunks)")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question:=st.chat_input("Ask something about your document..."):
    if st.session_state.chunks is None:
        st.warning("Please upload a document first (see sidebar).")
    else:
        st.session_state.messages.append({"role":"user","content":question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                top_chunks=retrieve(question,st.session_state.chunks,st.session_state.embeddings)
                answer=generate_answer(question,top_chunks)
            st.markdown(answer)
            with st.expander("Retrieved context"):
                for i,chunk in enumerate(top_chunks,1):
                    st.markdown(f"**Chunk {i}:** {chunk[:300]}...")
        st.session_state.messages.append({"role":"assistant","content":answer})