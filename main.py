"""
main.py
--------
FastAPI backend for the RAG (Retrieval-Augmented Generation) chatbot.

Flow:
1. POST /upload      -> user uploads a document, it gets chunked + embedded + stored in ChromaDB
2. POST /chat         -> user asks a question, relevant chunks are retrieved, sent to Groq LLM, answer returned
3. GET  /documents    -> list uploaded documents
4. DELETE /documents  -> clear all documents (start fresh)
"""

import os
import shutil
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

from document_processor import process_file

load_dotenv()

# ---------- CONFIG ----------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
UPLOAD_DIR = "uploads"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "documents"
LLM_MODEL = "llama-3.3-70b-versatile"
TOP_K = 4  # how many chunks to retrieve per question

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------- INIT ----------
app = FastAPI(title="RAG Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Local embedding model (free, runs on CPU, no API key needed)
print("Loading embedding model... (first run downloads ~80MB, please wait)")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Persistent local vector database
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

# Groq client (only created if key is present; checked again at request time)
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


# ---------- MODELS ----------
class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


# ---------- ROUTES ----------

@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".pdf", ".docx", ".txt"]:
        raise HTTPException(status_code=400, detail="Only PDF, DOCX, and TXT files are supported.")

    # Save file to disk
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Extract + chunk text
    try:
        chunks = process_file(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

    if not chunks:
        raise HTTPException(status_code=400, detail="No readable text found in this file.")

    # Embed chunks locally (free)
    embeddings = embedder.encode(chunks).tolist()

    # Store in ChromaDB, tagged with source filename
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"source": file.filename, "chunk_index": i} for i in range(len(chunks))]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )

    return {
        "message": f"'{file.filename}' processed successfully.",
        "chunks_added": len(chunks),
    }


@app.get("/documents")
def list_documents():
    data = collection.get()
    sources = set()
    for meta in data.get("metadatas", []):
        if meta and "source" in meta:
            sources.add(meta["source"])
    return {"documents": sorted(sources), "total_chunks": len(data.get("ids", []))}


@app.delete("/documents")
def clear_documents():
    global collection
    chroma_client.delete_collection(COLLECTION_NAME)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    for f in os.listdir(UPLOAD_DIR):
        os.remove(os.path.join(UPLOAD_DIR, f))
    return {"message": "All documents cleared."}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not groq_client:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not set. Add it to your .env file.",
        )

    if collection.count() == 0:
        raise HTTPException(status_code=400, detail="No documents uploaded yet. Upload a document first.")

    # 1. Embed the user's question
    question_embedding = embedder.encode([request.question]).tolist()

    # 2. Retrieve the most relevant chunks
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=min(TOP_K, collection.count()),
    )

    retrieved_chunks = results["documents"][0]
    retrieved_sources = [m["source"] for m in results["metadatas"][0]]

    if not retrieved_chunks:
        return ChatResponse(answer="I couldn't find anything relevant in the uploaded documents.", sources=[])

    context = "\n\n---\n\n".join(retrieved_chunks)

    # 3. Build a strict prompt so the LLM only answers from context,
    #    and formats the answer appropriately (paragraph vs bullet points)
    system_prompt = (
        "You are a helpful assistant that answers questions using ONLY the "
        "provided context from the user's uploaded documents. "
        "If the answer is not in the context, say clearly that the documents "
        "don't contain that information — do not make anything up.\n\n"
        "Formatting rules:\n"
        "- If the answer is a definition, explanation, or single concept, write it as a short paragraph.\n"
        "- If the answer involves multiple items, types, steps, features, or a list of any kind, "
        "format it as markdown bullet points (using '- ') or a numbered list, not a paragraph.\n"
        "- Use **bold** for key terms when helpful.\n"
        "- Keep answers concise and skimmable."
    )

    user_prompt = f"Context from documents:\n{context}\n\nQuestion: {request.question}\n\nAnswer based only on the context above:"

    # 4. Call Groq LLM
    try:
        completion = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        answer = completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq API error: {str(e)}")

    return ChatResponse(answer=answer, sources=list(set(retrieved_sources)))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)