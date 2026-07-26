"""
document_processor.py
----------------------
Handles reading PDF / DOCX / TXT files and splitting them into
overlapping text chunks that are small enough to embed and retrieve
accurately.
"""

import os
from pypdf import PdfReader
from docx import Document


def extract_text(file_path: str) -> str:
    """Extract raw text from a PDF, DOCX, or TXT file."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
        return text

    elif ext == ".docx":
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)

    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    else:
        raise ValueError(f"Unsupported file type: {ext}")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
    """
    Split text into overlapping word-based chunks.

    chunk_size = number of words per chunk
    overlap    = number of words repeated between consecutive chunks
                 (this keeps context from being cut off at chunk boundaries)
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap  # move forward, keeping overlap

    return chunks


def process_file(file_path: str) -> list[str]:
    """Full pipeline: extract text from a file, then chunk it."""
    text = extract_text(file_path)
    return chunk_text(text)