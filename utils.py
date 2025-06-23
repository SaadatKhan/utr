from typing import List
from PyPDF2 import PdfReader
from config import CHUNK_SIZE
import logging


def read_document(path: str) -> str:
    
    if path.lower().endswith(".pdf"):
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    else:                                  # treat as plain-text
        with open(path, "r", encoding="utf-8") as f:
            return f.read()


def split_into_chunks(text: str, size: int = CHUNK_SIZE) -> List[str]:
    
    return [
        text[i : i + size].strip()
        for i in range(0, len(text), size)
        if len(text[i : i + size].strip()) > 20
    ]

def show_top_k_chunks(
    chunks: List[str],
    sims,
    k: int = 10,
    preview_len: int = 200,
) -> None:
    
    logging.info("")  # blank line for readability
    logging.info("🔍 Top-k chunks by cosine similarity:")
    for i, (chunk, sim) in enumerate(zip(chunks[:k], sims[:k]), 1):
        preview = chunk.replace("\n", " ")
        #tail    = "…" if len(chunk) > preview_len else ""
        logging.info("%2d.  sim=%.3f  |  %s%s", i, sim, preview,preview_len)


def display_aem(aem_obj, preview_len: int = 120) -> None:
    
    for i, chunk in enumerate(aem_obj.get_aem(), 1):
        preview = chunk.replace("\n", " ")
        #tail    = "…" if len(chunk) > preview_len else ""
        logging.info("   %d. %s%s", i, preview,preview_len)

