"""
tool_factory.py - Kreira FAISS, BM25 i hibridni retriever.
Eksportuje tri factory funkcije za tri odvojena agenta.
"""

import logging
from typing import List

from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

from src.config import get_embeddings, TOP_K_VECTOR, TOP_K_BM25, TOP_K_HYBRID, HYBRID_ALPHA

logger = logging.getLogger(__name__)


# ── Izgradnja indeksa ─────────────────────────────────────────────────────────

def build_vector_store(docs: List[Document]) -> FAISS:
    logger.info("Gradim FAISS vektorski indeks...")
    store = FAISS.from_documents(docs, get_embeddings())
    logger.info("FAISS gotov.")
    return store


def build_bm25_retriever(docs: List[Document]) -> BM25Retriever:
    logger.info("Gradim BM25 indeks...")
    r = BM25Retriever.from_documents(docs)
    r.k = TOP_K_BM25
    logger.info("BM25 gotov.")
    return r


# ── Formatovanje ──────────────────────────────────────────────────────────────

def _fmt(docs: List[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        src  = d.metadata.get("source", "?")
        page = d.metadata.get("page", "?")
        parts.append(f"[Izvor {i} | str.{page} | {src}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


# ── Hibridna pretraga (RRF) ───────────────────────────────────────────────────

def _hybrid(query, vector_store, bm25_retriever):
    vec_docs  = vector_store.similarity_search(query, k=TOP_K_VECTOR)
    bm25_docs = bm25_retriever.get_relevant_documents(query)
    scores = {}
    for rank, doc in enumerate(vec_docs):
        key = doc.page_content[:200]
        scores.setdefault(key, {"doc": doc, "score": 0.0})
        scores[key]["score"] += HYBRID_ALPHA * (1.0 / (rank + 1))
    for rank, doc in enumerate(bm25_docs):
        key = doc.page_content[:200]
        scores.setdefault(key, {"doc": doc, "score": 0.0})
        scores[key]["score"] += (1 - HYBRID_ALPHA) * (1.0 / (rank + 1))
    return [v["doc"] for v in sorted(scores.values(), key=lambda x: x["score"], reverse=True)[:TOP_K_HYBRID]]


# ── Factory funkcije ──────────────────────────────────────────────────────────

def create_tools(vector_store: FAISS, bm25_retriever: BM25Retriever) -> List[Tool]:
    """Hybrid agent — has all 3 retrieval strategies available."""
    return [
        Tool(name="hybrid_search",
             func=lambda q: _fmt(_hybrid(q, vector_store, bm25_retriever)),
             description="Hybrid search combining BM25 + FAISS using Reciprocal Rank Fusion. Best first choice for any question."),
        Tool(name="vector_search",
             func=lambda q: _fmt(vector_store.similarity_search(q, k=TOP_K_VECTOR)),
             description="Semantic vector search using FAISS embeddings. Good for conceptual questions and synonyms."),
        Tool(name="bm25_search",
             func=lambda q: _fmt(bm25_retriever.get_relevant_documents(q)),
             description="BM25 lexical keyword search. Good for exact names, terms, dates and numbers."),
    ]


def create_vector_only_tools(vector_store: FAISS) -> List[Tool]:
    """Vector-only agent — uses ONLY FAISS semantic search."""
    return [
        Tool(name="vector_search",
             func=lambda q: _fmt(vector_store.similarity_search(q, k=TOP_K_VECTOR)),
             description="Semantic vector search in FAISS index. This is the only available tool."),
    ]


def create_bm25_only_tools(bm25_retriever: BM25Retriever) -> List[Tool]:
    """BM25-only agent — uses ONLY BM25 lexical search."""
    return [
        Tool(name="bm25_search",
             func=lambda q: _fmt(bm25_retriever.get_relevant_documents(q)),
             description="BM25 lexical keyword search. This is the only available tool."),
    ]