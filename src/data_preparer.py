"""
data_preparer.py - Loads PDF and generates test set using LLM.
All prompts and outputs are in English.
"""

import json
import random
import logging
from typing import List, Dict

from langchain_community.document_loaders import PyMuPDFLoader
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import get_llm, CHUNK_SIZE, CHUNK_OVERLAP, NUM_TEST_QUESTIONS

logger = logging.getLogger(__name__)


def load_and_split_pdf(pdf_path: str) -> List[Document]:
    """Loads a PDF and splits it into chunks."""
    logger.info(f"Loading PDF: {pdf_path}")
    loader = PyMuPDFLoader(pdf_path)
    pages  = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    docs = splitter.split_documents(pages)
    logger.info(f"Split into {len(docs)} chunks.")
    return docs


def generate_test_questions(
    docs: List[Document],
    n: int = NUM_TEST_QUESTIONS,
    progress_cb=None,
) -> List[Dict]:
    """
    Generates n question+answer pairs from random chunks using the LLM.
    Returns a list of dicts: {question, ground_truth, context}.
    All questions and answers are in English.
    """
    llm         = get_llm(temperature=0.3)
    sample_docs = random.sample(docs, min(n * 2, len(docs)))
    test_set    = []

    for i, doc in enumerate(sample_docs[:n]):
        if progress_cb:
            progress_cb(i + 1, n, f"Generating question {i + 1}/{n}...")

        prompt = f"""You are a question generation assistant. Based on the text below, generate ONE clear and specific question that can be answered from the text, along with the correct answer.

Respond ONLY with valid JSON in this exact format (no extra text, no markdown):
{{"question": "...", "ground_truth": "..."}}

Rules:
- The question must be answerable from the provided text only.
- The ground_truth must be a concise, factual answer in English.
- Do not add any explanation outside the JSON.

Text:
{doc.page_content[:800]}
"""
        try:
            response = llm.invoke(prompt)
            raw = response.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(raw)
            if "question" in parsed and "ground_truth" in parsed:
                test_set.append({
                    "question":     parsed["question"],
                    "ground_truth": parsed["ground_truth"],
                    "context":      doc.page_content,
                })
        except Exception as e:
            logger.warning(f"Skipping chunk {i}: {e}")

    logger.info(f"Generated {len(test_set)} test questions.")
    return test_set