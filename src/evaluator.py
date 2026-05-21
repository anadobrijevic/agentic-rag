"""
evaluator.py - RAGAS evaluation of the RAG pipeline.

Evaluates all metrics at once for simplicity and robustness.
"""

import logging
from typing import List, Dict, Any, Callable, Optional

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Pretpostavka je da ove funkcije postoje u vašem projektu
from src.config import get_llm, get_embeddings

logger = logging.getLogger(__name__)

# Definišemo metrike koje ćemo koristiti
METRICS = [
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
]

def _build_dataset(test_results: List[Dict[str, Any]]) -> Dataset:
    """Helper function to convert test results into a Hugging Face Dataset."""
    # Filtriramo rezultate da bismo osigurali da imamo sve potrebne kolone
    # RAGAS zahteva 'question', 'answer', 'contexts', i 'ground_truth' za neke metrike
    filtered_results = []
    for r in test_results:
        if all(k in r for k in ["question", "answer", "contexts", "ground_truth"]):
            filtered_results.append(r)
        else:
            logger.warning(f"Skipping result due to missing keys: {r.get('question', 'N/A')}")
            
    if not filtered_results:
        # Ako nema validnih rezultata, vraćamo prazan Dataset
        return Dataset.from_dict({
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [],
        })

    return Dataset.from_dict({
        "question":     [r["question"]     for r in filtered_results],
        "answer":       [r["answer"]       for r in filtered_results],
        "contexts":     [r["contexts"]     for r in filtered_results],
        "ground_truth": [r["ground_truth"] for r in filtered_results],
    })

def run_ragas_evaluation(
    test_results: List[Dict[str, Any]],
    progress_cb: Optional[Callable] = None,
    progress_offset: float = 0.0,
    progress_range: float = 1.0,
) -> pd.DataFrame:
    """
    Runs RAGAS evaluation over agent results by evaluating all metrics at once.
    """
    if not test_results:
        logger.warning("No results to evaluate.")
        return pd.DataFrame()

    dataset = _build_dataset(test_results)
    if not dataset or len(dataset) == 0:
        logger.error("Dataset is empty after filtering. Cannot run evaluation.")
        return pd.DataFrame()

    ragas_llm = LangchainLLMWrapper(get_llm(temperature=0.0))
    ragas_emb = LangchainEmbeddingsWrapper(get_embeddings())

    logger.info("Evaluating all RAGAS metrics at once...")
    if progress_cb:
        progress_cb(progress_offset, "Evaluating RAGAS metrics...")

    try:
        # Prosleđujemo celu listu METRICS da se izvrše odjednom.
        # Parametar 'is_async' je UKLONJEN da bi bio kompatibilan sa vašom verzijom RAGAS-a.
        result = evaluate(
            dataset=dataset,
            metrics=METRICS,
            llm=ragas_llm,
            embeddings=ragas_emb,
            raise_exceptions=True,
        )

        if result is None:
            logger.error("RAGAS evaluation returned None.")
            return pd.DataFrame()

        result_df = result.to_pandas()

    except Exception as e:
        logger.error(f"RAGAS evaluation failed critically: {e}", exc_info=True)
        # Vraćamo prazan DataFrame u slučaju nepredviđene greške
        return pd.DataFrame()

    if progress_cb:
        progress_cb(progress_offset + progress_range, "Evaluation complete.")

    logger.info("RAGAS evaluation complete.")
    return result_df

def compute_summary(df: pd.DataFrame) -> Dict[str, float]:
    """
    Returns the average score per metric from the evaluation DataFrame.
    """
    summary = {}
    metric_names = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
    
    if df.empty:
        logger.warning("Cannot compute summary from an empty DataFrame.")
        return {name: 0.0 for name in metric_names}

    for col in metric_names:
        if col in df.columns:
            # .mean() ignoriše NaN vrednosti po default-u (skipna=True)
            summary[col] = round(float(df[col].mean()), 4)
        else:
            logger.warning(f"Metric column '{col}' not found in DataFrame for summary.")
            summary[col] = 0.0
            
    return summary
