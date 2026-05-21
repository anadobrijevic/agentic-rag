"""
app.py - Agentic RAG Benchmark — Gradio interface
Three separate agents: BM25 / Vector / Hybrid
Displays questions, agent answers, ground truth and RAGAS metrics.
Saves all results to the results/ folder.
"""

import logging
import json
import shutil
import os
from pathlib import Path
from typing import List, Dict, Any

import gradio as gr
import pandas as pd

os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"]  = "localhost,127.0.0.1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

_state: Dict[str, Any] = {
    "docs": None, "vector_store": None, "bm25_retriever": None,
    "agent_hybrid": None, "agent_vector": None, "agent_bm25": None,
    "test_set": None,
    "results_hybrid": None, "results_vector": None, "results_bm25": None,
}


def index_pdf(pdf_file, progress=gr.Progress(track_tqdm=True)):
    from src.data_preparer import load_and_split_pdf
    from src.tool_factory import (
        build_vector_store, build_bm25_retriever,
        create_tools, create_vector_only_tools, create_bm25_only_tools,
    )
    from src.agent_orchestrator import build_agent

    if pdf_file is None:
        return "Please upload a PDF file first.", gr.update(interactive=False)
    try:
        progress(0.1, desc="Loading PDF...")
        data_dir = Path("data"); data_dir.mkdir(exist_ok=True)
        dest = data_dir / "book.pdf"
        shutil.copy(pdf_file.name, dest)

        progress(0.3, desc="Splitting into chunks...")
        docs = load_and_split_pdf(str(dest))
        _state["docs"] = docs

        progress(0.5, desc="Building FAISS vector index...")
        _state["vector_store"] = build_vector_store(docs)

        progress(0.7, desc="Building BM25 index...")
        _state["bm25_retriever"] = build_bm25_retriever(docs)

        progress(0.9, desc="Initializing three agents...")
        _state["agent_hybrid"] = build_agent(create_tools(_state["vector_store"], _state["bm25_retriever"]), mode="hybrid")
        _state["agent_vector"] = build_agent(create_vector_only_tools(_state["vector_store"]), mode="vector")
        _state["agent_bm25"]   = build_agent(create_bm25_only_tools(_state["bm25_retriever"]), mode="bm25")

        progress(1.0)
        msg = (
            f"PDF indexed successfully!\n"
            f"Total chunks: {len(docs)}\n"
            f"FAISS vector index: ready\n"
            f"BM25 lexical index: ready\n"
            f"Hybrid RRF: ready\n"
            f"3 ReAct agents initialized"
        )
        return msg, gr.update(interactive=True)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return f"Error: {str(e)}", gr.update(interactive=False)


def generate_questions(n_questions: int, progress=gr.Progress(track_tqdm=True)):
    from src.data_preparer import generate_test_questions

    if _state["docs"] is None:
        return "Please index a PDF first.", None, gr.update(interactive=False)
    try:
        def cb(current, total, msg):
            progress(current / total, desc=msg)

        test_set = generate_test_questions(_state["docs"], n=int(n_questions), progress_cb=cb)
        _state["test_set"] = test_set

        with open(RESULTS_DIR / "test_set.json", "w", encoding="utf-8") as f:
            json.dump(test_set, f, ensure_ascii=False, indent=2)

        df = pd.DataFrame([
            {
                "#": i + 1,
                "Question": t["question"],
                "Ground Truth": t["ground_truth"],
                "Context (excerpt)": t["context"][:150] + "...",
            }
            for i, t in enumerate(test_set)
        ])
        msg = f"Generated {len(test_set)} questions — saved to results/test_set.json"
        return msg, df, gr.update(interactive=True)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return f"Error: {str(e)}", None, gr.update(interactive=False)


def _run_one_agent(agent, test_set, label, progress, p0, p_range):
    from src.agent_orchestrator import run_agent
    results = []
    for i, item in enumerate(test_set):
        progress(p0 + (i + 1) / len(test_set) * p_range, desc=f"[{label}] {i+1}/{len(test_set)}")
        r = run_agent(agent, item["question"])
        results.append({
            "question":     item["question"],
            "ground_truth": item["ground_truth"],
            "answer":       r["answer"],
            "contexts":     r["contexts"],
            "steps":        len(r["steps"]),
            "retrieval":    label,
        })
    return results


def run_benchmark(progress=gr.Progress(track_tqdm=True)):
    if _state["agent_hybrid"] is None or _state["test_set"] is None:
        return "Please index a PDF and generate questions first.", None, None, None, gr.update(interactive=False)
    try:
        ts = _state["test_set"]
        res_bm25   = _run_one_agent(_state["agent_bm25"],   ts, "BM25",   progress, 0.0,  0.33)
        res_vector = _run_one_agent(_state["agent_vector"], ts, "Vector", progress, 0.33, 0.33)
        res_hybrid = _run_one_agent(_state["agent_hybrid"], ts, "Hybrid", progress, 0.66, 0.34)

        _state["results_bm25"]   = res_bm25
        _state["results_vector"] = res_vector
        _state["results_hybrid"] = res_hybrid

        for name, res in [("bm25", res_bm25), ("vector", res_vector), ("hybrid", res_hybrid)]:
            with open(RESULTS_DIR / f"raw_{name}.json", "w", encoding="utf-8") as f:
                json.dump([{**r, "contexts": [c[:400] for c in r["contexts"]]} for r in res], f, ensure_ascii=False, indent=2)

        def make_df(results, label):
            return pd.DataFrame([{
                "#":                    i + 1,
                "Question":             r["question"],
                "Ground Truth":         r["ground_truth"],
                f"Agent Answer [{label}]": r["answer"],
                "Steps":                r["steps"],
            } for i, r in enumerate(results)])

        msg = f"Benchmark complete: {len(ts)} questions x 3 agents = {len(ts)*3} answers\nSaved to results/raw_*.json"
        return msg, make_df(res_bm25, "BM25"), make_df(res_vector, "Vector"), make_df(res_hybrid, "Hybrid"), gr.update(interactive=True)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return f"Error: {str(e)}", None, None, None, gr.update(interactive=False)


def run_evaluation(progress=gr.Progress(track_tqdm=True)):
    from src.evaluator import run_ragas_evaluation, compute_summary

    if _state["results_hybrid"] is None:
        return "Please run the benchmark first.", None, None
    try:
        all_summaries = {}
        all_dfs       = {}

        # 3 strategies × 4 metrics = 12 steps total
        # Divide progress bar into 3 equal slices: 0→0.33, 0.33→0.66, 0.66→1.0
        runs = [
            ("BM25",   _state["results_bm25"],   0.00, 0.33),
            ("Vector", _state["results_vector"], 0.33, 0.33),
            ("Hybrid", _state["results_hybrid"], 0.66, 0.34),
        ]

        for name, results, offset, rng in runs:

            def make_cb(n=name, o=offset, r=rng):
                def cb(pos, msg):
                    progress(o + pos * r, desc=f"[{n}] {msg}")
                return cb

            df = run_ragas_evaluation(
                results,
                progress_cb=make_cb(),
                progress_offset=0.0,
                progress_range=1.0,
            )
            all_summaries[name] = compute_summary(df)
            all_dfs[name]       = df
            df.to_csv(RESULTS_DIR / f"ragas_{name.lower()}.csv", index=False)

        progress(1.0, desc="Building summary tables...")

        metrics = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
        rows = []
        for metric in metrics:
            row = {"Metric": metric}
            for name in ["BM25", "Vector", "Hybrid"]:
                v = all_summaries[name].get(metric, 0.0)
                emoji = "🟢" if v >= 0.8 else "🟡" if v >= 0.6 else "🟠" if v >= 0.4 else "🔴"
                row[name] = f"{v:.4f} {emoji}"
            rows.append(row)
        summary_df = pd.DataFrame(rows)

        detail_parts = []
        for name, df in all_dfs.items():
            d = df.copy(); d.insert(0, "Strategy", name)
            detail_parts.append(d)
        detail_df = pd.concat(detail_parts, ignore_index=True)

        summary_df.to_csv(RESULTS_DIR / "ragas_summary.csv", index=False)
        detail_df.to_csv( RESULTS_DIR / "ragas_detail_all.csv", index=False)

        msg = "RAGAS evaluation complete!\n\n"
        for name, s in all_summaries.items():
            msg += f"── {name} ──\n"
            for k, v in s.items():
                emoji = "🟢" if v >= 0.8 else "🟡" if v >= 0.6 else "🟠" if v >= 0.4 else "🔴"
                msg += f"  {k:<30} {v:.4f} {emoji}\n"
            msg += "\n"
        msg += "Saved to results/ragas_*.csv"
        return msg, summary_df, detail_df
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return f"Error: {str(e)}", None, None


def chat_with_agent(question: str, strategy: str, history: List):
    from src.agent_orchestrator import run_agent
    agent_map = {
        "Hybrid (recommended)":    _state["agent_hybrid"],
        "Vector only (FAISS)":     _state["agent_vector"],
        "BM25 only (lexical)":     _state["agent_bm25"],
    }
    agent = agent_map.get(strategy)
    if agent is None:
        history = (history or []) + [[question, "Please index a PDF before asking questions."]]
        return history, ""
    try:
        r = run_agent(agent, question)
        tools_u = list({s[0].tool for s in r["steps"] if hasattr(s[0], "tool")})
        full = f"{r['answer']}\n\n---\n_Strategy: {strategy} | Tools used: {', '.join(tools_u) or 'direct'} | Steps: {len(r['steps'])}_"
        return (history or []) + [[question, full]], ""
    except Exception as e:
        return (history or []) + [[question, f"Error: {str(e)}"]], ""


def build_ui():
    with gr.Blocks(title="Agentic RAG Benchmark") as demo:
        gr.Markdown("""
# Agentic RAG Benchmark 
### Comparison: BM25 · Vector (FAISS) · Hybrid retrieval with ReAct Agent and RAGAS Evaluation
---
""")

        with gr.Tab("1. Index PDF"):
            gr.Markdown("""
**Three retrieval strategies being compared:**
- **BM25** — Lexical keyword search (advanced full-text search, TF-IDF based)
- **Vector (FAISS)** — Semantic search using embeddings, understands meaning and synonyms
- **Hybrid** — Combines BM25 + Vector using Reciprocal Rank Fusion (RRF) algorithm
""")
            with gr.Row():
                pdf_input = gr.File(label="Upload PDF Book", file_types=[".pdf"])
                index_btn = gr.Button("Index PDF", variant="primary", scale=0)
            index_status = gr.Textbox(label="Status", lines=7, interactive=False)

        with gr.Tab("2. Generate Test Questions"):
            gr.Markdown("""
**The LLM (GPT-4o-mini) reads chunks from the book and for each generates:**
- A question based on the text content
- A ground truth answer used for RAGAS evaluation

Results are saved to `results/test_set.json`
""")
            with gr.Row():
                n_slider = gr.Slider(2, 30, value=10, step=1, label="Number of questions")
                gen_btn  = gr.Button("Generate Questions", variant="primary", scale=0, interactive=False)
            gen_status = gr.Textbox(label="Status", lines=2, interactive=False)
            gen_table  = gr.DataFrame(label="Questions + Ground Truth", wrap=True)

        with gr.Tab("3. Agent Benchmark"):
            gr.Markdown("**All 3 agents answer the same questions — compare answers against ground truth directly**")
            bench_btn    = gr.Button("Run Benchmark (all 3 agents)", variant="primary", interactive=False)
            bench_status = gr.Textbox(label="Status", lines=3, interactive=False)
            with gr.Tabs():
                with gr.Tab("BM25 Answers"):
                    bench_bm25 = gr.DataFrame(label="BM25 Agent Results", wrap=True)
                with gr.Tab("Vector Answers"):
                    bench_vec  = gr.DataFrame(label="Vector Agent Results", wrap=True)
                with gr.Tab("Hybrid Answers"):
                    bench_hyb  = gr.DataFrame(label="Hybrid Agent Results", wrap=True)

        with gr.Tab("4. RAGAS Evaluation"):
            gr.Markdown("""
**Automatic evaluation of all 3 retrieval strategies:**

| Metric | Description |
|---|---|
| Faithfulness | Answer does not hallucinate beyond the retrieved context |
| Answer Relevancy | Answer actually addresses the question asked |
| Context Recall | All necessary documents were retrieved |
| Context Precision | Retrieved documents are relevant (low noise) |
""")
            eval_btn     = gr.Button("Run RAGAS Evaluation", variant="primary", interactive=False)
            eval_status  = gr.Textbox(label="Results", lines=15, interactive=False)
            eval_summary = gr.DataFrame(label="Average Metrics Comparison — BM25 vs Vector vs Hybrid")
            eval_detail  = gr.DataFrame(label="Per-question details", wrap=True)

        with gr.Tab("5. Interactive Q&A"):
            strategy_radio = gr.Radio(
                choices=["Hybrid (recommended)", "Vector only (FAISS)", "BM25 only (lexical)"],
                value="Hybrid (recommended)",
                label="Select retrieval strategy",
            )
            chatbot = gr.Chatbot(label="Conversation", height=420)
            with gr.Row():
                chat_input = gr.Textbox(placeholder="Ask a question about the book...", label="Question", scale=4)
                send_btn   = gr.Button("Send", variant="primary", scale=0)
            gr.Examples(
                examples=[
                    "What is the main topic of this book?",
                    "What are the key concepts described?",
                    "Summarize the main conclusions.",
                ],
                inputs=chat_input,
            )

        index_btn.click(fn=index_pdf, inputs=[pdf_input], outputs=[index_status, gen_btn])
        gen_btn.click(fn=generate_questions, inputs=[n_slider], outputs=[gen_status, gen_table, bench_btn])
        bench_btn.click(fn=run_benchmark, inputs=[], outputs=[bench_status, bench_bm25, bench_vec, bench_hyb, eval_btn])
        eval_btn.click(fn=run_evaluation, inputs=[], outputs=[eval_status, eval_summary, eval_detail])
        send_btn.click(fn=chat_with_agent, inputs=[chat_input, strategy_radio, chatbot], outputs=[chatbot, chat_input])
        chat_input.submit(fn=chat_with_agent, inputs=[chat_input, strategy_radio, chatbot], outputs=[chatbot, chat_input])

    return demo


if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    ui = build_ui()
    ui.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="blue"),
    )