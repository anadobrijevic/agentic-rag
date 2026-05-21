"""
agent_orchestrator.py - ReAct agent sa tri moda: hybrid / vector / bm25
"""

import logging
from typing import List, Dict, Any

from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.agents import AgentExecutor, create_react_agent

from src.config import get_llm

logger = logging.getLogger(__name__)

REACT_TEMPLATE = """Answer the following question using the available tools. You MUST follow this EXACT format every single time:

Thought: I need to search for information about this question.
Action: {tool_names_list}
Action Input: the search query
Observation: [tool result will appear here]
Thought: I now have enough information to answer.
Final Answer: [your answer based only on the documents]

STRICT RULES:
- You MUST use exactly one of these tool names: {tool_names}
- Action Input must be a plain string query, nothing else
- After at most 2 Observation steps, you MUST write "Final Answer:"
- If the documents don't contain the answer, write: Final Answer: The document does not contain information about this topic.
- Never say "Agent stopped" — always provide a Final Answer.

Tools available:
{tools}

Question: {input}

{agent_scratchpad}"""


def _first_tool_name(tools: List[Tool]) -> str:
    """Vraća naziv prvog alata za primer u promptu."""
    return tools[0].name if tools else "search"


def build_agent(tools: List[Tool], mode: str = "hybrid") -> AgentExecutor:
    llm = get_llm(temperature=0.0)

    tool_names_list = tools[0].name if len(tools) == 1 else " or ".join(t.name for t in tools)

    template = REACT_TEMPLATE.replace("{tool_names_list}", tool_names_list)
    prompt = PromptTemplate.from_template(template)

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    def handle_error(error) -> str:
        """Kad agent ne može da parsira, vrati uputstvo za ispravan format."""
        return (
            "Parsing error. You MUST respond in this exact format:\n"
            "Thought: [your reasoning]\n"
            f"Action: {tools[0].name}\n"
            "Action Input: [your search query]\n"
        )

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=6,
        max_execution_time=120,
        handle_parsing_errors=handle_error,
        return_intermediate_steps=True,
        early_stopping_method="generate",
    )
    logger.info(f"Agent [{mode}] spreman sa alatima: {[t.name for t in tools]}")
    return executor


def run_agent(executor: AgentExecutor, question: str) -> Dict[str, Any]:
    try:
        result  = executor.invoke({"input": question})
        answer  = result.get("output", "")

        # Ako agent stane bez odgovora — izvuci iz poslednjeg Observation koraka
        if not answer or "stopped" in answer.lower() or "iteration limit" in answer.lower():
            steps = result.get("intermediate_steps", [])
            if steps:
                last_obs = steps[-1][1] if len(steps[-1]) >= 2 else ""
                if isinstance(last_obs, str) and len(last_obs) > 30:
                    answer = f"[Izvučeno iz poslednjeg konteksta]\n{last_obs[:800]}"
                else:
                    answer = "Agent nije uspeo da formira odgovor. Pokušaj ponovo ili promeni pitanje."
            else:
                answer = "Agent nije pronašao relevantan kontekst za ovo pitanje."

        contexts = []
        for step in result.get("intermediate_steps", []):
            if len(step) >= 2 and isinstance(step[1], str) and len(step[1]) > 20:
                contexts.append(step[1])

        return {
            "answer":   answer,
            "contexts": contexts or ["Nije pronađen kontekst."],
            "steps":    result.get("intermediate_steps", []),
        }
    except Exception as e:
        logger.error(f"Greška u agentu: {e}")
        return {"answer": f"Greška: {str(e)}", "contexts": [], "steps": []}