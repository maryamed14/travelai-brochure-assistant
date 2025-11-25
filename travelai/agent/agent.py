from __future__ import annotations

from typing import List

from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI

from .tools import BrochureSearchTool


def build_travel_agent(model_name: str = "gpt-4o-mini"):
    """
    Build a LangChain agent that:
    - Uses ReAct reasoning (ZERO_SHOT_REACT_DESCRIPTION).
    - Can decide when to call the brochure_search tool.
    - Uses a small max_iterations to avoid tool loops.
    """
    llm = ChatOpenAI(model=model_name, temperature=0.2)

    tools: List[BrochureSearchTool] = [BrochureSearchTool()]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,          # show Thought/Action/Observation in logs
        max_iterations=5,      # allow a few steps, then force a final answer
        early_stopping_method="generate",  # on limit, generate an answer instead of error
    )

    return agent
