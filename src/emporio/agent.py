"""Builds the customer service agent and runs one conversation turn.

`create_agent` (LangChain 1.x, built on LangGraph) gives us the ReAct loop:
the model reasons, calls tools, reads their results and repeats until it can
answer. `InMemorySaver` keeps the conversation history per `thread_id`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver

from emporio.prompts import build_system_prompt
from emporio.tools import ALL_TOOLS

# Lite model: separate free-tier quota, fast, and passes the E2E suite.
DEFAULT_MODEL = "gemini-3.5-flash-lite"


def build_agent(model_name: str | None = None):
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("Missing GOOGLE_API_KEY. Copy .env.example to .env and add your key.")
    model = ChatGoogleGenerativeAI(
        model=model_name or os.getenv("GEMINI_MODEL") or DEFAULT_MODEL,
        # No temperature: Gemini 3.x uses fixed sampling defaults and ignores it.
        max_retries=3,
    )
    return create_agent(
        model,
        tools=ALL_TOOLS,
        system_prompt=build_system_prompt(),
        checkpointer=InMemorySaver(),
    )


@dataclass
class Turn:
    reply: str
    tool_calls: list[dict] = field(default_factory=list)  # [{"name", "args", "result"}]


def chat(agent, thread_id: str, message: str) -> Turn:
    config = {"configurable": {"thread_id": thread_id}}
    before = len(agent.get_state(config).values.get("messages", []))
    state = agent.invoke({"messages": [{"role": "user", "content": message}]}, config=config)
    new_messages = state["messages"][before + 1 :]  # skip the user message we just sent

    results = {m.tool_call_id: m.content for m in new_messages if isinstance(m, ToolMessage)}
    calls = [
        {"name": c["name"], "args": c["args"], "result": results.get(c["id"])}
        for m in new_messages
        if isinstance(m, AIMessage)
        for c in m.tool_calls
    ]
    return Turn(reply=state["messages"][-1].text, tool_calls=calls)
