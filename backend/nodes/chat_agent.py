"""
Chat Agent — intent router + general chat.
Compatible with LangChain 0.2+ / Zephyr prompt format.
"""

from typing import Dict, Any
from backend.llm import get_llm
from backend.rag.vectorstore import get_retriever
from langchain_core.prompts import PromptTemplate

INTENT_PROMPT = PromptTemplate(
    input_variables=["query"],
    template="""Classify this sales agent query into exactly one of these intents:
prospect | qualify | outreach | reply | nextaction | general

Query: {query}

Rules:
- prospect: finding or searching for distributors
- qualify: scoring, assessing, or evaluating distributors
- outreach: writing emails, messages, or communication
- reply: handling a distributor's response or objection
- nextaction: deciding what to do next with a lead
- general: anything else about the brand, market, or strategy

Respond with only the single intent word, nothing else.
Intent:"""
)

GENERAL_PROMPT = PromptTemplate(
    input_variables=["query", "context"],
    template="""You are an AI sales agent for StepForward Footwear helping with global distributor expansion.

Relevant brand and market context:
{context}

User question: {query}

Give a concise, commercially useful answer in 3-5 sentences.
Answer:"""
)


def classify_intent(query: str) -> str:
    llm = get_llm(temperature=0.1, max_new_tokens=10)
    chain = INTENT_PROMPT | llm
    try:
        raw = chain.invoke({"query": query})
        text = raw.content if hasattr(raw, "content") else str(raw)
        intent = text.strip().lower().split()[0]
        valid = {"prospect", "qualify", "outreach", "reply", "nextaction", "general"}
        return intent if intent in valid else "general"
    except Exception:
        return "general"


def chat_general(query: str) -> str:
    retriever = get_retriever(k=3)
    docs = retriever.invoke(query)
    context = "\n".join(d.page_content for d in docs)
    llm = get_llm(temperature=0.4, max_new_tokens=300)
    chain = GENERAL_PROMPT | llm
    try:
        raw = chain.invoke({"query": query, "context": context})
        return (raw.content if hasattr(raw, "content") else str(raw)).strip()
    except Exception as e:
        return f"Sorry, I couldn't process that query: {e}"


def chat_node(state: Dict[str, Any]) -> Dict[str, Any]:
    history = state.get("chat_history", [])
    if not history:
        return state
    latest = history[-1]
    if latest.get("role") != "user":
        return state
    query = latest["content"]
    intent = classify_intent(query)
    response = chat_general(query)
    history.append({"role": "assistant", "content": response, "intent": intent})
    state["chat_history"] = history
    return state
