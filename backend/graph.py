"""
LangGraph workflow assembly.
Connects all 6 nodes into a stateful directed graph.

Flow:
  prospect → qualify → prioritize → outreach → next_action → END
  reply_handler can be invoked independently via the API.
  chat_node is invoked independently via the API.
"""

from typing import TypedDict, List, Any, Optional
from langgraph.graph import StateGraph, END

from backend.nodes.prospect import prospect_node
from backend.nodes.qualify import qualify_node
from backend.nodes.prioritize import prioritize_node
from backend.nodes.outreach import outreach_node
from backend.nodes.next_action import next_action_node
from backend.nodes.reply_handler import reply_handler_node


# ── State schema ──────────────────────────────────────────────────────────────
class GraphState(TypedDict):
    dealers: List[Any]
    qualified_leads: List[Any]
    outreach_emails: List[Any]
    next_actions: List[Any]
    reply_analyses: List[Any]
    pending_replies: List[Any]
    chat_history: List[Any]
    region_filter: Optional[str]
    error: Optional[str]


# ── Build graph ───────────────────────────────────────────────────────────────
def build_pipeline() -> StateGraph:
    """Build and compile the main sales pipeline graph."""
    graph = StateGraph(GraphState)

    graph.add_node("prospect", prospect_node)
    graph.add_node("qualify", qualify_node)
    graph.add_node("prioritize", prioritize_node)
    graph.add_node("outreach", outreach_node)
    graph.add_node("next_action", next_action_node)
    graph.add_node("reply_handler", reply_handler_node)

    graph.set_entry_point("prospect")
    graph.add_edge("prospect", "qualify")
    graph.add_edge("qualify", "prioritize")
    graph.add_edge("prioritize", "outreach")
    graph.add_edge("outreach", "next_action")
    graph.add_edge("next_action", END)

    return graph.compile()


def build_reply_graph() -> StateGraph:
    """Standalone graph for the reply handler (called independently)."""
    graph = StateGraph(GraphState)
    graph.add_node("reply_handler", reply_handler_node)
    graph.set_entry_point("reply_handler")
    graph.add_edge("reply_handler", END)
    return graph.compile()


# Compile once at import time
pipeline = build_pipeline()
reply_pipeline = build_reply_graph()


def run_pipeline(region_filter: str = None) -> GraphState:
    """Run the full prospect → next_action pipeline."""
    initial_state: GraphState = {
        "dealers": [],
        "qualified_leads": [],
        "outreach_emails": [],
        "next_actions": [],
        "reply_analyses": [],
        "pending_replies": [],
        "chat_history": [],
        "region_filter": region_filter,
        "error": None,
    }
    return pipeline.invoke(initial_state)


def run_reply_handler(dealer_id: str, reply_text: str) -> GraphState:
    """Run just the reply handler for a single distributor reply."""
    initial_state: GraphState = {
        "dealers": [],
        "qualified_leads": [],
        "outreach_emails": [],
        "next_actions": [],
        "reply_analyses": [],
        "pending_replies": [{"dealer_id": dealer_id, "reply_text": reply_text}],
        "chat_history": [],
        "region_filter": None,
        "error": None,
    }
    return reply_pipeline.invoke(initial_state)
