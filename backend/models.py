"""
Pydantic schemas shared across backend nodes and API.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class Dealer(BaseModel):
    dealer_id: str
    location: str
    country: str
    region: str
    brands_carried: List[str]
    product_categories: List[str]
    total_revenue: float
    avg_margin_pct: float
    total_quantity_sold: int
    avg_stock_level: float
    top_product: str
    years_active: int
    website: Optional[str] = None
    company_size: str


class QualifiedLead(BaseModel):
    dealer: Dealer
    fit_score: float            # 0-100
    priority_tier: str          # Hot / Warm / Cold
    fit_reasons: List[str]
    risk_flags: List[str]
    rag_context: Optional[str] = None


class OutreachEmail(BaseModel):
    dealer_id: str
    subject: str
    body: str


class ReplyAnalysis(BaseModel):
    dealer_id: str
    original_reply: str
    sentiment: str              # Positive / Neutral / Objection / Negative
    suggested_response: str
    next_action: str


class NextAction(BaseModel):
    dealer_id: str
    priority_tier: str
    recommended_action: str     # Follow-up / Meeting / Escalate / Deprioritize
    reasoning: str
    urgency: str                # High / Medium / Low


class AgentState(BaseModel):
    """LangGraph state object passed between nodes."""
    dealers: List[Dict[str, Any]] = []
    qualified_leads: List[Dict[str, Any]] = []
    outreach_emails: List[Dict[str, Any]] = []
    next_actions: List[Dict[str, Any]] = []
    reply_analyses: List[Dict[str, Any]] = []
    chat_history: List[Dict[str, str]] = []
    error: Optional[str] = None
