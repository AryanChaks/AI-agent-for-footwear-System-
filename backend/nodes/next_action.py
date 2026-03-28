"""
Node 6 — Next Action
CRM-style decision engine. For each lead, recommends the
next best sales action based on tier, score, and reply signals.
"""

from typing import Dict, Any, List


ACTION_RULES = {
    "Hot": {
        "action": "Schedule Meeting",
        "urgency": "High",
        "reasoning": "Top-tier fit. Prioritise a discovery call within 48 hours.",
    },
    "Warm": {
        "action": "Follow-up",
        "urgency": "Medium",
        "reasoning": "Good potential. Send follow-up in 5 days if no reply.",
    },
    "Cold": {
        "action": "Deprioritize",
        "urgency": "Low",
        "reasoning": "Poor fit for current expansion goals. Park for 6 months.",
    },
}

SENTIMENT_OVERRIDE = {
    "Positive": {
        "action": "Schedule Meeting",
        "urgency": "High",
        "reasoning": "Positive response received. Move to meeting immediately.",
    },
    "Objection": {
        "action": "Follow-up",
        "urgency": "Medium",
        "reasoning": "Objection raised. Send tailored response addressing concerns.",
    },
    "Negative": {
        "action": "Deprioritize",
        "urgency": "Low",
        "reasoning": "Negative response. Remove from active pipeline.",
    },
    "Escalate": {
        "action": "Escalate to Senior",
        "urgency": "High",
        "reasoning": "High-value lead requires senior sales rep involvement.",
    },
}


def next_action_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combine priority tier + reply sentiment (if available)
    to produce the final recommended action per lead.
    """
    leads = state.get("qualified_leads", [])
    reply_analyses = state.get("reply_analyses", [])

    # Build reply lookup by dealer_id
    reply_map = {r["dealer_id"]: r for r in reply_analyses}

    next_actions = []
    for lead in leads:
        dealer_id = lead["dealer"]["dealer_id"]
        tier = lead["priority_tier"]
        score = lead["fit_score"]

        # Start from tier-based rule
        base = ACTION_RULES[tier].copy()

        # Override if we have a reply signal
        if dealer_id in reply_map:
            sentiment = reply_map[dealer_id].get("sentiment", "Neutral")
            if sentiment in SENTIMENT_OVERRIDE:
                override = SENTIMENT_OVERRIDE[sentiment]
                base.update(override)

        # Escalate very high-scoring Hot leads
        if tier == "Hot" and score >= 88:
            base["action"] = "Escalate to Senior"
            base["urgency"] = "High"
            base["reasoning"] = (
                f"Exceptional fit score ({score}/100). "
                "Flag for senior BDM involvement."
            )

        next_actions.append({
            "dealer_id": dealer_id,
            "priority_tier": tier,
            "fit_score": score,
            "recommended_action": base["action"],
            "urgency": base["urgency"],
            "reasoning": base["reasoning"],
        })

    state["next_actions"] = next_actions
    print(f"[NextAction] Actions assigned for {len(next_actions)} leads.")
    return state
