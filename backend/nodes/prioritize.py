"""
Node 3 — Prioritize
Assigns Hot / Warm / Cold using percentile-based cutoffs
so tiers always have a meaningful spread regardless of pool size.
"""

from typing import Dict, Any, List


def _percentile_thresholds(leads: List[Dict]) -> tuple:
    """
    Top 25% = Hot, middle 50% = Warm, bottom 25% = Cold.
    Guarantees spread even when all scores are close together.
    """
    scores = sorted([l["fit_score"] for l in leads])
    n = len(scores)
    hot_cutoff  = scores[int(n * 0.75)]   # top 25%
    cold_cutoff = scores[int(n * 0.25)]   # bottom 25%
    return hot_cutoff, cold_cutoff


def _assign_tier(score: float, hot_cutoff: float, cold_cutoff: float) -> str:
    if score >= hot_cutoff:
        return "Hot"
    elif score > cold_cutoff:
        return "Warm"
    return "Cold"


def prioritize_node(state: Dict[str, Any]) -> Dict[str, Any]:
    leads = state.get("qualified_leads", [])
    if not leads:
        state["error"] = "No qualified leads to prioritize."
        return state

    # Need at least 3 dealers for percentile split to be meaningful
    if len(leads) < 3:
        for lead in leads:
            lead["priority_tier"] = "Warm"
    else:
        hot_cutoff, cold_cutoff = _percentile_thresholds(leads)
        for lead in leads:
            lead["priority_tier"] = _assign_tier(
                lead["fit_score"], hot_cutoff, cold_cutoff
            )

    tier_order = {"Hot": 0, "Warm": 1, "Cold": 2}
    leads.sort(key=lambda x: (tier_order[x["priority_tier"]], -x["fit_score"]))
    state["qualified_leads"] = leads

    counts = {t: sum(1 for l in leads if l["priority_tier"] == t) for t in tier_order}
    print(f"[Prioritize] Hot: {counts['Hot']} | Warm: {counts['Warm']} | Cold: {counts['Cold']}")
    return state
