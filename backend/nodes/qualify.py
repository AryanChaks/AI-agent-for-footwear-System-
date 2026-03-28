"""
Node 2 — Qualify
Scores each dealer using rule-based pool-relative normalisation + RAG + LLM narrative.
Compatible with LangChain 0.2+ / Zephyr prompt format.
"""

from typing import Dict, Any, List, Tuple
from backend.llm import get_llm
from backend.rag.vectorstore import get_retriever
from langchain_core.prompts import PromptTemplate

QUALIFY_PROMPT = PromptTemplate(
    input_variables=["dealer_info", "brand_context"],
    template="""You are a distributor qualification analyst for StepForward Footwear.

Brand context:
{brand_context}

Dealer profile:
{dealer_info}

In 2-3 sentences, assess this dealer's fit as a StepForward distributor.
Mention category alignment, margin compatibility, and any risks.
Be concise and commercially specific.

Assessment:"""
)


def _raw_scores(dealer: Dict[str, Any]) -> Dict[str, float]:
    cats = {c.lower() for c in dealer.get("product_categories", [])}
    good_cats = {"flats", "casual", "lifestyle", "comfort", "formal",
                 "sandals", "loafers", "kids", "school", "walking"}
    bad_cats = {"sports", "running", "training", "performance"}
    good_overlap = len(cats & good_cats)
    bad_overlap = len(cats & bad_cats)

    if good_overlap >= 2:
        cat_score = 2.0
    elif good_overlap == 1:
        cat_score = 1.0
    elif bad_overlap >= 2:
        cat_score = 0.0
    else:
        cat_score = 0.5

    return {
        "years_active": float(dealer.get("years_active", 0)),
        "category_fit": cat_score,
        "margin_pct": float(dealer.get("avg_margin_pct", 0)),
        "revenue": float(dealer.get("total_revenue", 0)),
        "stock_level": float(dealer.get("avg_stock_level", 0)),
    }


def _normalise_pool(dealers: List[Dict[str, Any]]) -> List[Tuple[float, List[str], List[str]]]:
    raw = [_raw_scores(d) for d in dealers]
    dims = ["years_active", "category_fit", "margin_pct", "revenue", "stock_level"]
    weights = {
        "years_active": 15,
        "category_fit": 30,
        "margin_pct":   20,
        "revenue":      20,
        "stock_level":  15,
    }

    minmax = {}
    for d in dims:
        vals = [r[d] for r in raw]
        lo, hi = min(vals), max(vals)
        minmax[d] = (lo, hi)

    results = []
    for i, dealer in enumerate(dealers):
        r = raw[i]
        score = 0.0
        reasons = []
        risks = []

        for d in dims:
            lo, hi = minmax[d]
            norm = 0.5 if hi == lo else (r[d] - lo) / (hi - lo)
            score += norm * weights[d]

        years = dealer.get("years_active", 0)
        if years >= 6:
            reasons.append(f"Established {years}-year market presence")
        elif years >= 3:
            reasons.append(f"Growing {years}-year presence")
        else:
            risks.append("Less than 3 years active — early stage")

        cats = {c.lower() for c in dealer.get("product_categories", [])}
        good_cats = {"flats", "casual", "lifestyle", "comfort", "formal",
                     "sandals", "loafers", "kids", "school", "walking"}
        bad_cats = {"sports", "running", "training", "performance"}
        good_overlap = cats & good_cats
        bad_overlap = cats & bad_cats
        if len(good_overlap) >= 2:
            reasons.append(f"Strong category fit ({', '.join(sorted(good_overlap))})")
        elif len(good_overlap) == 1:
            reasons.append(f"Partial category fit ({', '.join(good_overlap)})")
        elif len(bad_overlap) >= 2:
            risks.append("Primarily sports/performance — poor category fit")

        margin = dealer.get("avg_margin_pct", 0)
        if margin >= 35:
            reasons.append(f"High margin expectation ({margin:.1f}%) — aligns with our 25-40% offer")
        elif margin >= 25:
            reasons.append(f"Margin expectation ({margin:.1f}%) aligns well")
        elif margin >= 15:
            reasons.append(f"Moderate margin expectation ({margin:.1f}%)")
        else:
            risks.append(f"Very low margin expectation ({margin:.1f}%) — may resist our pricing")

        rev = dealer.get("total_revenue", 0)
        if rev >= 50_000_000:
            risks.append(f"Very large distributor (${rev:,.0f}) — may deprioritise new brand")
        elif rev >= 10_000_000:
            reasons.append(f"Strong revenue base (${rev:,.0f})")
        elif rev >= 1_000_000:
            reasons.append(f"Healthy revenue (${rev:,.0f})")
        else:
            risks.append(f"Low revenue (${rev:,.0f}) — limited market reach")

        stock = dealer.get("avg_stock_level", 0)
        if stock >= 100:
            reasons.append(f"Excellent stock management (avg {stock:.0f} units)")
        elif stock >= 50:
            reasons.append(f"Adequate stock levels (avg {stock:.0f} units)")
        else:
            risks.append(f"Low average stock ({stock:.0f} units) — logistics risk")

        results.append((round(score, 1), reasons, risks))

    return results


def qualify_node(state: Dict[str, Any]) -> Dict[str, Any]:
    dealers = state.get("dealers", [])
    if not dealers:
        state["error"] = "No dealers to qualify."
        return state

    llm = get_llm(temperature=0.3)
    retriever = get_retriever(k=3)
    chain = QUALIFY_PROMPT | llm
    scored = _normalise_pool(dealers)

    qualified = []
    for dealer, (score, reasons, risks) in zip(dealers, scored):
        query = f"distributor fit for {dealer['top_product']} {dealer['product_categories']} dealer"
        docs = retriever.invoke(query)
        brand_context = "\n".join(d.page_content for d in docs)

        dealer_info = (
            f"ID: {dealer['dealer_id']}, Location: {dealer['location']}, "
            f"Categories: {dealer['product_categories']}, "
            f"Brands: {dealer['brands_carried']}, "
            f"Revenue: ${dealer['total_revenue']:,.0f}, "
            f"Margin: {dealer['avg_margin_pct']}%, "
            f"Years active: {dealer['years_active']}, "
            f"Stock: {dealer['avg_stock_level']} avg units"
        )

        try:
            raw = chain.invoke({"dealer_info": dealer_info, "brand_context": brand_context})
            narrative = raw.content if hasattr(raw, "content") else str(raw)
            narrative = narrative.strip()
        except Exception as e:
            narrative = f"Assessment unavailable: {e}"

        qualified.append({
            "dealer": dealer,
            "fit_score": score,
            "fit_reasons": reasons,
            "risk_flags": risks,
            "rag_context": brand_context[:300],
            "llm_narrative": narrative,
        })

    state["qualified_leads"] = qualified
    print(f"[Qualify] Scored {len(qualified)} dealers.")
    return state
