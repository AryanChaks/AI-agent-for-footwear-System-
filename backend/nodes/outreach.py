"""
Node 4 — Outreach
Generates personalised outreach emails for Hot/Warm leads.
Compatible with LangChain 0.2+ / Zephyr prompt format.
"""

from typing import Dict, Any
from backend.llm import get_llm
from backend.rag.vectorstore import get_retriever
from langchain_core.prompts import PromptTemplate

OUTREACH_PROMPT = PromptTemplate(
    input_variables=["dealer_info", "brand_context", "fit_reasons"],
    template="""You are a business development manager for StepForward Footwear.
Write a short, professional outreach email to a potential distributor.

Brand context:
{brand_context}

Distributor profile:
{dealer_info}

Why they are a good fit:
{fit_reasons}

Instructions:
- Keep it under 180 words
- Start with a specific hook about their business
- Mention the margin opportunity clearly (25-40%)
- End with a single clear call to action (15-min call)
- Do NOT use generic phrases like "I hope this email finds you well"
- Output format: Subject: <subject line> then a blank line then the email body

Email:"""
)


def outreach_node(state: Dict[str, Any]) -> Dict[str, Any]:
    leads = state.get("qualified_leads", [])
    if not leads:
        state["error"] = "No leads for outreach."
        return state

    llm = get_llm(temperature=0.6, max_new_tokens=400)
    retriever = get_retriever(k=2)
    chain = OUTREACH_PROMPT | llm

    emails = []
    for lead in leads:
        if lead["priority_tier"] == "Cold":
            continue

        dealer = lead["dealer"]
        docs = retriever.invoke(
            f"outreach email distributor {dealer['top_product']} {dealer['location']}"
        )
        brand_context = "\n".join(d.page_content for d in docs)

        dealer_info = (
            f"Company: {dealer['dealer_id']}, "
            f"Location: {dealer['location']}, {dealer['country']}, "
            f"Specialises in: {', '.join(dealer['product_categories'])}, "
            f"Current brands: {', '.join(dealer['brands_carried'])}, "
            f"Annual revenue: ${dealer['total_revenue']:,.0f}, "
            f"Market presence: {dealer['years_active']} years"
        )
        fit_text = "; ".join(lead["fit_reasons"])

        try:
            raw_result = chain.invoke({
                "dealer_info": dealer_info,
                "brand_context": brand_context,
                "fit_reasons": fit_text,
            })
            raw = raw_result.content if hasattr(raw_result, "content") else str(raw_result)
            raw = raw.strip()

            if "Subject:" in raw:
                parts = raw.split("\n\n", 1)
                subject = parts[0].replace("Subject:", "").strip()
                body = parts[1].strip() if len(parts) > 1 else raw
            else:
                subject = f"Partnership Opportunity — StepForward Footwear x {dealer['dealer_id']}"
                body = raw
        except Exception as e:
            subject = "Partnership Opportunity — StepForward Footwear"
            body = f"Email generation failed: {e}"

        emails.append({
            "dealer_id": dealer["dealer_id"],
            "priority_tier": lead["priority_tier"],
            "subject": subject,
            "body": body,
        })

    state["outreach_emails"] = emails
    print(f"[Outreach] Generated {len(emails)} emails (Hot + Warm leads).")
    return state
