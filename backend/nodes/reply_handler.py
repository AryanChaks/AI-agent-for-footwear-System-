"""
Node 5 — Reply Handler
Analyses a distributor reply, classifies sentiment, suggests a response.
Compatible with LangChain 0.2+ / Zephyr prompt format.
"""

from typing import Dict, Any
from backend.llm import get_llm
from backend.rag.vectorstore import get_retriever
from langchain_core.prompts import PromptTemplate

REPLY_PROMPT = PromptTemplate(
    input_variables=["reply_text", "brand_context"],
    template="""You are a senior business development manager for StepForward Footwear.
A potential distributor has replied to your outreach. Analyse their reply and respond.

Brand context:
{brand_context}

Distributor reply:
\"\"\"{reply_text}\"\"\"

Provide your output in this exact format:
SENTIMENT: <Positive / Neutral / Objection / Negative>
NEXT_ACTION: <Follow-up / Schedule Meeting / Escalate to Senior / Deprioritize>
SUGGESTED_RESPONSE:
<Your 100-150 word professional reply addressing their specific points>

Output:"""
)


def _parse_reply_output(raw: str) -> tuple:
    sentiment = "Neutral"
    next_action = "Follow-up"
    response_lines = []
    in_response = False

    for line in raw.strip().split("\n"):
        if line.startswith("SENTIMENT:"):
            sentiment = line.replace("SENTIMENT:", "").strip()
        elif line.startswith("NEXT_ACTION:"):
            next_action = line.replace("NEXT_ACTION:", "").strip()
        elif line.startswith("SUGGESTED_RESPONSE:"):
            in_response = True
        elif in_response:
            response_lines.append(line)

    response = "\n".join(response_lines).strip() if response_lines else raw.strip()
    return sentiment, next_action, response


def reply_handler_node(state: Dict[str, Any]) -> Dict[str, Any]:
    pending_replies = state.get("pending_replies", [])
    if not pending_replies:
        state["reply_analyses"] = []
        return state

    llm = get_llm(temperature=0.4, max_new_tokens=350)
    retriever = get_retriever(k=3)
    chain = REPLY_PROMPT | llm

    analyses = []
    for item in pending_replies:
        dealer_id = item.get("dealer_id", "Unknown")
        reply_text = item.get("reply_text", "")

        docs = retriever.invoke(f"distributor objection response {reply_text[:100]}")
        brand_context = "\n".join(d.page_content for d in docs)

        try:
            raw_result = chain.invoke({"reply_text": reply_text, "brand_context": brand_context})
            raw = raw_result.content if hasattr(raw_result, "content") else str(raw_result)
            sentiment, next_action, suggested_response = _parse_reply_output(raw)
        except Exception as e:
            sentiment = "Unknown"
            next_action = "Follow-up"
            suggested_response = f"Response generation failed: {e}"

        analyses.append({
            "dealer_id": dealer_id,
            "original_reply": reply_text,
            "sentiment": sentiment,
            "suggested_response": suggested_response,
            "next_action": next_action,
        })

    state["reply_analyses"] = analyses
    print(f"[ReplyHandler] Processed {len(analyses)} replies.")
    return state


def handle_single_reply(dealer_id: str, reply_text: str) -> Dict[str, Any]:
    state = {"pending_replies": [{"dealer_id": dealer_id, "reply_text": reply_text}]}
    result = reply_handler_node(state)
    if result["reply_analyses"]:
        return result["reply_analyses"][0]
    return {"error": "No analysis produced."}
