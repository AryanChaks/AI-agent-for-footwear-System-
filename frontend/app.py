"""
Streamlit Frontend — StepForward Footwear Distributor Sales Agent
Tabs: Upload | Prospect | Qualify & Rank | Outreach | Next Action | Chat
"""

import streamlit as st
import requests
import pandas as pd
import json
from typing import Optional

# ── Config ────────────────────────────────────────────────────────────────────
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="StepForward — Distributor Agent",
    page_icon="👟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fb; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1a1a2e;
        color: white;
    }
    .tier-hot {
        background: #fff0f0; border-left: 4px solid #e74c3c;
        padding: 12px; border-radius: 6px; margin: 6px 0;
    }
    .tier-warm {
        background: #fff8f0; border-left: 4px solid #f39c12;
        padding: 12px; border-radius: 6px; margin: 6px 0;
    }
    .tier-cold {
        background: #f0f4ff; border-left: 4px solid #95a5a6;
        padding: 12px; border-radius: 6px; margin: 6px 0;
    }
    .metric-card {
        background: white; border-radius: 10px; padding: 16px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08); text-align: center;
    }
    .chat-user {
        background: #1a1a2e; color: white;
        padding: 10px 14px; border-radius: 12px 12px 2px 12px;
        margin: 6px 0; max-width: 80%; margin-left: auto;
        font-size: 14px;
    }
    .chat-assistant {
        background: white; color: #1a1a2e;
        padding: 10px 14px; border-radius: 12px 12px 12px 2px;
        margin: 6px 0; max-width: 85%;
        border: 1px solid #e8ecf0; font-size: 14px;
    }
    .intent-badge {
        display: inline-block; background: #e8f4fd; color: #1a6bb5;
        border-radius: 12px; padding: 2px 10px; font-size: 12px;
        font-weight: 500; margin-top: 4px;
    }
    .action-escalate { color: #c0392b; font-weight: 600; }
    .action-meeting  { color: #27ae60; font-weight: 600; }
    .action-followup { color: #e67e22; font-weight: 600; }
    .action-deprior  { color: #95a5a6; font-weight: 600; }
    .email-box {
        background: #fff; border: 1px solid #e0e6ed;
        border-radius: 8px; padding: 16px; font-size: 13px;
        line-height: 1.7; white-space: pre-wrap;
    }
    .section-header {
        font-size: 13px; font-weight: 600; color: #666;
        text-transform: uppercase; letter-spacing: 0.5px;
        margin: 16px 0 8px;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
for key in ["pipeline_result", "chat_history", "reply_result"]:
    if key not in st.session_state:
        st.session_state[key] = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []


# ── Helpers ───────────────────────────────────────────────────────────────────
def api_post(endpoint: str, payload: dict = None, files=None):
    try:
        if files:
            r = requests.post(f"{API_URL}{endpoint}", files=files, timeout=120)
        else:
            r = requests.post(f"{API_URL}{endpoint}", json=payload or {}, timeout=180)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to backend. Make sure FastAPI is running."
    except Exception as e:
        return None, str(e)


def api_get(endpoint: str):
    try:
        r = requests.get(f"{API_URL}{endpoint}", timeout=30)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to backend."
    except Exception as e:
        return None, str(e)


def tier_badge(tier: str) -> str:
    colors = {"Hot": "#e74c3c", "Warm": "#f39c12", "Cold": "#95a5a6"}
    return f'<span style="background:{colors.get(tier,"#ccc")};color:white;padding:2px 10px;border-radius:10px;font-size:12px;font-weight:600">{tier}</span>'


def action_style(action: str) -> str:
    if "Escalate" in action:
        return f'<span class="action-escalate">⬆ {action}</span>'
    if "Meeting" in action:
        return f'<span class="action-meeting">📅 {action}</span>'
    if "Follow" in action:
        return f'<span class="action-followup">🔁 {action}</span>'
    return f'<span class="action-deprior">⏸ {action}</span>'


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 👟 StepForward")
    st.markdown("**Distributor Sales Agent**")
    st.markdown("---")

    st.markdown("### 🔧 Pipeline Controls")
    region_filter = st.selectbox(
        "Target Region",
        ["All Regions", "Southeast Asia", "South Asia", "Middle East"],
    )
    region_val = None if region_filter == "All Regions" else region_filter

    run_btn = st.button("▶ Run Full Pipeline", type="primary", use_container_width=True)
    if run_btn:
        with st.spinner("Running LangGraph pipeline... (this may take 60-90s)"):
            result, err = api_post("/run-pipeline", {"region_filter": region_val})
            if err:
                st.error(err)
            else:
                st.session_state.pipeline_result = result
                st.success("Pipeline complete!")

    st.markdown("---")
    st.markdown("### 📊 Pipeline Status")
    if st.session_state.pipeline_result:
        r = st.session_state.pipeline_result
        leads = r.get("qualified_leads", [])
        hot = sum(1 for l in leads if l["priority_tier"] == "Hot")
        warm = sum(1 for l in leads if l["priority_tier"] == "Warm")
        cold = sum(1 for l in leads if l["priority_tier"] == "Cold")
        emails = len(r.get("outreach_emails", []))
        st.metric("Total Leads", len(leads))
        c1, c2, c3 = st.columns(3)
        c1.metric("🔴 Hot", hot)
        c2.metric("🟡 Warm", warm)
        c3.metric("⚪ Cold", cold)
        st.metric("Emails Drafted", emails)
    else:
        st.info("Run the pipeline to see stats.")

    st.markdown("---")
    st.caption("Model: Mistral-7B-Instruct-v0.2\nEmbeddings: all-MiniLM-L6-v2\nStore: FAISS (local)")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_upload, tab_prospect, tab_qualify, tab_outreach, tab_action, tab_chat = st.tabs([
    "📁 Upload Data",
    "🔍 Prospect",
    "📊 Qualify & Rank",
    "✉️ Outreach",
    "🎯 Next Action",
    "💬 Chat",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
with tab_upload:
    st.header("📁 Upload Dealer Data")
    st.markdown(
        "Upload your Kaggle footwear sales CSV to populate the dealer database. "
        "The system will aggregate sales data per dealer and prepare them for prospecting."
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("##### Expected CSV columns")
        st.markdown("""
        | Column | Description |
        |---|---|
        | `Dealer` | Dealer ID |
        | `Dealer Location` | City/area |
        | `Brand` | Brand carried |
        | `Product` | Product category |
        | `Total Revenue (₹)` | Revenue per row |
        | `Margin (%)` | Margin percentage |
        | `Quantity Sold` | Units sold |
        | `Stock Availability` | Stock on hand |
        """)

    with col2:
        uploaded = st.file_uploader("Choose CSV file", type=["csv"])
        if uploaded:
            df_preview = pd.read_csv(uploaded)
            st.markdown(f"**Preview** — {len(df_preview)} rows, {len(df_preview.columns)} columns")
            st.dataframe(df_preview.head(5), use_container_width=True)
            uploaded.seek(0)

            if st.button("⬆ Upload & Process", type="primary"):
                with st.spinner("Processing CSV..."):
                    result, err = api_post(
                        "/upload-csv",
                        files={"file": (uploaded.name, uploaded, "text/csv")},
                    )
                    if err:
                        st.error(err)
                    else:
                        st.success(result.get("message", "Upload successful!"))

    st.markdown("---")
    st.markdown("##### Current Dealer Database")
    dealers, err = api_get("/dealers")
    if err:
        st.warning(err)
    elif dealers:
        df_dealers = pd.DataFrame(dealers)
        display_cols = ["dealer_id", "location", "country", "total_revenue",
                        "avg_margin_pct", "total_quantity_sold", "years_active", "company_size"]
        st.dataframe(df_dealers[display_cols], use_container_width=True)
        st.caption(f"{len(dealers)} dealers in database")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PROSPECT
# ══════════════════════════════════════════════════════════════════════════════
with tab_prospect:
    st.header("🔍 Distributor Prospecting")
    st.markdown("Raw dealer pool discovered from the database. Run the pipeline from the sidebar to populate.")

    if not st.session_state.pipeline_result:
        st.info("👈 Run the pipeline from the sidebar to discover leads.")
    else:
        leads = st.session_state.pipeline_result.get("qualified_leads", [])
        dealers = [l["dealer"] for l in leads]

        if not dealers:
            st.warning("No dealers found.")
        else:
            st.markdown(f"**{len(dealers)} distributors discovered** across {len(set(d['region'] for d in dealers))} regions")

            # Filter controls
            col1, col2 = st.columns(2)
            with col1:
                loc_filter = st.multiselect(
                    "Filter by Location",
                    options=sorted(set(d["location"] for d in dealers)),
                    default=[],
                )
            with col2:
                size_filter = st.multiselect(
                    "Filter by Company Size",
                    options=sorted(set(d["company_size"] for d in dealers)),
                    default=[],
                )

            filtered = dealers
            if loc_filter:
                filtered = [d for d in filtered if d["location"] in loc_filter]
            if size_filter:
                filtered = [d for d in filtered if d["company_size"] in size_filter]

            # Cards
            for dealer in filtered:
                with st.expander(f"🏢 {dealer['dealer_id']} — {dealer['location']}, {dealer['country']}"):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Revenue", f"${dealer['total_revenue']:,.0f}")
                    c2.metric("Avg Margin", f"{dealer['avg_margin_pct']}%")
                    c3.metric("Units Sold", f"{dealer['total_quantity_sold']:,}")
                    c4.metric("Years Active", dealer["years_active"])

                    st.markdown(f"**Categories:** {', '.join(dealer['product_categories'])}")
                    st.markdown(f"**Brands:** {', '.join(dealer['brands_carried'])}")
                    st.markdown(f"**Top Product:** {dealer['top_product']} | **Size:** {dealer['company_size']}")
                    if dealer.get("website"):
                        st.markdown(f"**Website:** {dealer['website']}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — QUALIFY & RANK
# ══════════════════════════════════════════════════════════════════════════════
with tab_qualify:
    st.header("📊 Lead Qualification & Ranking")
    st.markdown("Dealers scored across 5 dimensions and ranked into Hot / Warm / Cold tiers.")

    if not st.session_state.pipeline_result:
        st.info("👈 Run the pipeline from the sidebar first.")
    else:
        leads = st.session_state.pipeline_result.get("qualified_leads", [])

        if not leads:
            st.warning("No qualified leads available.")
        else:
            # Summary row
            hot = [l for l in leads if l["priority_tier"] == "Hot"]
            warm = [l for l in leads if l["priority_tier"] == "Warm"]
            cold = [l for l in leads if l["priority_tier"] == "Cold"]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Leads", len(leads))
            c2.metric("🔴 Hot", len(hot))
            c3.metric("🟡 Warm", len(warm))
            c4.metric("⚪ Cold", len(cold))

            # Tier filter
            tier_show = st.radio("Show tier:", ["All", "Hot", "Warm", "Cold"], horizontal=True)
            display_leads = leads if tier_show == "All" else [l for l in leads if l["priority_tier"] == tier_show]

            st.markdown("---")

            for lead in display_leads:
                dealer = lead["dealer"]
                tier = lead["priority_tier"]
                score = lead["fit_score"]
                tier_class = f"tier-{tier.lower()}"

                st.markdown(
                    f'<div class="{tier_class}">'
                    f'<b>{dealer["dealer_id"]}</b> — {dealer["location"]} &nbsp;'
                    f'{tier_badge(tier)} &nbsp;'
                    f'<b>Score: {score}/100</b>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                with st.expander(f"Details — {dealer['dealer_id']}"):
                    col_l, col_r = st.columns(2)
                    with col_l:
                        st.markdown('<p class="section-header">✅ Fit Reasons</p>', unsafe_allow_html=True)
                        for r in lead.get("fit_reasons", []):
                            st.markdown(f"• {r}")
                    with col_r:
                        st.markdown('<p class="section-header">⚠️ Risk Flags</p>', unsafe_allow_html=True)
                        risks = lead.get("risk_flags", [])
                        if risks:
                            for r in risks:
                                st.markdown(f"• {r}")
                        else:
                            st.markdown("No major risks identified.")

                    if lead.get("llm_narrative"):
                        st.markdown('<p class="section-header">🤖 AI Assessment</p>', unsafe_allow_html=True)
                        st.markdown(f"_{lead['llm_narrative']}_")

                    # Score bar
                    st.markdown('<p class="section-header">Score Breakdown</p>', unsafe_allow_html=True)
                    st.progress(int(score), text=f"{score}/100")

                    # ⚠️ Human review checkpoint
                    st.markdown("---")
                    st.markdown("**⚠️ Human Review Checkpoint**")
                    override = st.selectbox(
                        "Override tier (optional):",
                        ["Keep as is", "Hot", "Warm", "Cold"],
                        key=f"override_{dealer['dealer_id']}",
                    )
                    if override != "Keep as is":
                        st.success(f"Tier override saved: {override} (in production this would update the DB)")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — OUTREACH
# ══════════════════════════════════════════════════════════════════════════════
with tab_outreach:
    st.header("✉️ Outreach Emails")
    st.markdown("Personalised emails generated for Hot and Warm leads. Review and edit before sending.")

    if not st.session_state.pipeline_result:
        st.info("👈 Run the pipeline from the sidebar first.")
    else:
        emails = st.session_state.pipeline_result.get("outreach_emails", [])
        leads_map = {
            l["dealer"]["dealer_id"]: l
            for l in st.session_state.pipeline_result.get("qualified_leads", [])
        }

        if not emails:
            st.warning("No emails generated (only Hot and Warm leads get emails).")
        else:
            st.markdown(f"**{len(emails)} emails drafted**")

            for email in emails:
                dealer_id = email["dealer_id"]
                tier = email.get("priority_tier", "")
                with st.expander(f"✉️ {dealer_id} — {tier_badge(tier)}", expanded=False):
                    st.markdown(unsafe_allow_html=True, body="")

                    # Subject
                    subject_key = f"subject_{dealer_id}"
                    if subject_key not in st.session_state:
                        st.session_state[subject_key] = email["subject"]
                    subject_edit = st.text_input(
                        "Subject line",
                        value=st.session_state[subject_key],
                        key=f"subj_input_{dealer_id}",
                    )

                    # Body
                    body_key = f"body_{dealer_id}"
                    if body_key not in st.session_state:
                        st.session_state[body_key] = email["body"]
                    body_edit = st.text_area(
                        "Email body",
                        value=st.session_state[body_key],
                        height=220,
                        key=f"body_input_{dealer_id}",
                    )

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("💾 Save edits", key=f"save_{dealer_id}"):
                            st.session_state[subject_key] = subject_edit
                            st.session_state[body_key] = body_edit
                            st.success("Saved!")
                    with col2:
                        # ⚠️ Human review checkpoint
                        if st.button("✅ Approve for sending", key=f"approve_{dealer_id}"):
                            st.success("✅ Approved! (In production this would queue the email.)")

                    st.markdown("---")
                    st.caption("⚠️ Human review required before any email is sent.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — NEXT ACTION
# ══════════════════════════════════════════════════════════════════════════════
with tab_action:
    st.header("🎯 Next Best Action")
    st.markdown("CRM-style action recommendations for each lead, plus reply handler for incoming messages.")

    # ── Sub-section: Next Actions ──────────────────────────────────────────
    st.markdown("### Recommended Actions")
    if not st.session_state.pipeline_result:
        st.info("👈 Run the pipeline from the sidebar first.")
    else:
        actions = st.session_state.pipeline_result.get("next_actions", [])
        if not actions:
            st.warning("No actions available.")
        else:
            # Summary table
            df_actions = pd.DataFrame(actions)
            display_df = df_actions[["dealer_id", "priority_tier", "fit_score", "recommended_action", "urgency"]].copy()
            display_df.columns = ["Dealer", "Tier", "Score", "Recommended Action", "Urgency"]
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("### Action Details")
            for action in actions:
                urgency_color = {"High": "🔴", "Medium": "🟡", "Low": "⚪"}.get(action["urgency"], "⚪")
                with st.expander(
                    f"{urgency_color} {action['dealer_id']} — {action['recommended_action']}"
                ):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Tier", action["priority_tier"])
                    col2.metric("Score", f"{action['fit_score']}/100")
                    col3.metric("Urgency", action["urgency"])

                    st.markdown(f"**Reasoning:** {action['reasoning']}")
                    st.markdown(
                        action_style(action["recommended_action"]),
                        unsafe_allow_html=True,
                    )

    st.markdown("---")

    # ── Sub-section: Reply Handler ─────────────────────────────────────────
    st.markdown("### 💬 Reply Handler")
    st.markdown("Paste a distributor's reply to get a suggested response and next action.")

    col1, col2 = st.columns([1, 2])
    with col1:
        reply_dealer = st.text_input("Dealer ID", placeholder="e.g. Dealer_3")
        reply_text = st.text_area(
            "Distributor reply",
            height=150,
            placeholder="Paste the distributor's email reply here...",
        )
        if st.button("🤖 Analyse Reply", type="primary"):
            if not reply_dealer or not reply_text:
                st.warning("Please enter both dealer ID and reply text.")
            else:
                with st.spinner("Analysing reply..."):
                    result, err = api_post(
                        "/handle-reply",
                        {"dealer_id": reply_dealer, "reply_text": reply_text},
                    )
                    if err:
                        st.error(err)
                    else:
                        st.session_state.reply_result = result

    with col2:
        if st.session_state.reply_result:
            r = st.session_state.reply_result
            st.markdown("#### Analysis")
            sentiment_colors = {
                "Positive": "🟢", "Neutral": "🟡",
                "Objection": "🟠", "Negative": "🔴",
            }
            icon = sentiment_colors.get(r.get("sentiment", ""), "⚪")
            st.markdown(f"**Sentiment:** {icon} {r.get('sentiment', 'Unknown')}")
            st.markdown(
                f"**Next Action:** {action_style(r.get('next_action', 'Follow-up'))}",
                unsafe_allow_html=True,
            )
            st.markdown("**Suggested Response:**")
            st.markdown(
                f'<div class="email-box">{r.get("suggested_response", "")}</div>',
                unsafe_allow_html=True,
            )
            if st.button("✅ Approve response"):
                st.success("Response approved! (In production this would queue the reply.)")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.header("💬 Chat with the Agent")
    st.markdown(
        "Ask anything about distributors, the market, or your pipeline. "
        "The agent will detect your intent and route to the right context."
    )

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_messages:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-user">{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                intent_label = msg.get("intent", "general")
                st.markdown(
                    f'<div class="chat-assistant">{msg["content"]}'
                    f'<br><span class="intent-badge">intent: {intent_label}</span></div>',
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    # Suggested queries
    st.markdown("**Suggested queries:**")
    sugg_cols = st.columns(3)
    suggestions = [
        "Which distributors are best fit for StepForward?",
        "What margin should we offer distributors in Singapore?",
        "How should I respond to a distributor who says we're unknown?",
        "Which regions should we prioritise for expansion?",
        "What is the MOQ for first-time distributors?",
        "How do we handle a distributor already carrying Nike?",
    ]
    for i, sugg in enumerate(suggestions):
        if sugg_cols[i % 3].button(sugg, key=f"sugg_{i}"):
            st.session_state.chat_messages.append({"role": "user", "content": sugg})
            with st.spinner("Thinking..."):
                result, err = api_post("/chat", {"query": sugg, "chat_history": []})
                if err:
                    response = f"Error: {err}"
                    intent = "error"
                else:
                    response = result.get("response", "No response.")
                    intent = result.get("intent", "general")
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": response,
                "intent": intent,
            })
            st.rerun()

    # Input box
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Your message...", placeholder="Ask the agent anything...")
        submitted = st.form_submit_button("Send ➤", use_container_width=True)

    if submitted and user_input:
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.spinner("Thinking..."):
            result, err = api_post("/chat", {"query": user_input, "chat_history": []})
            if err:
                response = f"Error: {err}"
                intent = "error"
            else:
                response = result.get("response", "No response.")
                intent = result.get("intent", "general")
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": response,
            "intent": intent,
        })
        st.rerun()

    if st.session_state.chat_messages:
        if st.button("🗑 Clear chat"):
            st.session_state.chat_messages = []
            st.rerun()
