# 👟 StepForward Footwear — AI Distributor Sales Agent

An end-to-end AI sales agent built with LangChain, LangGraph, FastAPI and Streamlit
to help a footwear brand identify, qualify, and engage global distributors.

---

## Architecture

```
CSV Upload → Mock DB (dealers.json)
                │
         LangGraph Pipeline
         ┌──────────────────────────────────────────┐
         │  Prospect → Qualify → Prioritize         │
         │     → Outreach → Next Action             │
         └──────────────────────────────────────────┘
                │                    │
          RAG (FAISS)           Reply Handler
          Brand docs            (on-demand)
                │
         FastAPI (REST)
                │
         Streamlit UI
         ┌──────────────────────────────────────────┐
         │  Upload | Prospect | Qualify | Outreach  │
         │  Next Action | Chat                      │
         └──────────────────────────────────────────┘
```

## Tech Stack

| Component | Technology |
|---|---|
| Workflow orchestration | LangGraph 0.0.55 |
| Agent framework | LangChain 0.1.20 |
| LLM | Zephyr-7B-Beta (HF Inference API) |
| Embeddings | all-MiniLM-L6-v2 (HuggingFace, local) |
| Vector store | FAISS (CPU) |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Database | Mock JSON (dealers.json) |

## Project Structure

```
footwear-agent/
├── backend/
│   ├── main.py               # FastAPI app — all REST endpoints
│   ├── graph.py              # LangGraph workflow assembly
│   ├── llm.py                # LLM initialisation (HF Inference API)
│   ├── models.py             # Pydantic schemas
│   ├── nodes/
│   │   ├── prospect.py       # Node 1: load dealer pool
│   │   ├── qualify.py        # Node 2: rule scoring + RAG + LLM narrative
│   │   ├── prioritize.py     # Node 3: Hot/Warm/Cold tiering
│   │   ├── outreach.py       # Node 4: personalised email generation
│   │   ├── next_action.py    # Node 5: CRM action recommendation
│   │   ├── reply_handler.py  # Node 6: objection/reply analysis
│   │   └── chat_agent.py     # Intent router + general chat
│   └── rag/
│       ├── vectorstore.py    # FAISS build + load
│       └── brand_docs/       # Brand profile, FAQ, market intelligence
├── frontend/
│   └── app.py                # Streamlit UI (6 tabs)
├── data/
│   └── dealers.json          # Mock dealer database
├── run_colab.ipynb           # Colab run guide (8 cells)
├── requirements.txt
└── .env.example
```

## How to Run (Google Colab)

See `run_colab.ipynb` — open it in Colab and run cells 1-6 in order.
You will need:
- A free HuggingFace token: https://huggingface.co/settings/tokens
- (Optional) ngrok account for persistent URLs: https://ngrok.com

## Human Review Checkpoints

The agent flags 3 explicit human review points:
1. **After Qualify** — user can override lead tier before outreach
2. **Before Outreach Send** — user reviews/edits email, clicks Approve
3. **Escalation** — very high-scoring leads (≥88/100) flagged for senior BDM

## Assumptions

- Dealer data sourced from Kaggle footwear sales dataset (public)
- Brand profile (StepForward) is a simulated mid-market footwear brand
- No real emails are sent — all outreach is draft-only
- Region/country data inferred from dealer location names
- Years active defaulted to 3 when CSV-imported (not in source data)

