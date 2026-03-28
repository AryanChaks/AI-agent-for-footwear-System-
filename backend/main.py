"""
FastAPI backend — exposes the LangGraph pipeline as REST endpoints.
"""

import json
import os
from pathlib import Path
from typing import Optional, List

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

from backend.graph import run_pipeline, run_reply_handler
from backend.nodes.chat_agent import classify_intent, chat_general
from backend.rag.vectorstore import build_vectorstore

# Absolute path — works regardless of where the process is launched from
DATA_PATH = Path(os.environ.get("DEALERS_DB_PATH", "/content/footwear-agent/data/dealers.json"))

app = FastAPI(title="StepForward Footwear — Distributor Sales Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunPipelineRequest(BaseModel):
    region_filter: Optional[str] = None


class ReplyRequest(BaseModel):
    dealer_id: str
    reply_text: str


class ChatRequest(BaseModel):
    query: str
    chat_history: Optional[List[dict]] = []


@app.get("/health")
def health():
    return {"status": "ok", "model": "mistralai/Mistral-7B-Instruct-v0.2"}


@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        df.columns = df.columns.str.strip()

        print("Columns found:", list(df.columns))   # visible in FastAPI logs

        # Flexible column matching — handle minor naming variations
        col_map = {}
        for col in df.columns:
            cl = col.lower().strip()
            if cl == "dealer":
                col_map["Dealer"] = col
            elif "location" in cl and "dealer" in cl:
                col_map["Dealer Location"] = col
            elif cl == "brand":
                col_map["Brand"] = col
            elif cl == "product":
                col_map["Product"] = col
            elif "total revenue" in cl:
                col_map["Total Revenue"] = col
            elif "margin" in cl and "%" in col:
                col_map["Margin (%)"] = col
            elif cl == "quantity sold":
                col_map["Quantity Sold"] = col
            elif "stock" in cl:
                col_map["Stock Availability"] = col

        required = ["Dealer", "Dealer Location", "Brand", "Product",
                    "Total Revenue", "Margin (%)", "Quantity Sold", "Stock Availability"]
        missing = [r for r in required if r not in col_map]
        if missing:
            raise HTTPException(400, f"Could not find columns: {missing}. Found: {list(df.columns)}")

        # Rename to standard names
        df = df.rename(columns={v: k for k, v in col_map.items()})

        df["Margin_num"] = (
            df["Margin (%)"].astype(str).str.replace("%", "").str.strip().astype(float)
        )

        # Determine company size from revenue percentiles
        def company_size(rev):
            if rev >= 50_000_000:
                return "Large"
            elif rev >= 10_000_000:
                return "Mid-market"
            else:
                return "SME"

        agg = df.groupby("Dealer").agg(
            location=("Dealer Location", "first"),
            brands_carried=("Brand", lambda x: list(x.unique())),
            product_categories=("Product", lambda x: list(x.unique())),
            total_revenue=("Total Revenue", "sum"),
            avg_margin_pct=("Margin_num", "mean"),
            total_quantity_sold=("Quantity Sold", "sum"),
            avg_stock_level=("Stock Availability", "mean"),
            top_product=("Product", lambda x: x.value_counts().index[0]),
            transaction_count=("Product", "count"),
        ).reset_index()

        dealers = []
        for _, row in agg.iterrows():
            rev = round(row["total_revenue"], 2)
            # Infer years_active from transaction count (proxy)
            txn = row["transaction_count"]
            years = max(1, min(10, round(txn / 200)))   # rough heuristic
            dealers.append({
                "dealer_id": row["Dealer"],
                "location": row["location"],
                "country": "Singapore",
                "region": "Southeast Asia",
                "brands_carried": row["brands_carried"],
                "product_categories": row["product_categories"],
                "total_revenue": rev,
                "avg_margin_pct": round(row["avg_margin_pct"], 1),
                "total_quantity_sold": int(row["total_quantity_sold"]),
                "avg_stock_level": round(row["avg_stock_level"], 1),
                "top_product": row["top_product"],
                "years_active": years,
                "website": f"{row['Dealer'].lower().replace('_', '')}.sg",
                "company_size": company_size(rev),
            })

        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DATA_PATH, "w") as f:
            json.dump(dealers, f, indent=2)

        print(f"[Upload] Wrote {len(dealers)} dealers to {DATA_PATH}")
        return {"message": f"Uploaded {len(dealers)} dealers from CSV.", "count": len(dealers)}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"CSV processing failed: {e}")


@app.get("/dealers")
def get_dealers():
    with open(DATA_PATH) as f:
        return json.load(f)


@app.post("/run-pipeline")
def run_full_pipeline(req: RunPipelineRequest):
    try:
        result = run_pipeline(region_filter=req.region_filter)
        return {
            "qualified_leads": result.get("qualified_leads", []),
            "outreach_emails": result.get("outreach_emails", []),
            "next_actions": result.get("next_actions", []),
            "error": result.get("error"),
        }
    except Exception as e:
        raise HTTPException(500, f"Pipeline failed: {e}")


@app.post("/handle-reply")
def handle_reply(req: ReplyRequest):
    try:
        result = run_reply_handler(req.dealer_id, req.reply_text)
        analyses = result.get("reply_analyses", [])
        if analyses:
            return analyses[0]
        raise HTTPException(500, "No analysis produced.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Reply handler failed: {e}")


@app.post("/chat")
def chat(req: ChatRequest):
    try:
        intent = classify_intent(req.query)
        response = chat_general(req.query)
        return {
            "intent": intent,
            "response": response,
            "suggested_tab": intent,
        }
    except Exception as e:
        raise HTTPException(500, f"Chat failed: {e}")


@app.post("/rebuild-vectorstore")
def rebuild_vectorstore():
    try:
        build_vectorstore()
        return {"message": "Vectorstore rebuilt successfully."}
    except Exception as e:
        raise HTTPException(500, f"Rebuild failed: {e}")
