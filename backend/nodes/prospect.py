"""
Node 1 — Prospect
Loads dealers from the mock JSON database.
Uses absolute path consistent with main.py.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any

DATA_PATH = Path(os.environ.get("DEALERS_DB_PATH", "/content/footwear-agent/data/dealers.json"))


def prospect_node(state: Dict[str, Any]) -> Dict[str, Any]:
    print(f"[Prospect] Loading dealer pool from {DATA_PATH}...")
    with open(DATA_PATH) as f:
        dealers = json.load(f)

    region_filter = state.get("region_filter")
    if region_filter:
        dealers = [d for d in dealers if region_filter.lower() in d["region"].lower()]

    print(f"[Prospect] Found {len(dealers)} dealers.")
    state["dealers"] = dealers
    return state
