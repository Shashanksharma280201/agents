"""
LangGraph Invoice Agent
- Supports Claude, OpenAI, Gemini (set LLM_PROVIDER in .env)
- Reads a PDF (invoice) using pdfplumber
- Extracts structured data (vendor, cost, addresses, line items...)
- Retries once if required fields are missing
- Saves results to agents.db → invoices table

Usage:
    python3 invoice_agent.py invoice.pdf
"""

import json
import time
from pathlib import Path
from typing import Literal, TypedDict

import pdfplumber
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage

from llm_provider import get_llm_with_schema, current_model
from db import save_run, save_invoice


# ─── Schemas ──────────────────────────────────────────────────────────────────

class LineItem(BaseModel):
    description: str = ""
    quantity: float | None = None
    unit_price: float | None = None
    total: float | None = None

class InvoiceData(BaseModel):
    invoice_number: str | None = None
    vendor_name: str | None = None
    vendor_address: str | None = None
    bill_to: str | None = None
    ship_to: str | None = None
    line_items: list[LineItem] = Field(default_factory=list)
    subtotal: float | None = None
    tax: float | None = None
    total_amount: float | None = None
    currency: str = "USD"
    invoice_date: str | None = None
    due_date: str | None = None
    payment_terms: str | None = None
    notes: str | None = None


# ─── State ────────────────────────────────────────────────────────────────────

class InvoiceState(TypedDict):
    file_path: str
    raw_text: str
    invoice_data: dict | None
    retry_count: int
    missing_fields: list[str]
    error: str | None


# ─── Prompt ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an invoice data extraction expert.
Extract ALL available fields from the invoice text provided.
- For amounts: extract as numbers only (no currency symbols)
- For dates: use YYYY-MM-DD format if possible
- For addresses: include full multi-line address as a single string
- If a field is not present in the invoice, leave it as null
- Extract ALL line items you can find"""


# ─── Nodes ────────────────────────────────────────────────────────────────────

def read_pdf_node(state: InvoiceState) -> InvoiceState:
    """Extract raw text + tables from all pages of the PDF."""
    try:
        text_parts = []
        with pdfplumber.open(state["file_path"]) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                for table in page.extract_tables():
                    for row in table:
                        text_parts.append(" | ".join(str(c or "") for c in row))
                text_parts.append(f"--- Page {i} ---\n{text}")
        return {"raw_text": "\n".join(text_parts), "error": None}
    except Exception as e:
        return {"raw_text": "", "error": f"PDF read failed: {e}"}


def extract_node(state: InvoiceState) -> InvoiceState:
    """LLM extracts structured invoice data from raw text."""
    if state.get("error"):
        return state

    hint = ""
    if state.get("retry_count", 0) > 0:
        missing = ", ".join(state.get("missing_fields", []))
        hint = f"\n\nIMPORTANT: Previous extraction was missing: {missing}. Look more carefully for these fields."

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Extract invoice data from this text:{hint}\n\n{state['raw_text']}"),
    ]

    try:
        llm = get_llm_with_schema(InvoiceData)
        result: InvoiceData = llm.invoke(messages)
        return {
            "invoice_data": result.model_dump(),
            "retry_count": state.get("retry_count", 0) + 1,
            "error": None,
        }
    except Exception as e:
        return {"error": f"Extraction failed: {e}"}


def validate_node(state: InvoiceState) -> InvoiceState:
    """Check which required fields are still missing."""
    if state.get("error") or not state.get("invoice_data"):
        return state
    required = ["vendor_name", "total_amount", "invoice_date"]
    missing = [f for f in required if not state["invoice_data"].get(f)]
    return {"missing_fields": missing}


def save_node(state: InvoiceState) -> InvoiceState:
    """Save extracted invoice data to SQLite."""
    if state.get("error") or not state.get("invoice_data"):
        return state

    file_name = Path(state["file_path"]).name

    run_id = save_run(
        agent_name="invoice_agent",
        query=file_name,
        messages=[HumanMessage(content=state["raw_text"][:500])],
        answer=json.dumps(state["invoice_data"]),
        model=current_model(),
    )

    invoice_id = save_invoice(
        file_name=file_name,
        invoice_data=state["invoice_data"],
        raw_text=state["raw_text"],
        agent_run_id=run_id,
    )

    print(f"\n[DB] Invoice saved → id={invoice_id}, run_id={run_id}")
    return state


# ─── Routing ──────────────────────────────────────────────────────────────────

def should_retry(state: InvoiceState) -> Literal["extract", "save"]:
    if state.get("error"):
        return "save"
    missing = state.get("missing_fields", [])
    if missing and state.get("retry_count", 0) < 2:
        print(f"\n[RETRY] Missing fields: {missing} — retrying...")
        return "extract"
    return "save"


# ─── Graph ────────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(InvoiceState)
    graph.add_node("read_pdf", read_pdf_node)
    graph.add_node("extract",  extract_node)
    graph.add_node("validate", validate_node)
    graph.add_node("save",     save_node)
    graph.add_edge(START,      "read_pdf")
    graph.add_edge("read_pdf", "extract")
    graph.add_edge("extract",  "validate")
    graph.add_conditional_edges("validate", should_retry)
    graph.add_edge("save",     END)
    return graph.compile()


# ─── Main ─────────────────────────────────────────────────────────────────────

def extract_invoice(file_path: str, verbose: bool = True) -> dict:
    app = build_graph()
    state: InvoiceState = {
        "file_path": file_path,
        "raw_text": "",
        "invoice_data": None,
        "retry_count": 0,
        "missing_fields": [],
        "error": None,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"Invoice  : {file_path}")
        print(f"Provider : {current_model()}")
        print(f"{'='*60}")

    t0 = time.monotonic()
    final = app.invoke(state)
    duration = int((time.monotonic() - t0) * 1000)

    if final.get("error"):
        print(f"\n[ERROR] {final['error']}")
        return {}

    if verbose and final.get("invoice_data"):
        d = final["invoice_data"]
        print(f"\n{'─'*60}")
        print(f"  Vendor       : {d.get('vendor_name')}")
        print(f"  Invoice #    : {d.get('invoice_number')}")
        print(f"  Date         : {d.get('invoice_date')}")
        print(f"  Bill To      : {d.get('bill_to')}")
        print(f"  Ship To      : {d.get('ship_to')}")
        print(f"  Total        : {d.get('currency')} {d.get('total_amount')}")
        print(f"  Tax          : {d.get('tax')}")
        print(f"  Line Items   : {len(d.get('line_items', []))} items")
        print(f"  Duration     : {duration}ms")
        print(f"{'─'*60}")

    return final.get("invoice_data", {})


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 invoice_agent.py <path-to-invoice.pdf>")
        sys.exit(1)
    extract_invoice(sys.argv[1])
