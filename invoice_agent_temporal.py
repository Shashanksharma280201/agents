"""
LangGraph Invoice Agent — Temporal Edition
==========================================
Wraps the invoice extraction pipeline in a Temporal workflow so that:
  - Each step is a retryable Activity
  - Crashes mid-pipeline resume automatically
  - 200 PDFs are processed in parallel with concurrency control
  - Full audit trail visible in Temporal UI (http://localhost:8233)

Usage:
  Terminal 1 — start Temporal dev server:
    temporal server start-dev

  Terminal 2 — start the worker:
    python3 invoice_agent_temporal.py worker

  Terminal 3 — submit all PDFs in invoice/ folder:
    python3 invoice_agent_temporal.py batch invoice/

  Or submit a single file:
    python3 invoice_agent_temporal.py run invoice/dummy_invoice_001.pdf
"""

import asyncio
import json
import sys
from datetime import timedelta
from pathlib import Path
from typing import Optional

# ── Temporal imports ──────────────────────────────────────────────────────────
from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.worker import Worker
from temporalio.common import RetryPolicy

# ── Project imports (same as invoice_agent.py) ────────────────────────────────
with workflow.unsafe.imports_passed_through():
    import pdfplumber
    from pydantic import BaseModel, Field
    from langchain_core.messages import HumanMessage, SystemMessage
    from llm_provider import get_llm_with_schema, current_model
    from db import save_run, save_invoice

TASK_QUEUE = "invoice-queue"


# ─── Pydantic schema (same as invoice_agent.py) ───────────────────────────────

class LineItem(BaseModel):
    description: str = ""
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    total: Optional[float] = None

class InvoiceData(BaseModel):
    invoice_number: Optional[str] = None
    vendor_name: Optional[str] = None
    vendor_address: Optional[str] = None
    bill_to: Optional[str] = None
    ship_to: Optional[str] = None
    line_items: list[LineItem] = Field(default_factory=list)
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    total_amount: Optional[float] = None
    currency: str = "USD"
    invoice_date: Optional[str] = None
    due_date: Optional[str] = None
    payment_terms: Optional[str] = None
    notes: Optional[str] = None


SYSTEM_PROMPT = """You are an invoice data extraction expert.
Extract ALL available fields from the invoice text provided.
- For amounts: extract as numbers only (no currency symbols)
- For dates: use YYYY-MM-DD format if possible
- For addresses: include full multi-line address as a single string
- If a field is not present in the invoice, leave it as null
- Extract ALL line items you can find"""


# ─── Activities ───────────────────────────────────────────────────────────────
# Each activity = one step. Temporal retries these independently on failure.

@activity.defn
async def read_pdf_activity(file_path: str) -> str:
    """Extract raw text + tables from all PDF pages."""
    text_parts = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            for table in page.extract_tables():
                for row in table:
                    text_parts.append(" | ".join(str(c or "") for c in row))
            text_parts.append(f"--- Page {i} ---\n{text}")
    return "\n".join(text_parts)


@activity.defn
async def extract_activity(payload: dict) -> dict:
    """LLM extracts structured invoice data. payload = {raw_text, hint}"""
    raw_text = payload["raw_text"]
    hint     = payload.get("hint", "")

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Extract invoice data from this text:{hint}\n\n{raw_text}"),
    ]
    llm = get_llm_with_schema(InvoiceData)
    result: InvoiceData = llm.invoke(messages)
    return result.model_dump()


@activity.defn
async def validate_activity(invoice_data: dict) -> list[str]:
    """Return list of missing required fields."""
    required = ["vendor_name", "total_amount", "invoice_date"]
    return [f for f in required if not invoice_data.get(f)]


@activity.defn
async def save_db_activity(payload: dict) -> dict:
    """Persist results to SQLite. Returns {run_id, invoice_id}."""
    file_name    = payload["file_name"]
    invoice_data = payload["invoice_data"]
    raw_text     = payload["raw_text"]
    model        = payload["model"]

    run_id = save_run(
        agent_name="invoice_agent_temporal",
        query=file_name,
        messages=[HumanMessage(content=raw_text[:500])],
        answer=json.dumps(invoice_data),
        model=model,
    )
    invoice_id = save_invoice(
        file_name=file_name,
        invoice_data=invoice_data,
        raw_text=raw_text,
        agent_run_id=run_id,
    )
    return {"run_id": run_id, "invoice_id": invoice_id}


# ─── Workflow ──────────────────────────────────────────────────────────────────

@workflow.defn
class InvoiceWorkflow:
    """
    Flow:
      read_pdf → extract → validate → (retry if missing) → save_db
    """

    @workflow.run
    async def run(self, file_path: str) -> dict:
        file_name = Path(file_path).name

        # Step 1: read PDF
        raw_text = await workflow.execute_activity(
            read_pdf_activity,
            file_path,
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )

        # Step 2 + 3: extract → validate → retry once if fields missing
        hint = ""
        invoice_data = {}
        for attempt in range(2):
            invoice_data = await workflow.execute_activity(
                extract_activity,
                {"raw_text": raw_text, "hint": hint},
                start_to_close_timeout=timedelta(minutes=3),
                retry_policy=RetryPolicy(
                    maximum_attempts=5,
                    initial_interval=timedelta(seconds=10),
                ),
            )
            missing = await workflow.execute_activity(
                validate_activity,
                invoice_data,
                start_to_close_timeout=timedelta(seconds=10),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )
            if not missing or attempt == 1:
                break
            hint = f"\n\nIMPORTANT: Previous extraction was missing: {', '.join(missing)}. Look more carefully."

        # Step 4: save to DB
        result = await workflow.execute_activity(
            save_db_activity,
            {
                "file_name":    file_name,
                "invoice_data": invoice_data,
                "raw_text":     raw_text,
                "model":        current_model(),
            },
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=RetryPolicy(maximum_attempts=5),
        )

        return {
            "file":       file_name,
            "vendor":     invoice_data.get("vendor_name"),
            "total":      invoice_data.get("total_amount"),
            "currency":   invoice_data.get("currency"),
            "run_id":     result["run_id"],
            "invoice_id": result["invoice_id"],
        }


# ─── Worker ───────────────────────────────────────────────────────────────────

async def run_worker():
    client = await Client.connect("localhost:7233")
    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[InvoiceWorkflow],
        activities=[read_pdf_activity, extract_activity, validate_activity, save_db_activity],
    )
    print(f"[WORKER] Started on task queue: {TASK_QUEUE}")
    print(f"[WORKER] Provider: {current_model()}")
    print(f"[WORKER] Temporal UI: http://localhost:8233")
    print(f"[WORKER] Waiting for workflows...\n")
    await worker.run()


# ─── Submit single workflow ───────────────────────────────────────────────────

async def run_single(file_path: str):
    client = await Client.connect("localhost:7233")
    stem   = Path(file_path).stem
    handle = await client.start_workflow(
        InvoiceWorkflow.run,
        file_path,
        id=f"invoice-{stem}",
        task_queue=TASK_QUEUE,
    )
    print(f"[SUBMITTED] {file_path}  →  workflow id: invoice-{stem}")
    result = await handle.result()
    print(f"[DONE] {result}")
    return result


# ─── Batch: submit all PDFs with concurrency limit ────────────────────────────

async def run_batch(pdf_dir: str, max_concurrent: int = 10):
    pdfs = sorted(Path(pdf_dir).glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {pdf_dir}")
        return

    client    = await Client.connect("localhost:7233")
    semaphore = asyncio.Semaphore(max_concurrent)
    total     = len(pdfs)
    done      = 0
    failed    = 0

    print(f"[BATCH] Submitting {total} PDFs from {pdf_dir}/")
    print(f"[BATCH] Concurrency: {max_concurrent} parallel workflows")
    print(f"[BATCH] Provider: {current_model()}")
    print(f"[BATCH] Temporal UI: http://localhost:8233\n")

    async def process(pdf: Path):
        nonlocal done, failed
        async with semaphore:
            try:
                handle = await client.start_workflow(
                    InvoiceWorkflow.run,
                    str(pdf),
                    id=f"invoice-{pdf.stem}",
                    task_queue=TASK_QUEUE,
                )
                result = await handle.result()
                done += 1
                print(f"  ✓ [{done}/{total}] {pdf.name} → vendor={result.get('vendor')} total={result.get('currency')} {result.get('total')}")
            except Exception as e:
                failed += 1
                print(f"  ✗ [{done+failed}/{total}] {pdf.name} → ERROR: {e}")

    await asyncio.gather(*[process(pdf) for pdf in pdfs])
    print(f"\n[BATCH] Complete — {done} succeeded, {failed} failed out of {total}")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "worker":
        asyncio.run(run_worker())

    elif cmd == "run" and len(sys.argv) >= 3:
        asyncio.run(run_single(sys.argv[2]))

    elif cmd == "batch" and len(sys.argv) >= 3:
        pdf_dir        = sys.argv[2]
        max_concurrent = int(sys.argv[3]) if len(sys.argv) >= 4 else 10
        asyncio.run(run_batch(pdf_dir, max_concurrent))

    else:
        print(__doc__)
        sys.exit(1)
