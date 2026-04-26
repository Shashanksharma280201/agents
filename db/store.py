import json
import re
from .connection import get_connection, init_db

init_db()


def _extract_urls(messages: list) -> list[str]:
    urls = []
    for msg in messages:
        content = getattr(msg, "content", "") or ""
        urls.extend(re.findall(r"https?://[^\s\)\]\>\"']+", content))
    return list(dict.fromkeys(urls))  # deduplicated, order preserved


def _serialise(messages: list) -> str:
    from langchain_core.messages import messages_to_dict
    return json.dumps(messages_to_dict(messages), ensure_ascii=False)


def save_run(
    agent_name: str,
    query: str,
    messages: list,
    answer: str | None = None,
    model: str | None = None,
    duration_ms: int | None = None,
) -> int:
    sources = _extract_urls(messages)
    conn = get_connection()
    with conn:
        cur = conn.execute(
            """
            INSERT INTO agent_runs (agent_name, query, answer, sources, messages, model, duration_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                agent_name,
                query,
                answer,
                json.dumps(sources),
                _serialise(messages),
                model,
                duration_ms,
            ),
        )
        run_id = cur.lastrowid
    conn.close()
    return run_id


def get_runs(agent_name: str | None = None, limit: int = 20) -> list[dict]:
    conn = get_connection()
    if agent_name:
        rows = conn.execute(
            "SELECT * FROM agent_runs WHERE agent_name = ? ORDER BY created_at DESC LIMIT ?",
            (agent_name, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM agent_runs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def save_invoice(
    file_name: str,
    invoice_data: dict,
    raw_text: str,
    agent_run_id: int | None = None,
) -> int:
    conn = get_connection()
    d = invoice_data
    with conn:
        cur = conn.execute(
            """
            INSERT INTO invoices (
                agent_run_id, file_name, invoice_number, vendor_name, vendor_address,
                bill_to, ship_to, line_items, subtotal, tax, total_amount, currency,
                invoice_date, due_date, payment_terms, notes, raw_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                agent_run_id,
                file_name,
                d.get("invoice_number"),
                d.get("vendor_name"),
                d.get("vendor_address"),
                d.get("bill_to"),
                d.get("ship_to"),
                json.dumps(d.get("line_items", [])),
                d.get("subtotal"),
                d.get("tax"),
                d.get("total_amount"),
                d.get("currency", "USD"),
                d.get("invoice_date"),
                d.get("due_date"),
                d.get("payment_terms"),
                d.get("notes"),
                raw_text,
            ),
        )
        invoice_id = cur.lastrowid
    conn.close()
    return invoice_id


def get_invoices(limit: int = 20) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM invoices ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
