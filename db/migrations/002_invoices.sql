CREATE TABLE IF NOT EXISTS invoices (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_run_id    INTEGER,                      -- links to agent_runs table
    file_name       TEXT        NOT NULL,
    invoice_number  TEXT,
    vendor_name     TEXT,
    vendor_address  TEXT,
    bill_to         TEXT,
    ship_to         TEXT,
    line_items      TEXT,                         -- JSON array
    subtotal        REAL,
    tax             REAL,
    total_amount    REAL,
    currency        TEXT,
    invoice_date    TEXT,
    due_date        TEXT,
    payment_terms   TEXT,
    notes           TEXT,
    raw_text        TEXT,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_invoices_vendor   ON invoices (vendor_name);
CREATE INDEX IF NOT EXISTS idx_invoices_date     ON invoices (invoice_date DESC);
