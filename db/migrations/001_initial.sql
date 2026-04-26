CREATE TABLE IF NOT EXISTS agent_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_name  TEXT        NOT NULL,
    query       TEXT        NOT NULL,
    answer      TEXT,
    sources     TEXT,                        -- JSON array of URLs
    messages    TEXT        NOT NULL,        -- JSON serialised message history
    model       TEXT,
    duration_ms INTEGER,
    created_at  TEXT        NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_agent_runs_agent_name ON agent_runs (agent_name);
CREATE INDEX IF NOT EXISTS idx_agent_runs_created_at ON agent_runs (created_at DESC);
