# Agents

LangGraph-based AI agents with shared SQLite storage and multi-provider LLM support.

## Setup

```bash
pip install langgraph langchain langchain-anthropic langchain-openai langchain-google-genai \
            langchain-community ddgs pdfplumber temporalio reportlab sounddevice openai-whisper \
            python-dotenv scipy

# Linux only (for microphone support)
sudo apt-get install -y libportaudio2
```

Copy `.env.example` to `.env` and fill in your API keys.

---

## Files

| File | What it does |
|---|---|
| `llm_provider.py` | Shared LLM factory — switch provider via `.env` |
| `search_agent.py` | Web search agent (text / audio file / mic) |
| `invoice_agent.py` | PDF invoice extractor |
| `invoice_agent_temporal.py` | Invoice extractor with Temporal workflow orchestration |
| `db/` | Shared SQLite layer used by all agents |

---

## Running each file

### Search Agent
```bash
# text query
python3 search_agent.py "What is LangGraph?"

# audio file
python3 search_agent.py recording.wav

# microphone — press Enter to stop
python3 search_agent.py --listen

# microphone — fixed duration
python3 search_agent.py --listen --seconds 10
```

### Invoice Agent
```bash
python3 invoice_agent.py invoice/wordpress-pdf-invoice-plugin-sample.pdf
```

### Invoice Agent with Temporal
```bash
# Terminal 1 — start Temporal dev server
temporal server start-dev

# Terminal 2 — start worker
python3 invoice_agent_temporal.py worker

# Terminal 3 — single file
python3 invoice_agent_temporal.py run invoice/dummy_invoice_001.pdf

# Terminal 3 — batch (all PDFs, 10 concurrent)
python3 invoice_agent_temporal.py batch invoice/ 10
```
Temporal UI → http://localhost:8233

### Generate 200 dummy invoices
```bash
python3 generate_invoices.py
# creates invoice/dummy_invoice_001.pdf ... dummy_invoice_200.pdf
```
---

## Switching LLM provider

Edit `.env`:
```
LLM_PROVIDER=claude    # claude-sonnet-4-6
LLM_PROVIDER=openai    # gpt-4o
LLM_PROVIDER=gemini    # gemini-2.0-flash
```

No code changes needed — all agents pick it up automatically.
