# Autonomous Research Assistant

Capstone project (UMBC DATA606) — an agentic research assistant built on
Claude, LangChain, ChromaDB, and Streamlit.

The agent receives a research question, decomposes it, pulls from multiple
sources (web, ArXiv, PubMed, Semantic Scholar), retrieves relevant chunks
from a local vector store, self-critiques its draft, and returns a
structured report through a Streamlit dashboard.

## Stack

- Python 3.10+
- Anthropic Claude (Opus 4.5 / Sonnet 4.5 / Haiku 4.5)
- LangChain + LangChain Community
- ChromaDB (persistent vector store)
- Sentence-Transformers (`BAAI/bge-small-en-v1.5`)
- spaCy (`en_core_web_lg`) + KeyBERT for query decomposition
- Streamlit + Plotly for the dashboard
- SQLite for long-term session memory

## Project Structure

```
research-assistant/
├── agent/
│   ├── react_loop.py       # Core ReAct controller
│   ├── tools.py            # Tool definitions / dispatcher
│   ├── memory.py           # Short + long-term memory
│   └── critic.py           # Self-critique module
├── data/
│   ├── ingest.py           # Fetch + chunk + embed pipeline
│   ├── vectorstore.py      # ChromaDB wrapper
│   └── sources/            # ArXiv, PubMed, S2, SerpAPI connectors
├── nlp/
│   ├── decompose.py        # Query decomposition (spaCy + KeyBERT)
│   └── embeddings.py       # BAAI/bge wrapper
├── dashboard/
│   ├── app.py              # Streamlit entry point
│   └── charts.py           # Plotly chart helpers
├── tests/
├── logs/                   # Runtime logs (gitignored)
├── hello_claude.py         # Week 1 sanity check for the Claude API
├── requirements.txt
└── .env.example
```

## Setup

```powershell
# 1. Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the spaCy model
python -m spacy download en_core_web_lg

# 4. Copy the env template and fill in your keys
Copy-Item .env.example .env
# then edit .env with ANTHROPIC_API_KEY, SERPAPI_API_KEY, etc.

# 5. Verify the Claude connection
python hello_claude.py
```

## Roadmap

| Phase | Weeks | Focus                  | Deliverable                                |
| ----- | ----- | ---------------------- | ------------------------------------------ |
| 1     | 1–2   | Foundation & Setup     | Working ReAct loop with 1 tool             |
| 2     | 3–4   | Data Pipeline & RAG    | Multi-source fetch + ChromaDB retrieval    |
| 3     | 5–6   | Agent Intelligence     | Self-critique, memory, query decomposition |
| 4     | 7–8   | Dashboard & Evaluation | Full Streamlit UI + final report           |
