# AGENTS.md

Guidance for AI coding agents working in this repository.

A single FastAPI backend (port 8006) exposing a LangGraph ReAct agent —
the "Market Assistant" — that answers portfolio and market questions using
seven tools, MongoDB Atlas for storage/vector search/agent memory, AWS
Bedrock for chat completions, and VoyageAI for finance-domain embeddings.

## Build and test commands

```bash
cd backend
poetry env use python3.10   # the project pins python = ">=3.10,<3.11" exactly
poetry install --no-interaction -v --no-cache --no-root
poetry run uvicorn main:app --host 0.0.0.0 --port 8006 --reload
```

One-time setup after `.env` is filled in (see below):

```bash
# Seed data (no Bedrock/Voyage calls -- embeddings are already baked into
# the exported JSON under backend/db/collections/)
for f in backend/db/collections/*.json; do
  coll=$(basename "$f" .json | sed 's/^agentic_capital_markets\.//')
  mongoimport --uri "$MONGODB_URI" --db agentic_capital_markets \
    --collection "$coll" --file "$f" --jsonArray
done

# Vector search indexes (idempotent -- warns if already present)
cd backend/agent/db && poetry run python vector_search_index_creator.py
```

**There is no automated test suite in this repository.** No `pytest`
dependency, no `test_*.py` files. To verify a change, run the service and
exercise `POST /market-assistant/send-message`, or use the console chatbot
(`poetry run python console_market_assistant.py` from `backend/`).

## Project structure

```
backend/
  main.py                     FastAPI app, mounts the two routers below
  api_market_assistant.py     POST /market-assistant/send-message, /clear-all-memory
  api_checkpointer.py         POST /checkpointer/scheduler-overview
  checkpointer_memory_jobs.py Daily 04:00 UTC job: drops thread_ids not from today
  console_market_assistant.py CLI chat interface
  agent/
    react_agent.py             MarketAssistantReactAgent -- builds the LangGraph
                                ReAct agent, wires AsyncMongoDBSaver checkpointing
    react_agent_console.py     Console-only variant with its own client setup
    react_agent_tools.py       All 7 tools (see API overview)
    profiles.py                AgentProfiles -- reads the agent's system prompt
                                from the agent_profiles collection
    bedrock/client.py          BedrockClient wrapper for ChatBedrockConverse
    vogayeai/vogaye_ai_embeddings.py  VogayeAIEmbeddings (voyage-finance-2)
    db/mdb.py                   MongoDBConnector -- sync wrapper, sets appName
    db/vector_search_index_creator.py  Run once after seeding
  db/collections/*.json        mongoexport-format seed data, one file per collection
```

Notable files:

- [backend/agent/db/mdb.py](backend/agent/db/mdb.py) — the only client wrapper
  that reads `APP_NAME` by default; the three raw `MongoClient`/`AsyncMongoClient`
  calls elsewhere (`react_agent.py`, `react_agent_console.py`,
  `checkpointer_memory_jobs.py`) now also pass it explicitly.
- [backend/agent/react_agent_tools.py](backend/agent/react_agent_tools.py) —
  collection-name fallback defaults must stay snake_case
  (`portfolio_allocation`, not `portfolioAllocation`) to match the seed data;
  this drifted once already, in both this file and `react_agent_console.py`.

## API overview

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/` | Health check |
| POST | `/market-assistant/send-message` | `{message, thread_id?}` → agent response, tool calls used |
| POST | `/market-assistant/clear-all-memory` | Wipes both checkpointer collections |
| POST | `/checkpointer/scheduler-overview` | Status of the daily memory-cleanup job |

Agent tools (all in `react_agent_tools.py` unless noted):

| Tool | Data source |
| --- | --- |
| `get_portfolio_allocation_tool` | `portfolio_allocation` |
| `get_portfolio_ytd_return_tool` | `portfolio_performance` |
| `get_vix_closing_value_tool` | Embedded in `reports_market_analysis` |
| `market_analysis_reports_vector_search_tool` | `reports_market_analysis` (vector search) |
| `market_news_reports_vector_search_tool` | `reports_market_news` (vector search) |
| `market_social_media_reports_vector_search_tool` | `reports_market_sm` (vector search) |
| `tavily_search_tool` | Tavily web search API (general finance questions only) |

## Environment variables and configuration

See [backend/.env.example](backend/.env.example) for the full list with
defaults. The three that most often cause silent failures if missing:

- **`MONGODB_URI` / `DATABASE_NAME`** — read at construction time by four
  separate client-init call sites, not one shared factory. If you add a
  fifth, pass `appname=os.getenv("APP_NAME")` explicitly; nothing here
  centralizes it.
- **`VOYAGE_API_KEY`** — needed even though the seed data already has
  embeddings baked in, because live user queries still need embedding at
  request time for vector search to work at all.
- **The three `REPORT_*_VECTOR_INDEX_NAME` variables must match the index
  names actually created.** `vector_search_index_creator.py`'s `__main__`
  block derives the names it creates from these same env vars, so as long as
  you run it after `.env` is filled in, they can't drift — but if you rename
  an index by hand in the Atlas UI without updating `.env`, vector search
  silently returns nothing.

Constraints worth knowing before you debug a failure:

- **Python must be exactly 3.10.x.** `pyproject.toml` pins
  `python = ">=3.10,<3.11"`. 3.13 (a common current default) will not
  resolve the lockfile.
- **AWS SSO sessions expire mid-task.** If Bedrock calls start failing with
  `ExpiredTokenException` or `NoCredentialsError` partway through a long
  run, the fix is `aws sso login`, not a code change.
- **`risk_profiles` is seeded but never read.** Real data exists (4 profiles,
  one marked `active`), and the README lists it as a collection to create,
  but no tool, route, or prompt in the codebase queries it. Any agent answer
  that references a specific risk profile is not grounded in this
  collection — check `reports_market_analysis`/`reports_market_news` free
  text instead, since that is genuinely retrieved.
- **`main.py` never calls `load_dotenv()` itself** — this is fine, because
  every module that reads an env var at import time (`react_agent_tools.py`,
  `mdb.py`) calls it locally first, before its own reads. If you add a new
  module with module-level `os.getenv()` calls, follow that same
  self-contained pattern rather than relying on import order.

## MongoDB Skills

Use the official MongoDB agent skills from https://github.com/mongodb/agent-skills
whenever the task is MongoDB-specific and a matching skill exists.

## When To Use EDD.md

Use [EDD.md](./EDD.md) as the source of truth for the MongoDB data model in this repository.

Consult [EDD.md](./EDD.md) before making changes that touch:

- MongoDB collections, document structure, or field names
- FastAPI routes or agent tools that read or write database records
- Vector search index definitions or embedding fields
- Schema documentation, Mermaid diagrams, or entity modeling discussions
