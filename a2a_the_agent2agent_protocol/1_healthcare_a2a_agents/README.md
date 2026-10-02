# A2A Healthcare Agents

Multi-agent healthcare demo using the [Agent2Agent (A2A)](https://github.com/a2aproject/A2A) protocol. Specialized agents run as independent A2A servers; orchestrators route queries across them.

## Architecture

| Agent | Folder | Port | Role |
|-------|--------|------|------|
| Policy | `agents/policy_agent/` | 9999 | Insurance coverage Q&A (OpenAI + PDF) |
| Research | `agents/health_research_agent/` | 9998 | Health info via web search (Google ADK + Gemini) |
| Provider | `agents/healthcare_provider_agent/` | 9997 | Find doctors by location (LangGraph + MCP + OpenRouter) |
| BeeAI Concierge | `agents/beeai_agent/` | 9996 | Orchestrates all three via handoffs (BeeAI + Vertex AI) |

Each folder follows the same layout: `server.py`, optional `client.py`, and agent logic modules.

## Setup

```bash
cd a2a_the_agent2agent_protocol
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env` and fill in keys:

- `OPENAI_API_KEY` — policy agent
- `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION` — research + BeeAI (use `gcloud auth application-default login`)
- `OPENROUTER_AI_KEY` — provider agent

Place `2026AnthemgHIPSBC.pdf` in `agents/policy_agent/`.

## Start agents

Open one terminal per server (from project root or each agent folder):

```bash
python agents/policy_agent/server.py          # :9999
python agents/health_research_agent/server.py # :9998
python agents/healthcare_provider_agent/server.py # :9997
python agents/beeai_agent/server.py           # :9996  (needs all three above)
```

## Query agents

**BeeAI concierge** (recommended — combines policy + research + provider):

```bash
python agents/beeai_agent/client.py
python agents/beeai_agent/client.py "What mental health services are covered in Austin?"
```

**Run orchestrator locally** (no A2A server):

```bash
python agents/beeai_agent/orchestrator.py
```

**Google ADK orchestrator** (research → policy only):

```bash
python agents/google_adk_orchestrator/orchestrator.py "How can I get mental health therapy?"
```

**Individual agents:**

```bash
python agents/policy_agent/client.py "How much would I pay for therapy?"
```

## Notes

- All agents load config from `.env` via `agents/common/config.py`.
- Provider agent may hit OpenRouter rate limits on shared Nemotron pools — retry or add credits.
- `healthcare_provider_agent/client.py` uses Microsoft Agent Framework and needs **a2a-sdk 1.x**; other agents use **0.3.16**.
