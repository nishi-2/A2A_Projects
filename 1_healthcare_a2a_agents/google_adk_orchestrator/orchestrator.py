"""
Healthcare workflow orchestrator using Google ADK SequentialAgent.

Connects to remote A2A agents (start both servers first):
  - Health Research Agent: agents/health_research_agent/server.py
  - Policy Agent:          agents/policy_agent/server.py
"""

import argparse
import asyncio
import logging
import os
import sys
import warnings
from pathlib import Path

_agent_dir = Path(__file__).resolve().parent
for _path in (_agent_dir, _agent_dir.parent):
    _entry = str(_path)
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from google.adk.agents import SequentialAgent
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.runners import InMemoryRunner

from common.config import load_env

load_env()

logging.disable(level=logging.WARNING)
warnings.filterwarnings("ignore")

POLICY_AGENT_SERVER = "agents/policy_agent/server.py"
RESEARCH_AGENT_SERVER = "agents/health_research_agent/server.py"


def agent_url(port_env_key: str, default_port: str) -> str:
    host = os.environ.get("AGENT_HOST", "localhost")
    port = os.environ.get(port_env_key, default_port)
    return f"http://{host}:{port}"


def build_root_agent() -> SequentialAgent:
    policy_url = agent_url("POLICY_AGENT_PORT", "9999")
    research_url = agent_url("RESEARCH_AGENT_PORT", "9998")

    policy_agent = RemoteA2aAgent(name="policy_agent", agent_card=policy_url)
    print(f"\t --> {policy_agent.name} -> {policy_url}")

    health_research_agent = RemoteA2aAgent(name="health_research_agent", agent_card=research_url)
    print(f"\t --> {health_research_agent.name} -> {research_url}")

    root_agent = SequentialAgent(
        name="root_agent",
        description="Healthcare Routing Agent",
        sub_agents=[health_research_agent, policy_agent],
    )
    print(f"\t --> {root_agent.name} initialized")
    return root_agent


async def run_workflow(prompt: str) -> int:
    print("Running Healthcare Workflow Agent")
    print(f"Prompt: {prompt}\n")

    root_agent = build_root_agent()
    runner = InMemoryRunner(root_agent)

    try:
        events = await runner.run_debug(prompt, quiet=True)
    except Exception as exc:
        print(f"Error: {exc}\n", file=sys.stderr)
        print(
            "Make sure both agent servers are running:\n"
            f"  Terminal 1: python {POLICY_AGENT_SERVER}\n"
            f"  Terminal 2: python {RESEARCH_AGENT_SERVER}",
            file=sys.stderr,
        )
        return 1

    printed = False
    for event in events:
        if event.is_final_response() and event.content and event.content.parts:
            text = event.content.parts[0].text
            if text:
                print("=== Response ===")
                print(text)
                printed = True

    if not printed:
        print("No final response received.", file=sys.stderr)
        return 1

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Orchestrate Health Research + Policy agents via A2A")
    parser.add_argument(
        "prompt",
        nargs="?",
        default="How can I get mental health therapy?",
        help="Question for the healthcare workflow",
    )
    args = parser.parse_args()

    sys.exit(asyncio.run(run_workflow(args.prompt)))


if __name__ == "__main__":
    main()
