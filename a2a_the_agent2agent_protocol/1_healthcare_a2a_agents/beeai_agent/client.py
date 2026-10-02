"""A2A client for the BeeAI Healthcare Concierge Agent."""

import argparse
import asyncio
import os
import sys
from pathlib import Path

_agent_dir = Path(__file__).resolve().parent
for _path in (_agent_dir, _agent_dir.parent):
    _entry = str(_path)
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

import httpx

from common.config import load_env

load_env()

from beeai_framework.adapters.a2a.agents import A2AAgent
from beeai_framework.memory import UnconstrainedMemory

from orchestrator import ConciseGlobalTrajectoryMiddleware, DEFAULT_PROMPT


async def run_client(host: str, port: int, prompt: str) -> None:
    agent = A2AAgent(
        url=f"http://{host}:{port}",
        memory=UnconstrainedMemory(),
    )
    await agent.check_agent_exists()
    print(f"Connected to {agent.name}\n")

    response = await agent.run(prompt).middleware(ConciseGlobalTrajectoryMiddleware())
    print(response.last_message.text)


def main() -> None:
    parser = argparse.ArgumentParser(description="A2A client for the Healthcare Concierge Agent")
    parser.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT, help="Question to send to the agent")
    parser.add_argument("--host", default=os.environ.get("AGENT_HOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("HEALTHCARE_AGENT_PORT", "9996")))
    args = parser.parse_args()

    try:
        asyncio.run(run_client(args.host, args.port, args.prompt))
    except httpx.ConnectError:
        print(
            f"Could not connect to {args.host}:{args.port}. "
            "Start the server first: python server.py (from agents/beeai_agent/)",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
