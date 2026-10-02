"""Microsoft Agent Framework client for the Healthcare Provider Agent (requires a2a-sdk 1.x venv)."""

import asyncio
import os
import sys
from pathlib import Path

_agent_dir = Path(__file__).resolve().parent
for _path in (_agent_dir, _agent_dir.parent):
    _entry = str(_path)
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from agent_framework.a2a import A2AAgent

from common.config import load_env

load_env()

host = os.environ.get("AGENT_HOST", "localhost")
port = os.environ.get("PROVIDER_AGENT_PORT", "9997")
base_url = f"http://{host}:{port}"

prompt = "I'm based in Austin, TX. Are there any Psychiatrists near me?"


async def main() -> None:
    async with A2AAgent(name="HealthcareProviderAgent", url=base_url) as healthcare_provider_agent:
        result = await healthcare_provider_agent.run(prompt)
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
