import asyncio
import os
import sys
from pathlib import Path

_agent_dir = Path(__file__).resolve().parent
for _path in (_agent_dir, _agent_dir.parent):
    _entry = str(_path)
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from beeai_framework.adapters.a2a.serve.executors.tool_calling_agent_executor import (
    ToolCallingAgentExecutor,
)
from beeai_framework.serve.utils import LRUMemoryManager

from common.config import load_env

load_env()

from orchestrator import create_healthcare_agent


def main() -> None:
    print("Running A2A Server for Healthcare Agent...")

    host = os.environ.get("AGENT_HOST", "localhost")
    port = int(os.environ.get("HEALTHCARE_AGENT_PORT", "9996"))

    agent = asyncio.run(create_healthcare_agent())
    memory_manager = LRUMemoryManager(maxsize=100)

    agent_card = AgentCard(
        name=agent.meta.name,
        description=agent.meta.description,
        url=f"http://{host}:{port}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[
            AgentSkill(
                id=agent.meta.name,
                name=agent.meta.name,
                description=agent.meta.description,
                tags=[],
            )
        ],
    )

    executor = ToolCallingAgentExecutor(
        agent=agent,
        agent_card=agent_card,
        memory_manager=memory_manager,
        send_trajectory=True,
    )

    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    uvicorn.run(server.build(), host=host, port=port)


if __name__ == "__main__":
    main()
