"""LangGraph healthcare provider agent using MCP tools and OpenRouter Nemotron."""

import asyncio
import os
import sys
from pathlib import Path

_agent_dir = Path(__file__).resolve().parent
for _path in (_agent_dir, _agent_dir.parent):
    _entry = str(_path)
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import StdioConnection
from langchain_openai import ChatOpenAI

from common.config import load_env

AGENT_DIR = Path(__file__).resolve().parent
load_env()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
NEMOTRON_MODEL = "nvidia/nemotron-3-super-120b-a12b"


class ProviderAgent:
    def __init__(self) -> None:
        openrouter_api_key = os.environ.get("OPENROUTER_AI_KEY", "").strip()
        if not openrouter_api_key:
            raise ValueError("OPENROUTER_AI_KEY is missing from .env")

        self.mcp_client = MultiServerMCPClient(
            {
                "find_healthcare_providers": StdioConnection(
                    transport="stdio",
                    command="uv",
                    args=["run", "mcp_server.py"],
                    cwd=str(AGENT_DIR),
                )
            }
        )

        self.openrouter_api_key = openrouter_api_key
        self.base_url = OPENROUTER_BASE_URL
        self.agent = None

    async def initialize(self):
        """Initialize the agent asynchronously."""
        tools = await self.mcp_client.get_tools()
        self.agent = create_agent(
            ChatOpenAI(
                model=NEMOTRON_MODEL,
                api_key=self.openrouter_api_key,
                base_url=self.base_url,
                temperature=0,
                max_tokens=1000,
            ),
            tools,
            name="HealthcareProviderAgent",
            system_prompt="""Your task is to find and list providers using 
            the find_healthcare_providers MCP Tool based on the users query. 
            Only use providers based on the response from the tool. Output 
            the information in a table.""",
        )
        return self

    async def answer_query(self, prompt: str) -> str:
        if self.agent is None:
            raise RuntimeError("""Agent not initialized. Call initialize() 
            first.""")

        response = await self.agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]
            }
        )
        return response["messages"][-1].content


async def main() -> None:
    prompt = "I'm based in Austin, TX. Are there any Psychiatrists near me?"
    provider_agent = ProviderAgent()
    await provider_agent.initialize()

    print("Running Healthcare Provider Agent (OpenRouter Nemotron)")
    print(f"Prompt: {prompt}\n")

    answer = await provider_agent.answer_query(prompt)
    print("=== Response ===")
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
