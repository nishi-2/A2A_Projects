import asyncio
import os
import sys
from pathlib import Path
from typing import Any

_agent_dir = Path(__file__).resolve().parent
for _path in (_agent_dir, _agent_dir.parent):
    _entry = str(_path)
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

import google.auth
from google.auth import credentials

from common.config import load_env

load_env()

from beeai_framework.adapters.a2a.agents import A2AAgent
from beeai_framework.adapters.vertexai import VertexAIChatModel
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.requirement.requirements.conditional import ConditionalRequirement
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.middleware.trajectory import EventMeta, GlobalTrajectoryMiddleware
from beeai_framework.tools import Tool
from beeai_framework.tools.handoff import HandoffTool
from beeai_framework.tools.think import ThinkTool


def setup_gcp_env() -> tuple[credentials.Credentials, str]:
    """Load GCP config from .env and authenticate via ADC (no JSON key)."""
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")

    if not project_id:
        raise ValueError("GOOGLE_CLOUD_PROJECT is missing from .env")

    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = os.environ.get(
        "GOOGLE_GENAI_USE_VERTEXAI", "TRUE"
    )
    os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
    os.environ["GOOGLE_CLOUD_LOCATION"] = location

    creds, adc_project = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )

    return creds, project_id or adc_project


credentials, project_id = setup_gcp_env()


class ConciseGlobalTrajectoryMiddleware(GlobalTrajectoryMiddleware):
    def _format_prefix(self, meta: EventMeta) -> str:
        prefix = super()._format_prefix(meta)
        return prefix.rstrip(": ")

    def _format_payload(self, value: Any) -> str:
        return ""


async def create_healthcare_agent() -> RequirementAgent:
    host = os.environ.get("AGENT_HOST", "localhost")
    policy_agent_port = os.environ.get("POLICY_AGENT_PORT", "9999")
    research_agent_port = os.environ.get("RESEARCH_AGENT_PORT", "9998")
    provider_agent_port = os.environ.get("PROVIDER_AGENT_PORT", "9997")

    policy_agent = A2AAgent(
        url=f"http://{host}:{policy_agent_port}",
        memory=UnconstrainedMemory(),
    )
    await policy_agent.check_agent_exists()
    print("\t -->", f"{policy_agent.name} initialized")

    research_agent = A2AAgent(
        url=f"http://{host}:{research_agent_port}",
        memory=UnconstrainedMemory(),
    )
    await research_agent.check_agent_exists()
    print("\t -->", f"{research_agent.name} initialized")

    provider_agent = A2AAgent(
        url=f"http://{host}:{provider_agent_port}",
        memory=UnconstrainedMemory(),
    )
    await provider_agent.check_agent_exists()
    print("\t -->", f"{provider_agent.name} initialized")

    healthcare_agent = RequirementAgent(
        name="Healthcare Agent",
        description="A personal concierge for Healthcare Information, customized to your policy.",
        llm=VertexAIChatModel(
            model_id="gemini-2.5-flash",
            project=project_id,
            location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
            allow_parallel_tool_calls=True,
            allow_prompt_caching=False,
        ),
        tools=[
            thinktool := ThinkTool(),
            HandoffTool(
                target=policy_agent,
                name=policy_agent.name,
                description=policy_agent.agent_card.description,
            ),
            HandoffTool(
                target=research_agent,
                name=research_agent.name,
                description=research_agent.agent_card.description,
            ),
            HandoffTool(
                target=provider_agent,
                name=provider_agent.name,
                description=provider_agent.agent_card.description,
            ),
        ],
        requirements=[
            ConditionalRequirement(
                thinktool,
                force_at_step=1,
                force_after=Tool,
                consecutive_allowed=False,
            ),
        ],
        role="Healthcare Concierge",
        instructions=(
            f"""You are a concierge for healthcare services. Your task is 
            to handoff to one or more agents to answer questions and provide 
            a detailed summary of their answers. Be sure that all of their 
            questions are answered before responding.
            Use `{policy_agent.name}` to answer insurance-related questions.
            
            IMPORTANT: When returning answers about providers, only output 
            providers from `{provider_agent.name}` and only provide insurance 
            information based on the results from `{policy_agent.name}`.

            In your output, put which agent gave you the information!"""
        ),
    )

    print("\t -->", f"{healthcare_agent.meta.name} initialized")
    return healthcare_agent


DEFAULT_PROMPT = (
    "I'm based in Austin, TX. How do I get mental health therapy near me "
    "and what does my insurance cover?"
)


async def run_orchestrator(prompt: str = DEFAULT_PROMPT) -> None:
    agent = await create_healthcare_agent()
    response = await agent.run(prompt).middleware(ConciseGlobalTrajectoryMiddleware())
    print(response.last_message.text)


def main() -> None:
    print("Running A2A Orchestrator Agent...")
    asyncio.run(run_orchestrator())


if __name__ == "__main__":
    main()
