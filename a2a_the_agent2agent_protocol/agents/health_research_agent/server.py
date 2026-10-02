"""Health Research Agent — Google ADK + web search, exposed via A2A."""

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

import google.auth
import uvicorn
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.agents import LlmAgent
from google.adk.tools import google_search

from common.config import load_env

load_env()

logging.disable(level=logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def setup_gcp_env() -> tuple[google.auth.credentials.Credentials, str]:
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

    credentials, adc_project = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )

    return credentials, project_id or adc_project


setup_gcp_env()

PORT = int(os.environ.get("RESEARCH_AGENT_PORT", "9998"))
HOST = os.environ.get("AGENT_HOST", "localhost")

research_agent = LlmAgent(
    model="gemini-3.1-pro-preview",
    name="HealthResearchAgent",
    tools=[google_search],
    description="Provides healthcare information about symptoms, health "
    "conditions, treatments, and procedures using up-to-date web resources.",
    instruction="""You are a healthcare research agent tasked with 
    providing information about health conditions. Use the google_search 
    tool to find information on the web about options, symptoms, treatments, 
    and procedures. Cite your sources in your responses. Output all of the 
    information you find.""",
)


def main() -> None:
    a2a_app = to_a2a(research_agent, host=HOST, port=PORT)
    print("Running Health Research Agent")
    uvicorn.run(a2a_app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
