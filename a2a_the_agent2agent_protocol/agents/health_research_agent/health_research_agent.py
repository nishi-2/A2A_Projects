"""  This will have the Health Research Agent which will use Google's Agent Development Kit (ADK) 
and a tool to search the web for health information. """

""" Instead of manually defining the Executor and RequestHandler as in Policy Agent, 
we will use google.adk.a2a.utils.agent_to_a2a.to_a2a to automatically wrap the ADK agent into an A2A-compliant application. """

import os

import uvicorn
from dotenv import load_dotenv
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.agents import LlmAgent
from google.adk.tools import google_search

import logging
import warnings

logging.disable(level=logging.WARNING)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

load_dotenv()

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]


### Method to Authenticate with Google

import google.auth

def setup_gcp_env() -> tuple[google.auth.credentials.Credentials, str]:
    """Load GCP config from .env and authenticate via ADC (no JSON key)."""
    load_dotenv(PROJECT_ROOT / ".env", override=True)

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

    # Prefer .env project over ADC default if they differ
    return credentials, project_id or adc_project


credentials, project_id = setup_gcp_env()

### Authentication process ends

# Authenticate
credentials, project_id = setup_gcp_env()

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
    # Make your agent A2A-compatible
    a2a_app = to_a2a(research_agent, host=HOST, port=PORT)
    print("Running Health Research Agent")
    uvicorn.run(a2a_app, host=HOST, port=PORT)

    
if __name__ == "__main__":
    main()


