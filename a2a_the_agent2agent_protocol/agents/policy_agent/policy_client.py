"""A2A client for the Policy Agent """

import argparse
import asyncio
import json
import os
import sys

import httpx
from a2a.client import Client, ClientConfig, ClientFactory, create_text_message_object
from a2a.types import AgentCard, Artifact, Message, Task
from a2a.utils.message import get_message_text
from dotenv import load_dotenv


def print_agent_card(agent_card: AgentCard) -> None:
    print("=== Agent ===")
    print(f"Name:     {agent_card.name}")
    print(f"Version:  {agent_card.version}")
    print(f"URL:      {agent_card.url}")
    print(f"Protocol: {agent_card.protocol_version}")
    if agent_card.skills:
        print("Skills:")
        for skill in agent_card.skills:
            print(f"  - {skill.name}: {skill.description}")
    print()


async def run_client(host: str, port: str, prompt: str) -> int:
    base_url = f"http://{host}:{port}"

    async with httpx.AsyncClient(timeout=50.0) as httpx_client:
        client: Client = await ClientFactory.connect(base_url, client_config=ClientConfig(httpx_client=httpx_client),)

        agent_card = await client.get_card()
        print_agent_card(agent_card)

        print(f"Prompt: {prompt}\n")

        message = create_text_message_object(content=prompt)
        responses = client.send_message(message)

        text_content = ""
        metadata: dict = {}

        async for response in responses:
            if isinstance(response, Message):
                text_content = get_message_text(response)
                metadata = {
                    "message_id": response.message_id,
                    "role": getattr(response, "role", None),
                    "metadata": getattr(response, "metadata", None),
                }
            elif isinstance(response, tuple):
                task: Task = response[0]
                metadata = {
                    "task_id": task.id,
                    "task_state": getattr(task.status, "state", None)
                    if task.status else None,
                }
                if task.artifacts:
                    artifact: Artifact = task.artifacts[0]
                    text_content = get_message_text(artifact)
                    metadata["artifact_id"] = artifact.artifact_id
                    if getattr(artifact, "metadata", None):
                        metadata["artifact_metadata"] = artifact.metadata

        print("=== Response ===")
        if text_content:
            print(text_content)
        else:
            print("No response text received.")
            return 1

        print("\n=== Metadata ===")
        for key, value in metadata.items():
            if value is not None:
                print(f"{key}: {value}")

        # OpenAI token usage is computed on the server side and is not part of
        # the standard A2A response unless the server adds it to message metadata.
        token_keys = ("usage", "tokens", "input_tokens", "output_tokens", "total_tokens")
        msg_meta = metadata.get("metadata") or metadata.get("artifact_metadata") or {}
        if isinstance(msg_meta, dict) and any(k in msg_meta for k in token_keys):
            print("\n=== Token usage ===")
            print(json.dumps({k: msg_meta[k] for k in token_keys if k in msg_meta}, indent=2))
        else:
            print("\nToken usage: not available via A2A client (see server/OpenAI logs).")

        return 0


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="A2A client for the Policy Agent")
    parser.add_argument(
        "prompt",
        nargs="?",
        default="How much would I pay for mental health therapy?",
        help="Question to send to the agent",
    )
    parser.add_argument("--host", default=os.environ.get("AGENT_HOST", "localhost"))
    parser.add_argument(
        "--port",
        default=os.environ.get("POLICY_AGENT_PORT", "9999"),
    )
    args = parser.parse_args()

    try:
        exit_code = asyncio.run(run_client(args.host, args.port, args.prompt))
    except httpx.ConnectError:
        print(
            f"Could not connect to {args.host}:{args.port}. "
            "Start the server first: python -m agents",
            file=sys.stderr,
        )
        exit_code = 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
