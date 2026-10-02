"""Shared config for all A2A agents."""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

# agents/common/config.py -> parents[2] == a2a_the_agent2agent_protocol/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENTS_DIR = Path(__file__).resolve().parents[1]


def bootstrap_paths(caller_file: str | Path) -> Path:
    """Add agent package dir and agents/ to sys.path. Call before importing common."""
    agent_dir = Path(caller_file).resolve().parent
    for path in (agent_dir, AGENTS_DIR):
        entry = str(path)
        if entry not in sys.path:
            sys.path.insert(0, entry)
    return agent_dir


def load_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=True)
