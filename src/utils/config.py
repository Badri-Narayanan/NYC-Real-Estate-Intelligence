from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@lru_cache(maxsize=1)
def get_project_root() -> Path:
    """Return absolute path of the project root directory."""
    return PROJECT_ROOT


@lru_cache(maxsize=1)
def load_env() -> None:
    """Load .env from config/ exactly once per process."""
    env_path = PROJECT_ROOT / "config" / ".env"
    if env_path.exists():
        load_dotenv(env_path)


@lru_cache(maxsize=1)
def load_config(config_path: str | None = None) -> dict[str, Any]:
    """
    Load YAML config, merging in env-overridable fields.

    Returns a fully-resolved dict with absolute paths.
    """
    load_env()

    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "config.yaml"
    else:
        config_path = Path(config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Override data mode from env if set
    env_mode = os.getenv("DATA_MODE")
    if env_mode:
        cfg["data"]["mode"] = env_mode

    # Resolve all paths to absolute
    for key, rel in cfg["paths"].items():
        cfg["paths"][key] = str(PROJECT_ROOT / rel)

    # Inject API keys (kept out of YAML for safety)
    cfg["api_keys"] = {
        "anthropic": os.getenv("ANTHROPIC_API_KEY", ""),
        "walkscore": os.getenv("WALKSCORE_API_KEY", ""),
        "census": os.getenv("CENSUS_API_KEY", ""),
        "socrata": os.getenv("SOCRATA_APP_TOKEN", ""),
    }

    return cfg


def get_api_key(name: str) -> str:
    """Get an API key from environment. Returns empty string if missing."""
    load_env()
    keymap = {
        "anthropic": "ANTHROPIC_API_KEY",
        "walkscore": "WALKSCORE_API_KEY",
        "census": "CENSUS_API_KEY",
        "socrata": "SOCRATA_APP_TOKEN",
    }
    return os.getenv(keymap.get(name, name.upper()), "")