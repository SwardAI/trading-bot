import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


# Load .env file from project root
load_dotenv()

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"


def load_config(path: str | Path) -> dict:
    """Load a single YAML config file.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed config dictionary.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config or {}


def resolve_env_vars(config: dict) -> dict:
    """Recursively resolve keys ending in '_env' to their environment variable values.

    For example, {"api_key_env": "BINANCE_API_KEY"} becomes
    {"api_key": "<actual env var value>"} while keeping the original _env key.

    Args:
        config: Config dictionary to process.

    Returns:
        Config with resolved environment variables added.
    """
    resolved = {}
    for key, value in config.items():
        if isinstance(value, dict):
            resolved[key] = resolve_env_vars(value)
        elif isinstance(value, str) and key.endswith("_env"):
            resolved[key] = value
            # Add resolved key without _env suffix
            actual_key = key[:-4]  # strip '_env'
            env_value = os.getenv(value)
            if env_value:
                resolved[actual_key] = env_value
        else:
            resolved[key] = value
    return resolved


def validate_settings(config: dict) -> list[str]:
    """Validate the main settings config for required keys.

    Args:
        config: The settings config dictionary.

    Returns:
        List of validation error messages (empty if valid).
    """
    errors = []

    required_sections = ["bot", "exchanges", "risk_management"]
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: '{section}'")

    if "bot" in config:
        bot = config["bot"]
        if bot.get("mode") not in ("paper", "live"):
            errors.append(f"bot.mode must be 'paper' or 'live', got '{bot.get('mode')}'")

    if "risk_management" in config:
        risk = config["risk_management"]
        numeric_keys = [
            "max_risk_per_trade_pct", "max_order_size_usd",
            "max_total_exposure_pct", "daily_loss_limit_pct",
            "weekly_loss_limit_pct", "monthly_loss_limit_pct",
        ]
        for key in numeric_keys:
            val = risk.get(key)
            if val is not None and not isinstance(val, (int, float)):
                errors.append(f"risk_management.{key} must be a number, got {type(val).__name__}")

    return errors


def load_all_configs() -> dict:
    """Load and merge all config files (settings + strategy configs).

    Returns:
        Merged config dictionary with env vars resolved.

    Raises:
        ValueError: If settings config fails validation.
    """
    settings = load_config(CONFIG_DIR / "settings.yaml")
    grid = load_config(CONFIG_DIR / "grid_config.yaml")
    momentum = load_config(CONFIG_DIR / "momentum_config.yaml")
    funding = load_config(CONFIG_DIR / "funding_config.yaml")

    # Validate settings
    errors = validate_settings(settings)
    if errors:
        raise ValueError(f"Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    # Resolve env vars in settings
    settings = resolve_env_vars(settings)

    # Merge all configs
    config = {**settings}
    config.update(grid)
    config.update(momentum)
    config.update(funding)

    return config
