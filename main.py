"""CryptoQuantBot — entry point.

Usage:
    python main.py              Start the bot (paper mode by default)
    python main.py --check      Verify config and connections, then exit
"""

import argparse
import sys

from src.core.config_loader import load_all_configs
from src.core.logger import setup_logger

logger = setup_logger("main")


def check_config():
    """Load and validate all configs, print summary."""
    logger.info("Checking configuration...")

    config = load_all_configs()
    bot = config.get("bot", {})

    print(f"\nBot name:  {bot.get('name', 'N/A')}")
    print(f"Mode:      {bot.get('mode', 'N/A')}")
    print(f"Log level: {bot.get('log_level', 'N/A')}")

    exchanges = config.get("exchanges", {})
    enabled = [k for k, v in exchanges.items() if v.get("enabled")]
    print(f"Exchanges: {', '.join(enabled) if enabled else 'none enabled'}")

    strategies = []
    if config.get("grid_strategy", {}).get("enabled"):
        strategies.append("grid")
    if config.get("momentum_strategy", {}).get("enabled"):
        strategies.append("momentum")
    if config.get("funding_strategy", {}).get("enabled"):
        strategies.append("funding")
    print(f"Strategies: {', '.join(strategies) if strategies else 'none enabled'}")

    risk = config.get("risk_management", {})
    print(f"\nRisk limits:")
    print(f"  Max exposure:     {risk.get('max_total_exposure_pct', 'N/A')}%")
    print(f"  Daily loss limit: {risk.get('daily_loss_limit_pct', 'N/A')}%")
    print(f"  Cash reserve:     {risk.get('reserve_cash_pct', 'N/A')}%")

    print("\nConfig check passed.")


def main():
    parser = argparse.ArgumentParser(description="CryptoQuantBot")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify configuration and exit",
    )
    args = parser.parse_args()

    if args.check:
        try:
            check_config()
        except Exception as e:
            logger.error(f"Config check failed: {e}")
            sys.exit(1)
        return

    # Normal startup
    from src.core.bot import Bot

    try:
        bot = Bot()
        bot.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
