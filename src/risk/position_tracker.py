from src.core.database import Database
from src.core.exchange import ExchangeManager
from src.core.logger import setup_logger

logger = setup_logger("risk.position_tracker")

# Pairs considered highly correlated for exposure grouping
CORRELATED_GROUPS = {
    "crypto_major": ["BTC/USDT", "ETH/USDT"],
}


class PositionTracker:
    """Tracks real-time portfolio exposure from exchange balances and open orders.

    Provides exposure data used by RiskManager for pre-trade checks.

    Args:
        db: Database instance.
        exchange: ExchangeManager instance.
    """

    def __init__(self, db: Database, exchange: ExchangeManager):
        self.db = db
        self.exchange = exchange
        self._cached_balance: dict | None = None

    def refresh_balance(self) -> dict:
        """Fetch fresh balance from exchange and cache it.

        Returns:
            ccxt balance dict with 'total', 'free', 'used'.
        """
        self._cached_balance = self.exchange.fetch_balance()
        return self._cached_balance

    def get_balance(self) -> dict:
        """Get current balance summary.

        Returns:
            Dict with total_usd, free_usd, used_usd, exposure_pct.
        """
        balance = self._cached_balance or self.refresh_balance()
        total = balance.get("total", {})
        free = balance.get("free", {})

        total_usd = float(total.get("USDT", 0))
        free_usd = float(free.get("USDT", 0))
        used_usd = total_usd - free_usd

        return {
            "total_usd": total_usd,
            "free_usd": free_usd,
            "used_usd": used_usd,
            "exposure_pct": (used_usd / total_usd * 100) if total_usd > 0 else 0,
        }

    def get_total_exposure_usd(self) -> float:
        """Get total capital currently in positions (USD)."""
        balance = self.get_balance()
        return balance["used_usd"]

    def get_total_exposure_pct(self) -> float:
        """Get total exposure as percentage of portfolio."""
        balance = self.get_balance()
        return balance["exposure_pct"]

    def get_pair_exposure_usd(self, symbol: str) -> float:
        """Get exposure for a specific trading pair (USD).

        Calculates from open orders and grid inventory in the database.

        Args:
            symbol: Trading pair (e.g. "BTC/USDT").

        Returns:
            Total USD exposure for this pair.
        """
        exposure = 0.0

        # Grid inventory
        grid = self.db.fetch_one(
            "SELECT inventory_amount, inventory_avg_price FROM grid_state WHERE pair = ? ORDER BY updated_at DESC LIMIT 1",
            (symbol,),
        )
        if grid and grid["inventory_amount"] and grid["inventory_avg_price"]:
            exposure += abs(grid["inventory_amount"]) * grid["inventory_avg_price"]

        # Open momentum positions
        momentum = self.db.fetch_one(
            "SELECT amount, entry_price FROM momentum_positions WHERE pair = ? AND status = 'open'",
            (symbol,),
        )
        if momentum:
            exposure += momentum["amount"] * momentum["entry_price"]

        return exposure

    def get_pair_exposure_pct(self, symbol: str) -> float:
        """Get exposure for a pair as percentage of portfolio."""
        balance = self.get_balance()
        if balance["total_usd"] <= 0:
            return 0.0
        pair_usd = self.get_pair_exposure_usd(symbol)
        return pair_usd / balance["total_usd"] * 100

    def get_correlated_exposure_pct(self, symbol: str) -> float:
        """Get combined exposure for all pairs correlated with the given symbol.

        Args:
            symbol: Trading pair to check correlations for.

        Returns:
            Combined exposure percentage for the correlated group.
        """
        balance = self.get_balance()
        if balance["total_usd"] <= 0:
            return 0.0

        # Find which correlation group this symbol belongs to
        correlated_symbols = [symbol]
        for group_symbols in CORRELATED_GROUPS.values():
            if symbol in group_symbols:
                correlated_symbols = group_symbols
                break

        total_exposure = sum(
            self.get_pair_exposure_usd(s) for s in correlated_symbols
        )
        return total_exposure / balance["total_usd"] * 100

    def sync_with_exchange(self):
        """Reconcile local position data with exchange state.

        Exchange is the source of truth. Updates cached balance
        and logs any discrepancies.
        """
        logger.info("Syncing positions with exchange...")
        self.refresh_balance()
        balance = self.get_balance()
        logger.info(
            f"Balance synced: ${balance['total_usd']:.2f} total, "
            f"${balance['free_usd']:.2f} free, "
            f"{balance['exposure_pct']:.1f}% exposure"
        )
