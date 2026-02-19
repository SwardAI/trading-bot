import math
from datetime import datetime, timedelta, timezone

from src.core.database import Database
from src.core.logger import setup_logger

logger = setup_logger("journal.performance")


class PerformanceTracker:
    """Calculates trading performance metrics from the trades database.

    Args:
        db: Database instance.
    """

    def __init__(self, db: Database):
        self.db = db

    def get_strategy_metrics(self, strategy: str, days: int = 1) -> dict:
        """Get performance metrics for a strategy over a time period.

        Args:
            strategy: Strategy name ("grid", "momentum", or "all").
            days: Number of days to look back.

        Returns:
            Dict with num_trades, wins, losses, gross_profit, gross_loss,
            net_pnl, fees, win_rate, profit_factor.
        """
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        if strategy == "all":
            trades = self.db.fetch_all(
                "SELECT * FROM trades WHERE timestamp >= ? AND pnl_usd IS NOT NULL",
                (since,),
            )
        else:
            trades = self.db.fetch_all(
                "SELECT * FROM trades WHERE strategy = ? AND timestamp >= ? AND pnl_usd IS NOT NULL",
                (strategy, since),
            )

        if not trades:
            return self._empty_metrics()

        wins = [t for t in trades if t["pnl_usd"] > 0]
        losses = [t for t in trades if t["pnl_usd"] <= 0]
        gross_profit = sum(t["pnl_usd"] for t in wins)
        gross_loss = abs(sum(t["pnl_usd"] for t in losses))
        total_fees = sum(t["fee_usd"] for t in trades)

        return {
            "num_trades": len(trades),
            "num_wins": len(wins),
            "num_losses": len(losses),
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "net_pnl": gross_profit - gross_loss,
            "fees_paid": total_fees,
            "win_rate": len(wins) / len(trades) * 100 if trades else 0,
            "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
            "avg_win": gross_profit / len(wins) if wins else 0,
            "avg_loss": gross_loss / len(losses) if losses else 0,
        }

    def get_daily_pnl(self) -> float:
        """Get today's total P&L."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        result = self.db.fetch_one(
            "SELECT COALESCE(SUM(pnl_usd), 0) as pnl FROM trades WHERE date(timestamp) = ? AND pnl_usd IS NOT NULL",
            (today,),
        )
        return result["pnl"] if result else 0.0

    def get_weekly_pnl(self) -> float:
        """Get this week's total P&L."""
        now = datetime.now(timezone.utc)
        monday = (now - timedelta(days=now.weekday())).strftime("%Y-%m-%d")
        result = self.db.fetch_one(
            "SELECT COALESCE(SUM(pnl_usd), 0) as pnl FROM trades WHERE date(timestamp) >= ? AND pnl_usd IS NOT NULL",
            (monday,),
        )
        return result["pnl"] if result else 0.0

    def get_monthly_pnl(self) -> float:
        """Get this month's total P&L."""
        month_start = datetime.now(timezone.utc).strftime("%Y-%m-01")
        result = self.db.fetch_one(
            "SELECT COALESCE(SUM(pnl_usd), 0) as pnl FROM trades WHERE date(timestamp) >= ? AND pnl_usd IS NOT NULL",
            (month_start,),
        )
        return result["pnl"] if result else 0.0

    def get_max_drawdown(self, days: int = 30) -> float:
        """Calculate maximum drawdown percentage over a period.

        Uses account snapshots to find peak-to-trough decline.

        Args:
            days: Number of days to look back.

        Returns:
            Max drawdown as a positive percentage (e.g., 5.0 means -5%).
        """
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        snapshots = self.db.fetch_all(
            "SELECT total_balance_usd FROM account_snapshots WHERE timestamp >= ? ORDER BY timestamp",
            (since,),
        )

        if not snapshots:
            return 0.0

        peak = snapshots[0]["total_balance_usd"]
        max_dd = 0.0

        for snap in snapshots:
            balance = snap["total_balance_usd"]
            if balance > peak:
                peak = balance
            drawdown = (peak - balance) / peak * 100 if peak > 0 else 0
            max_dd = max(max_dd, drawdown)

        return max_dd

    def get_sharpe_ratio(self, days: int = 30, risk_free_rate: float = 0.04) -> float:
        """Calculate annualized Sharpe ratio from daily returns.

        Args:
            days: Number of days to look back.
            risk_free_rate: Annual risk-free rate (default 4%).

        Returns:
            Annualized Sharpe ratio.
        """
        since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
        daily_pnls = self.db.fetch_all(
            """SELECT date(timestamp) as day, SUM(pnl_usd) as daily_pnl
            FROM trades WHERE date(timestamp) >= ? AND pnl_usd IS NOT NULL
            GROUP BY date(timestamp) ORDER BY day""",
            (since,),
        )

        if len(daily_pnls) < 2:
            return 0.0

        returns = [d["daily_pnl"] for d in daily_pnls]
        avg_return = sum(returns) / len(returns)
        std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5

        if std_return == 0:
            return 0.0

        daily_rf = risk_free_rate / 365
        sharpe = (avg_return - daily_rf) / std_return * math.sqrt(365)
        return sharpe

    def save_daily_metrics(self):
        """Aggregate today's performance and save to daily_metrics table."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        for strategy in ["grid", "momentum", "funding"]:
            metrics = self.get_strategy_metrics(strategy, days=1)
            if metrics["num_trades"] == 0:
                continue

            self.db.execute(
                """INSERT INTO daily_metrics
                (date, strategy, num_trades, num_wins, num_losses,
                 gross_profit_usd, gross_loss_usd, net_pnl_usd, fees_paid_usd)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    today, strategy, metrics["num_trades"],
                    metrics["num_wins"], metrics["num_losses"],
                    metrics["gross_profit"], metrics["gross_loss"],
                    metrics["net_pnl"], metrics["fees_paid"],
                ),
            )

        logger.info(f"Daily metrics saved for {today}")

    def _empty_metrics(self) -> dict:
        return {
            "num_trades": 0, "num_wins": 0, "num_losses": 0,
            "gross_profit": 0, "gross_loss": 0, "net_pnl": 0,
            "fees_paid": 0, "win_rate": 0, "profit_factor": 0,
            "avg_win": 0, "avg_loss": 0,
        }
