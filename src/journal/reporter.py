from datetime import datetime, timezone

from src.core.database import Database
from src.core.logger import setup_logger
from src.journal.performance import PerformanceTracker
from src.risk.risk_manager import RiskManager

logger = setup_logger("journal.reporter")


class Reporter:
    """Generates daily and weekly performance reports.

    Args:
        db: Database instance.
        performance: PerformanceTracker instance.
        risk_manager: RiskManager instance (for exposure/risk status).
    """

    def __init__(self, db: Database, performance: PerformanceTracker, risk_manager: RiskManager | None = None):
        self.db = db
        self.performance = performance
        self.risk_manager = risk_manager

    def generate_daily_report(self) -> str:
        """Generate the daily performance report.

        Returns:
            Formatted report string.
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Portfolio balance
        snapshot = self.db.fetch_one(
            "SELECT * FROM account_snapshots ORDER BY timestamp DESC LIMIT 1"
        )
        total_balance = snapshot["total_balance_usd"] if snapshot else 0
        daily_pnl = self.performance.get_daily_pnl()
        daily_pnl_pct = (daily_pnl / total_balance * 100) if total_balance > 0 else 0

        # Strategy breakdowns
        grid_metrics = self.performance.get_strategy_metrics("grid", days=1)
        momentum_metrics = self.performance.get_strategy_metrics("momentum", days=1)
        mtf_donchian_metrics = self.performance.get_strategy_metrics("mtf_donchian", days=1)
        funding_metrics = self.performance.get_strategy_metrics("funding", days=1)

        # Grid round trips
        grid_rts = grid_metrics["num_wins"] + grid_metrics["num_losses"]

        # Open positions
        open_momentum = self.db.fetch_all(
            "SELECT * FROM momentum_positions WHERE status = 'open'"
        )
        open_mtf_donchian = self.db.fetch_all(
            "SELECT * FROM mtf_donchian_positions WHERE status = 'open'"
        )
        open_funding = self.db.fetch_all(
            "SELECT * FROM funding_positions WHERE status IN ('open', 'closing')"
        )
        grid_states = self.db.fetch_all("SELECT * FROM grid_state")

        # Risk status
        risk_status = self.risk_manager.get_status() if self.risk_manager else {}
        balance_info = risk_status.get("balance", {})
        cb_status = risk_status.get("circuit_breaker", {})

        # Get total trades today
        total_trades_result = self.db.fetch_one(
            "SELECT COUNT(*) as count FROM trades WHERE date(timestamp) = ?",
            (today,),
        )
        total_trades = total_trades_result["count"] if total_trades_result else 0

        # Total fees today
        fees_result = self.db.fetch_one(
            "SELECT COALESCE(SUM(fee_usd), 0) as fees FROM trades WHERE date(timestamp) = ?",
            (today,),
        )
        total_fees = fees_result["fees"] if fees_result else 0

        # Weekly P&L
        weekly_pnl = self.performance.get_weekly_pnl()
        weekly_pnl_pct = (weekly_pnl / total_balance * 100) if total_balance > 0 else 0

        # Format report
        pnl_emoji = "\U0001f4c8" if daily_pnl >= 0 else "\U0001f4c9"  # 📈 or 📉
        sign = "+" if daily_pnl >= 0 else ""

        lines = [
            f"\U0001f4ca <b>Daily Report \u2014 {today}</b>",
            "",
            f"\U0001f4b0 Portfolio: ${total_balance:,.2f} ({sign}${daily_pnl:,.2f} / {sign}{daily_pnl_pct:.2f}%)",
            "",
            "<b>Strategy Breakdown:</b>",
            f"  Grid:     {sign}${grid_metrics['net_pnl']:,.2f} ({grid_rts} round trips)",
            f"  Momentum: {sign}${momentum_metrics['net_pnl']:,.2f} ({momentum_metrics['num_wins']}W / {momentum_metrics['num_losses']}L)",
            f"  MTF Donch: {'+' if mtf_donchian_metrics['net_pnl'] >= 0 else ''}${mtf_donchian_metrics['net_pnl']:,.2f} ({mtf_donchian_metrics['num_wins']}W / {mtf_donchian_metrics['num_losses']}L)",
        ]

        # Funding line — show collected funding if active, otherwise "not active"
        total_funding_collected = sum(p.get("funding_collected_usd", 0) or 0 for p in open_funding)
        if open_funding or funding_metrics["net_pnl"] != 0:
            funding_sign = "+" if funding_metrics["net_pnl"] >= 0 else ""
            lines.append(
                f"  Funding:  {funding_sign}${funding_metrics['net_pnl']:,.2f} "
                f"({len(open_funding)} positions, ${total_funding_collected:,.2f} collected)"
            )
        else:
            lines.append(f"  Funding:  $0.00 (not active)")

        # Open positions
        if open_momentum or open_mtf_donchian or open_funding or grid_states:
            lines.append("")
            lines.append(f"{pnl_emoji} <b>Open Positions:</b>")
            for gs in grid_states:
                if gs["inventory_amount"] and gs["inventory_amount"] > 0:
                    inv_value = gs["inventory_amount"] * (gs["inventory_avg_price"] or 0)
                    lines.append(f"  Grid {gs['pair']}: {gs['inventory_amount']:.6f} (${inv_value:,.2f})")
            for pos in open_mtf_donchian:
                side_label = pos['side'].upper()
                mkt = "futures" if pos.get('market_type') == 'futures' else "spot"
                lines.append(
                    f"  MTF: {side_label} {pos['pair']} ({mkt}) "
                    f"@ ${pos['entry_price']:,.2f} (stop: ${pos['current_stop']:,.2f})"
                )
            for pos in open_momentum:
                lines.append(
                    f"  Momentum: {pos['side'].upper()} {pos['pair']} "
                    f"@ ${pos['entry_price']:,.2f} (stop: ${pos['current_stop']:,.2f})"
                )
            for pos in open_funding:
                collected = pos.get("funding_collected_usd", 0) or 0
                fees = pos.get("total_fees_usd", 0) or 0
                net = collected - fees
                lines.append(
                    f"  Funding {pos['pair']}: ${pos['notional_usd']:,.0f} notional, "
                    f"${collected:,.2f} collected (net ${net:,.2f})"
                )
        # Risk status
        exposure_pct = balance_info.get("exposure_pct", 0)
        max_exposure = 60
        live_total = balance_info.get("total_usd", 0)
        reserve_pct = (balance_info.get("free_usd", 0) / live_total * 100) if live_total > 0 else 0

        lines.extend([
            "",
            "\u26a1 <b>Risk Status:</b>",
            f"  Total exposure: {exposure_pct:.1f}% (limit: {max_exposure}%) {'✅' if exposure_pct < max_exposure else '❌'}",
            f"  Daily P&L: {sign}{daily_pnl_pct:.2f}% (limit: -3%) {'✅' if daily_pnl_pct > -3 else '❌'}",
            f"  Weekly P&L: {'+' if weekly_pnl >= 0 else ''}{weekly_pnl_pct:.2f}% (limit: -7%) {'✅' if weekly_pnl_pct > -7 else '❌'}",
            f"  Reserve cash: {reserve_pct:.0f}% (min: 20%) {'✅' if reserve_pct >= 20 else '❌'}",
        ])

        if cb_status.get("active"):
            lines.append(f"  \U0001f6a8 Circuit breaker: {cb_status['level'].upper()} ACTIVE")

        # 24h stats
        lines.extend([
            "",
            "\U0001f504 <b>24h Stats:</b>",
            f"  Total trades: {total_trades}",
            f"  Total fees: ${total_fees:,.2f}",
        ])

        report = "\n".join(lines)
        logger.info(f"Daily report generated for {today}")
        return report

    def generate_weekly_summary(self) -> str:
        """Generate a weekly performance summary.

        Returns:
            Formatted weekly summary string.
        """
        weekly_pnl = self.performance.get_weekly_pnl()
        max_dd = self.performance.get_max_drawdown(days=7)
        sharpe = self.performance.get_sharpe_ratio(days=7)

        grid_weekly = self.performance.get_strategy_metrics("grid", days=7)
        momentum_weekly = self.performance.get_strategy_metrics("momentum", days=7)
        mtf_weekly = self.performance.get_strategy_metrics("mtf_donchian", days=7)
        funding_weekly = self.performance.get_strategy_metrics("funding", days=7)

        sign = "+" if weekly_pnl >= 0 else ""

        lines = [
            "\U0001f4ca <b>Weekly Summary</b>",
            "",
            f"Net P&L: {sign}${weekly_pnl:,.2f}",
            f"Max Drawdown: {max_dd:.2f}%",
            f"Sharpe Ratio: {sharpe:.2f}",
            "",
            "<b>Grid:</b>",
            f"  Trades: {grid_weekly['num_trades']} | Win rate: {grid_weekly['win_rate']:.0f}%",
            f"  P&L: ${grid_weekly['net_pnl']:,.2f} | Profit factor: {grid_weekly['profit_factor']:.2f}",
            "",
            "<b>Momentum:</b>",
            f"  Trades: {momentum_weekly['num_trades']} | Win rate: {momentum_weekly['win_rate']:.0f}%",
            f"  P&L: ${momentum_weekly['net_pnl']:,.2f} | Profit factor: {momentum_weekly['profit_factor']:.2f}",
            "",
            "<b>MTF Donchian:</b>",
            f"  Trades: {mtf_weekly['num_trades']} | Win rate: {mtf_weekly['win_rate']:.0f}%",
            f"  P&L: ${mtf_weekly['net_pnl']:,.2f} | Profit factor: {mtf_weekly['profit_factor']:.2f}",
        ]

        # Only show funding section if there was activity
        if funding_weekly["num_trades"] > 0 or funding_weekly["net_pnl"] != 0:
            # Get total funding collected this week
            weekly_funding_collected = self.db.fetch_one(
                """SELECT COALESCE(SUM(funding_collected_usd), 0) as collected,
                          COALESCE(SUM(total_fees_usd), 0) as fees
                   FROM funding_positions
                   WHERE entry_time >= datetime('now', '-7 days')
                      OR status IN ('open', 'closing')"""
            )
            collected = weekly_funding_collected["collected"] if weekly_funding_collected else 0
            fees = weekly_funding_collected["fees"] if weekly_funding_collected else 0

            lines.extend([
                "",
                "<b>Funding:</b>",
                f"  Positions opened: {funding_weekly['num_trades']}",
                f"  Funding collected: ${collected:,.2f} | Fees: ${fees:,.2f}",
                f"  Net P&L: ${funding_weekly['net_pnl']:,.2f}",
            ])

        return "\n".join(lines)
