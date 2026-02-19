from src.backtest.engine import BacktestResult
from src.core.logger import setup_logger

logger = setup_logger("backtest.analyzer")


def print_backtest_report(result: BacktestResult):
    """Print a formatted backtest report to console.

    Args:
        result: BacktestResult from the engine.
    """
    sign = "+" if result.total_return_pct >= 0 else ""

    print("=" * 60)
    print(f"  BACKTEST RESULTS: {result.strategy.upper()}")
    print("=" * 60)
    print(f"  Period:          {result.start_date[:10]} to {result.end_date[:10]}")
    print(f"  Initial Capital: ${result.initial_capital:,.2f}")
    print(f"  Final Capital:   ${result.final_capital:,.2f}")
    print(f"  Total Return:    {sign}{result.total_return_pct:.2f}%")
    print(f"  Net Profit:      {sign}${result.net_profit:,.2f}")
    print("-" * 60)
    print(f"  Total Trades:    {result.total_trades}")
    print(f"  Winning Trades:  {result.winning_trades}")
    print(f"  Losing Trades:   {result.losing_trades}")
    print(f"  Win Rate:        {result.win_rate:.1f}%")
    print(f"  Profit Factor:   {result.profit_factor:.2f}")
    print(f"  Avg Trade P&L:   ${result.avg_trade_pnl:,.2f}")
    print("-" * 60)
    print(f"  Max Drawdown:    {result.max_drawdown_pct:.2f}%")
    print(f"  Sharpe Ratio:    {result.sharpe_ratio:.2f}")
    print("-" * 60)
    print(f"  Gross Profit:    ${result.gross_profit:,.2f}")
    print(f"  Gross Loss:      ${result.gross_loss:,.2f}")
    print("=" * 60)


def compare_with_buy_and_hold(result: BacktestResult, df) -> dict:
    """Compare strategy performance against buy-and-hold.

    Args:
        result: BacktestResult from strategy.
        df: OHLCV DataFrame used for the backtest.

    Returns:
        Dict with buy_hold_return, strategy_return, outperformance.
    """
    start_price = df.iloc[0]["close"]
    end_price = df.iloc[-1]["close"]

    buy_hold_return = (end_price - start_price) / start_price * 100
    outperformance = result.total_return_pct - buy_hold_return

    print(f"\n  Buy & Hold:      {'+' if buy_hold_return >= 0 else ''}{buy_hold_return:.2f}%")
    print(f"  Strategy:        {'+' if result.total_return_pct >= 0 else ''}{result.total_return_pct:.2f}%")
    print(f"  Outperformance:  {'+' if outperformance >= 0 else ''}{outperformance:.2f}%")
    print("=" * 60)

    return {
        "buy_hold_return": buy_hold_return,
        "strategy_return": result.total_return_pct,
        "outperformance": outperformance,
    }


def get_monthly_returns(result: BacktestResult) -> dict[str, float]:
    """Calculate monthly returns from the equity curve.

    Args:
        result: BacktestResult with trades.

    Returns:
        Dict mapping "YYYY-MM" to return percentage.
    """
    monthly = {}
    for trade in result.trades:
        if trade.pnl == 0:
            continue
        month = str(trade.timestamp)[:7]
        monthly[month] = monthly.get(month, 0) + trade.pnl

    # Convert to percentages
    for month in monthly:
        monthly[month] = monthly[month] / result.initial_capital * 100

    return monthly


def get_trade_stats(result: BacktestResult) -> dict:
    """Get detailed trade statistics.

    Args:
        result: BacktestResult.

    Returns:
        Dict with detailed stats.
    """
    pnls = [t.pnl for t in result.trades if t.pnl != 0]

    if not pnls:
        return {"count": 0}

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    return {
        "count": len(pnls),
        "avg_pnl": sum(pnls) / len(pnls),
        "median_pnl": sorted(pnls)[len(pnls) // 2],
        "best_trade": max(pnls),
        "worst_trade": min(pnls),
        "avg_win": sum(wins) / len(wins) if wins else 0,
        "avg_loss": sum(losses) / len(losses) if losses else 0,
        "largest_win": max(wins) if wins else 0,
        "largest_loss": min(losses) if losses else 0,
        "consecutive_wins_max": _max_consecutive(pnls, positive=True),
        "consecutive_losses_max": _max_consecutive(pnls, positive=False),
    }


def _max_consecutive(pnls: list[float], positive: bool) -> int:
    """Count max consecutive wins or losses."""
    max_streak = 0
    current = 0
    for p in pnls:
        if (positive and p > 0) or (not positive and p <= 0):
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak
