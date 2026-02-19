"""Generate performance reports from the trading database.

Usage:
    python scripts/generate_report.py                  # Daily report
    python scripts/generate_report.py --weekly         # Weekly summary
    python scripts/generate_report.py --metrics grid   # Strategy metrics
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.database import Database
from src.journal.performance import PerformanceTracker
from src.journal.reporter import Reporter


def main():
    parser = argparse.ArgumentParser(description="Generate performance reports")
    parser.add_argument("--weekly", action="store_true", help="Generate weekly summary")
    parser.add_argument("--metrics", type=str, help="Show metrics for a strategy (grid, momentum, all)")
    parser.add_argument("--days", type=int, default=1, help="Lookback period in days (default: 1)")
    parser.add_argument("--db", default="data/bot.db", help="Database path")
    args = parser.parse_args()

    db = Database(args.db)
    db.init_db()
    perf = PerformanceTracker(db)
    reporter = Reporter(db, perf)

    if args.metrics:
        metrics = perf.get_strategy_metrics(args.metrics, days=args.days)
        print(f"\n{'=' * 50}")
        print(f"  {args.metrics.upper()} Strategy Metrics ({args.days} day{'s' if args.days > 1 else ''})")
        print(f"{'=' * 50}")
        print(f"  Trades:        {metrics['num_trades']}")
        print(f"  Wins:          {metrics['num_wins']}")
        print(f"  Losses:        {metrics['num_losses']}")
        print(f"  Win Rate:      {metrics['win_rate']:.1f}%")
        print(f"  Net P&L:       ${metrics['net_pnl']:,.2f}")
        print(f"  Gross Profit:  ${metrics['gross_profit']:,.2f}")
        print(f"  Gross Loss:    ${metrics['gross_loss']:,.2f}")
        print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"  Fees Paid:     ${metrics['fees_paid']:,.2f}")
        print(f"  Avg Win:       ${metrics['avg_win']:,.2f}")
        print(f"  Avg Loss:      ${metrics['avg_loss']:,.2f}")
        print(f"{'=' * 50}")
    elif args.weekly:
        report = reporter.generate_weekly_summary()
        # Strip HTML tags for console display
        report = report.replace("<b>", "").replace("</b>", "")
        print(report)
    else:
        report = reporter.generate_daily_report()
        report = report.replace("<b>", "").replace("</b>", "")
        print(report)

    db.close()


if __name__ == "__main__":
    main()
