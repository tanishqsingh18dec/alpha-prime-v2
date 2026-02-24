"""
Trade Analyzer — Post-Trade Feedback Loop for Alpha Prime v2

After each trade closes, this module looks back 24h later to evaluate:
  - Did we exit too early? (price kept going up → we left money on the table)
  - Did we exit at a good time? (price dropped after → good call)
  - What type of exit signal was most profitable?

The bot uses this feedback to dynamically adjust exit parameters
(MIN_RETURN_TO_HOLD, TRAILING_STOP_PCT) so it self-corrects over time.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


class TradeAnalyzer:
    """
    Reads closed trades, checks what happened 24h after exit,
    and adjusts bot parameters based on patterns.
    """

    FEEDBACK_FILE = 'trade_feedback.json'
    MIN_TRADES_FOR_FEEDBACK = 10  # Need at least 10 closed trades to draw conclusions

    def __init__(self, trade_history_file, scanner):
        self.trade_file = trade_history_file
        self.scanner = scanner
        self.feedback = {
            'total_analyzed': 0,
            'exited_too_early': 0,     # price went up >5% after we sold
            'good_exit': 0,            # price dropped >3% after we sold
            'neutral_exit': 0,         # price moved <3% either way
            'avg_missed_upside': 0.0,  # avg % we left on the table
            'avg_avoided_downside': 0.0,
            'by_reason': {},           # breakdown by exit reason
        }

    def analyze_closed_trades(self):
        """
        Look at SELL trades older than 24h and check what happened after.
        Returns adjustment recommendations.
        """
        if not Path(self.trade_file).exists():
            return None

        try:
            df = pd.read_csv(self.trade_file)
        except Exception:
            return None

        if len(df) < self.MIN_TRADES_FOR_FEEDBACK:
            return None

        # Filter to SELL trades older than 24h
        sells = df[df['action'] == 'SELL'].copy()
        if sells.empty:
            return None

        sells['timestamp'] = pd.to_datetime(sells['timestamp'])
        cutoff = datetime.now() - timedelta(hours=24)
        eligible = sells[sells['timestamp'] < cutoff]

        if len(eligible) < 5:
            return None

        # For each closed trade, check what the price did 24h after
        results = []
        for _, trade in eligible.iterrows():
            symbol = trade['symbol']
            exit_price = trade['price']
            reason = trade.get('reason', 'unknown')

            # Fetch current price (approximates "what happened after")
            try:
                current_price = self.scanner.get_current_price(symbol, 'binance')
                if current_price and exit_price > 0:
                    price_change_after = (current_price - exit_price) / exit_price
                    results.append({
                        'symbol': symbol,
                        'exit_price': exit_price,
                        'price_after': current_price,
                        'change_pct': price_change_after,
                        'reason': reason,
                    })
            except Exception:
                continue

        if not results:
            return None

        # Compute statistics
        changes = [r['change_pct'] for r in results]
        too_early = [r for r in results if r['change_pct'] > 0.05]   # >5% upside missed
        good_exits = [r for r in results if r['change_pct'] < -0.03] # >3% drop avoided

        self.feedback['total_analyzed'] = len(results)
        self.feedback['exited_too_early'] = len(too_early)
        self.feedback['good_exit'] = len(good_exits)
        self.feedback['neutral_exit'] = len(results) - len(too_early) - len(good_exits)

        if too_early:
            self.feedback['avg_missed_upside'] = np.mean([r['change_pct'] for r in too_early])
        if good_exits:
            self.feedback['avg_avoided_downside'] = np.mean([r['change_pct'] for r in good_exits])

        # Breakdown by exit reason
        reason_stats = {}
        for r in results:
            reason_key = r['reason'].split('|')[0].strip()[:30]  # truncate
            if reason_key not in reason_stats:
                reason_stats[reason_key] = {'count': 0, 'avg_change': 0, 'changes': []}
            reason_stats[reason_key]['count'] += 1
            reason_stats[reason_key]['changes'].append(r['change_pct'])

        for key in reason_stats:
            reason_stats[key]['avg_change'] = np.mean(reason_stats[key]['changes'])
            del reason_stats[key]['changes']  # clean up for JSON

        self.feedback['by_reason'] = reason_stats

        # Generate adjustment recommendations
        return self._compute_adjustments()

    def _compute_adjustments(self):
        """
        Based on trade analysis, recommend parameter adjustments.
        Returns dict with parameter name → new value.
        """
        adjustments = {}
        total = self.feedback['total_analyzed']
        if total < self.MIN_TRADES_FOR_FEEDBACK:
            return adjustments

        early_exit_rate = self.feedback['exited_too_early'] / total

        # If >40% of exits were too early → we're cutting winners too fast
        if early_exit_rate > 0.40:
            # Tighten trailing stop less aggressively (widen from 6% to 7%)
            adjustments['TRAILING_STOP_PCT'] = 0.07
            # Raise the bar for what counts as "worth holding"
            adjustments['MIN_RETURN_TO_HOLD'] = 0.03

        # If >50% were good exits → our exits are working well, tighten them
        good_exit_rate = self.feedback['good_exit'] / total
        if good_exit_rate > 0.50:
            adjustments['TRAILING_STOP_PCT'] = 0.05
            adjustments['MIN_RETURN_TO_HOLD'] = 0.015

        return adjustments

    def get_kelly_stats(self):
        """
        Compute win rate and win/loss ratio from trade history
        for the Dynamic Kelly Criterion.

        Returns:
            (win_rate, win_loss_ratio, total_trades) or None if insufficient data.
        """
        if not Path(self.trade_file).exists():
            return None

        try:
            df = pd.read_csv(self.trade_file)
        except Exception:
            return None

        sells = df[df['action'] == 'SELL']
        if len(sells) < 10:
            return None

        wins = sells[sells['pnl'] > 0]
        losses = sells[sells['pnl'] <= 0]

        if len(losses) == 0 or len(wins) == 0:
            return None

        win_rate = len(wins) / len(sells)
        avg_win = wins['pnl'].mean()
        avg_loss = abs(losses['pnl'].mean())

        if avg_loss == 0:
            return None

        win_loss_ratio = avg_win / avg_loss

        return win_rate, win_loss_ratio, len(sells)
