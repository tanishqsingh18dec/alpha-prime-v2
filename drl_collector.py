#!/usr/bin/env python3
"""
DRL Training Data Collector for Alpha Prime v3  (Phase 5 Prep)
===============================================================
Collects and stores rich state-action-reward (SAR) tuples at every
slow_loop decision point. This data will train the SAC (Soft Actor-Critic)
agent once 3+ months of observations are accumulated.

What is recorded per slow_loop cycle:
  - Timestamp and cycle number
  - Market regime (rule-based + HMM + confidence)
  - Portfolio state (balance, value, positions, weights)
  - Per-coin scoring features (momentum, sentiment, OFI, VPIN, on-chain, etc.)
  - Actions taken (buys, sells, trims, holds)
  - Reward signal (portfolio return since last observation)

Storage:
  drl_training_data/state_snapshots.jsonl  — one JSON object per line (JSONL)
  drl_training_data/feature_matrix.csv     — flat numeric features for ML

Usage from slow_loop:
    drl_collector.record_snapshot(
        regime_info=..., portfolio_state=..., scored_coins=..., actions=...
    )
"""

import os
import json
import csv
from datetime import datetime

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'item'):
            return obj.item()
        return super(NumpyEncoder, self).default(obj)

from pathlib import Path

# ── CONFIG ─────────────────────────────────────────────────────────────────

DATA_DIR = 'drl_training_data'
SNAPSHOTS_FILE = os.path.join(DATA_DIR, 'state_snapshots.jsonl')
FEATURES_FILE  = os.path.join(DATA_DIR, 'feature_matrix.csv')

# Feature columns for the flat CSV (order matters for DRL state vector)
FEATURE_COLUMNS = [
    'timestamp',
    'cycle_number',
    # Market regime
    'regime_rule',           # RISK_ON, RISK_OFF, CHOP
    'regime_hmm',            # BULL, BEAR, SIDEWAYS
    'hmm_confidence',        # 0.0-1.0
    # Portfolio state
    'portfolio_value',
    'cash_balance',
    'num_positions',
    'portfolio_utilization',  # invested / total value
    # BTC benchmark
    'btc_price',
    'btc_return_1h',          # BTC return since last observation
    # Aggregate signals (averages across held positions)
    'avg_momentum_score',
    'avg_sentiment_score',
    'avg_fuel_score',
    'avg_rvol_score',
    'avg_ofi_score',
    'avg_onchain_score',
    'avg_vpin_score',
    # Risk metrics
    'max_position_pct',       # Largest position as % of portfolio
    'avg_pnl_pct',            # Average unrealized P&L across positions
    'num_toxic_coins',        # Coins with is_toxic=True
    'num_spoofed_coins',      # Coins with is_spoofed=True
    # Actions taken this cycle
    'num_buys',
    'num_sells',
    'num_trims',
    'num_holds',
    # Reward signal
    'portfolio_return_pct',   # % change since last observation
    'excess_return_pct',      # portfolio return - BTC return (alpha)
]


class DRLDataCollector:
    """
    Collects state-action-reward data at every decision point for future
    DRL (Soft Actor-Critic) training.

    Data is stored in two formats:
    1. JSONL (state_snapshots.jsonl) — full rich snapshots with all details
    2. CSV (feature_matrix.csv) — flat numeric feature vectors for ML training

    The collector is designed to be zero-impact on trading performance:
    - Writes are append-only (no reads during operation)
    - Failures are silently caught (never interrupts trading)
    - File handles are opened/closed per write (no leaked resources)
    """

    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        self._cycle_count = 0
        self._prev_portfolio_value = None
        self._prev_btc_price = None
        self._initialized = False
        self._init_csv()

    def _init_csv(self):
        """Create CSV with headers if it doesn't exist."""
        if not os.path.exists(FEATURES_FILE):
            with open(FEATURES_FILE, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(FEATURE_COLUMNS)
            self._initialized = True
        else:
            # Count existing rows to resume cycle numbering
            try:
                with open(FEATURES_FILE, 'r') as f:
                    self._cycle_count = sum(1 for _ in f) - 1  # Subtract header
                    self._cycle_count = max(0, self._cycle_count)
            except Exception:
                self._cycle_count = 0
            self._initialized = True

    def record_snapshot(self, regime_info: dict, portfolio_state: dict,
                         scored_coins: list, actions: dict,
                         btc_price: float = 0.0):
        """
        Record a full state snapshot at a decision point.

        Args:
            regime_info: {
                'rule_regime': str,     # e.g., 'RISK_ON'
                'hmm_regime': str,      # e.g., 'BULL'
                'hmm_confidence': float,# e.g., 0.85
                'btc_price': float,
            }
            portfolio_state: {
                'balance': float,       # Cash balance
                'total_value': float,   # Balance + position values
                'positions': dict,      # {symbol: position_dict}
            }
            scored_coins: [
                {
                    'symbol': str,
                    'final_score': float,
                    'momentum_score': float,
                    'sentiment_score': float,
                    'fuel_score': float,
                    'rvol_score': float,
                    'ofi_score': float,
                    'onchain_score': float,
                    'vpin_score': float,
                    'is_toxic': bool,
                    'is_spoofed': bool,
                    ...
                }, ...
            ]
            actions: {
                'buys': [symbol, ...],
                'sells': [symbol, ...],
                'trims': [symbol, ...],
                'holds': [symbol, ...],
            }
            btc_price: Current BTC price for benchmark
        """
        try:
            self._cycle_count += 1
            timestamp = datetime.now().isoformat()

            # ── Compute reward signal ──
            portfolio_value = portfolio_state.get('total_value', 0)
            btc_p = btc_price or regime_info.get('btc_price', 0)

            portfolio_return = 0.0
            btc_return = 0.0
            if self._prev_portfolio_value and self._prev_portfolio_value > 0:
                portfolio_return = (portfolio_value - self._prev_portfolio_value) / self._prev_portfolio_value * 100
            if self._prev_btc_price and self._prev_btc_price > 0 and btc_p > 0:
                btc_return = (btc_p - self._prev_btc_price) / self._prev_btc_price * 100

            self._prev_portfolio_value = portfolio_value
            self._prev_btc_price = btc_p

            excess_return = portfolio_return - btc_return

            # ── Aggregate position metrics ──
            positions = portfolio_state.get('positions', {})
            num_positions = len(positions)
            invested_value = sum(
                p.get('quantity', 0) * p.get('current_price', p.get('entry_price', 0))
                for p in positions.values()
            )
            utilization = invested_value / portfolio_value if portfolio_value > 0 else 0

            # Largest position as % of portfolio
            position_values = [
                p.get('quantity', 0) * p.get('current_price', p.get('entry_price', 0))
                for p in positions.values()
            ]
            max_pos_pct = max(position_values) / portfolio_value if position_values and portfolio_value > 0 else 0

            # Average unrealized P&L
            pnls = []
            for p in positions.values():
                entry = p.get('entry_price', 0)
                current = p.get('current_price', entry)
                if entry > 0:
                    pnls.append((current - entry) / entry * 100)
            avg_pnl = sum(pnls) / len(pnls) if pnls else 0

            # ── Aggregate scoring signals ──
            def avg_field(field):
                vals = [c.get(field, 0) for c in scored_coins if isinstance(c.get(field), (int, float))]
                return sum(vals) / len(vals) if vals else 0

            num_toxic = sum(1 for c in scored_coins if c.get('is_toxic'))
            num_spoofed = sum(1 for c in scored_coins if c.get('is_spoofed'))

            # ── Write JSONL (full rich snapshot) ──
            snapshot = {
                'timestamp': timestamp,
                'cycle': self._cycle_count,
                'regime': regime_info,
                'portfolio': {
                    'balance': portfolio_state.get('balance', 0),
                    'total_value': portfolio_value,
                    'positions': {
                        sym: {
                            'entry_price': p.get('entry_price', 0),
                            'quantity': p.get('quantity', 0),
                            'weight': p.get('weight', 0),
                            'cycles_held': p.get('cycles_held', 0),
                            'trim_count': p.get('trim_count', 0),
                            'entry_score': p.get('entry_score', 0),
                        }
                        for sym, p in positions.items()
                    },
                },
                'scored_coins': [
                    {k: v for k, v in c.items()
                     if k in ('symbol', 'final_score', 'momentum_score',
                              'sentiment_score', 'fuel_score', 'rvol_score',
                              'ofi_score', 'onchain_score', 'vpin_score',
                              'is_toxic', 'is_spoofed', 'ob_imbalance',
                              'price_zscore', 'trend_filter', 'funding_rate')}
                    for c in scored_coins[:20]  # Top 20 to limit file size
                ],
                'actions': actions,
                'reward': {
                    'portfolio_return_pct': round(portfolio_return, 4),
                    'btc_return_pct': round(btc_return, 4),
                    'excess_return_pct': round(excess_return, 4),
                },
            }

            with open(SNAPSHOTS_FILE, 'a') as f:
                f.write(json.dumps(snapshot, cls=NumpyEncoder) + '\n')

            # ── Write CSV (flat feature vector) ──
            row = {
                'timestamp': timestamp,
                'cycle_number': self._cycle_count,
                'regime_rule': regime_info.get('rule_regime', 'UNKNOWN'),
                'regime_hmm': regime_info.get('hmm_regime', 'UNKNOWN'),
                'hmm_confidence': round(regime_info.get('hmm_confidence', 0), 4),
                'portfolio_value': round(portfolio_value, 4),
                'cash_balance': round(portfolio_state.get('balance', 0), 4),
                'num_positions': num_positions,
                'portfolio_utilization': round(utilization, 4),
                'btc_price': round(btc_p, 2),
                'btc_return_1h': round(btc_return, 4),
                'avg_momentum_score': round(avg_field('momentum_score'), 4),
                'avg_sentiment_score': round(avg_field('sentiment_score'), 4),
                'avg_fuel_score': round(avg_field('fuel_score'), 4),
                'avg_rvol_score': round(avg_field('rvol_score'), 4),
                'avg_ofi_score': round(avg_field('ofi_score'), 4),
                'avg_onchain_score': round(avg_field('onchain_score'), 4),
                'avg_vpin_score': round(avg_field('vpin_score'), 4),
                'max_position_pct': round(max_pos_pct, 4),
                'avg_pnl_pct': round(avg_pnl, 4),
                'num_toxic_coins': num_toxic,
                'num_spoofed_coins': num_spoofed,
                'num_buys': len(actions.get('buys', [])),
                'num_sells': len(actions.get('sells', [])),
                'num_trims': len(actions.get('trims', [])),
                'num_holds': len(actions.get('holds', [])),
                'portfolio_return_pct': round(portfolio_return, 4),
                'excess_return_pct': round(excess_return, 4),
            }

            with open(FEATURES_FILE, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=FEATURE_COLUMNS)
                writer.writerow(row)

        except Exception as e:
            # NEVER interrupt trading — silently log and continue
            print(f"   ⚠️  DRL collector error: {e}")

    def get_stats(self) -> dict:
        """Return collection statistics for dashboard display."""
        try:
            snapshots_size = os.path.getsize(SNAPSHOTS_FILE) if os.path.exists(SNAPSHOTS_FILE) else 0
            features_size = os.path.getsize(FEATURES_FILE) if os.path.exists(FEATURES_FILE) else 0

            # Estimate time to reach 3 months (assuming hourly collection)
            hours_needed = 3 * 30 * 24  # ~2160 hours
            hours_collected = self._cycle_count
            pct_complete = min(100, (hours_collected / hours_needed) * 100)

            return {
                'cycles_collected': self._cycle_count,
                'snapshots_file_mb': round(snapshots_size / (1024 * 1024), 2),
                'features_file_mb': round(features_size / (1024 * 1024), 2),
                'hours_needed': hours_needed,
                'pct_complete': round(pct_complete, 1),
                'ready_for_training': self._cycle_count >= hours_needed,
            }
        except Exception:
            return {'cycles_collected': self._cycle_count, 'pct_complete': 0}
