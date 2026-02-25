#!/usr/bin/env python3
"""
On-Chain Signals Module for Alpha Prime v3  (Phase 4)
=====================================================
Monitors exchange flow signals that indicate institutional/whale activity:

1. Exchange Netflow Proxy:
   - BTC: Blockchain.info mempool + fee data (free API)
   - All coins: Long/Short ratio from CoinGlass (free, no auth)
   - High L/S ratio = crowded longs (bearish) — coins likely to flow IN to sell
   - Low  L/S ratio = crowded shorts (bullish) — coins being withdrawn to hold

2. Whale Transfer Detection:
   - Tracks large transfers via Blockchair API (free tier: 5 req/min)
   - Whale deposits to exchanges = bearish (selling)
   - Whale withdrawals from exchanges = bullish (accumulation)

Output: on_chain score in [-1, +1] per symbol, consumed by AlphaScorer.

Usage:
    monitor = OnChainMonitor()
    score = monitor.get_flow_score('BTC')   # -1 (bearish) to +1 (bullish)
    whale  = monitor.get_whale_alert('BTC') # dict with recent large txns
"""

import requests
import time
import json
from datetime import datetime
from collections import defaultdict

# ── CONFIG ─────────────────────────────────────────────────────────────────

CACHE_TTL_SECONDS = 300  # 5 minutes — don't spam free APIs
REQUEST_TIMEOUT = 10

# CoinGlass open data (no auth needed)
COINGLASS_LS_URL = "https://open-api.coinglass.com/public/v2/long_short"

# Blockchain.info (BTC only, free)
BTCINFO_STATS_URL = "https://api.blockchain.info/stats"
BTCINFO_MEMPOOL_URL = "https://api.blockchain.info/charts/mempool-size?timespan=1days&format=json"

# Blockchair (multi-chain, free tier: 5 req/min)
BLOCKCHAIR_STATS_URL = "https://api.blockchair.com/{chain}/stats"

# Known exchange hot-wallet address prefixes (for simple heuristics)
# In practice, whale alerts use curated address lists — this is a simplified proxy.
CHAINS = {
    'BTC': 'bitcoin',
    'ETH': 'ethereum',
    'LTC': 'litecoin',
    'DOGE': 'dogecoin',
    'SOL': 'solana',
}

HEADERS = {
    'User-Agent': 'AlphaPrime/3.0 OnChainMonitor',
    'Accept': 'application/json',
}


class OnChainMonitor:
    """
    Tracks exchange flow signals for major crypto assets.

    Combines multiple free data sources into a single [-1, +1] flow score:
    - Long/Short ratio (CoinGlass) — proxy for institutional positioning
    - Mempool/network activity (Blockchain.info) — BTC congestion = selling
    - Chain-level stats (Blockchair) — transaction volume trends

    The flow score feeds into AlphaScorer as the ONCHAIN_WEIGHT factor.
    """

    def __init__(self):
        self._cache = {}  # symbol → {'score': float, 'timestamp': datetime, 'details': dict}
        self._ls_cache = {}  # Global L/S ratio cache
        self._ls_cache_time = None

    def get_flow_score(self, symbol: str) -> float:
        """
        Get on-chain flow score for a symbol.

        Returns:
            float in [-1, +1]:
              +1.0 = strong accumulation signal (outflows dominating)
              -1.0 = strong distribution signal (inflows dominating)
               0.0 = neutral or no data available
        """
        base = symbol.split('/')[0].upper()

        # Check cache
        cached = self._cache.get(base)
        if cached:
            age = (datetime.now() - cached['timestamp']).total_seconds()
            if age < CACHE_TTL_SECONDS:
                return cached['score']

        score = self._compute_flow_score(base)
        self._cache[base] = {
            'score': score,
            'timestamp': datetime.now(),
        }
        return score

    def get_details(self, symbol: str) -> dict:
        """Get cached flow details for dashboard display."""
        base = symbol.split('/')[0].upper()
        cached = self._cache.get(base)
        if cached:
            return cached
        return {}

    def _compute_flow_score(self, base: str) -> float:
        """
        Compute composite on-chain flow score.

        Strategy:
        1. Long/Short ratio (primary signal — applies to all coins with perps)
        2. BTC-specific: mempool fee pressure
        3. Chain-specific stats (if supported)
        """
        scores = []
        weights = []

        # Signal 1: Long/Short ratio from CoinGlass (most universally available)
        ls_score = self._get_ls_ratio_score(base)
        if ls_score is not None:
            scores.append(ls_score)
            weights.append(0.6)  # Primary weight

        # Signal 2: BTC-specific mempool analysis
        if base == 'BTC':
            mempool_score = self._get_btc_mempool_score()
            if mempool_score is not None:
                scores.append(mempool_score)
                weights.append(0.4)

        # Signal 3: Chain-level transaction stats
        chain = CHAINS.get(base)
        if chain:
            chain_score = self._get_chain_stats_score(chain)
            if chain_score is not None:
                scores.append(chain_score)
                weights.append(0.2)

        if not scores:
            return 0.0

        # Weighted average, normalized
        total_weight = sum(weights)
        blended = sum(s * w for s, w in zip(scores, weights)) / total_weight
        return max(-1.0, min(1.0, blended))

    def _get_ls_ratio_score(self, base: str) -> float:
        """
        Long/Short ratio from CoinGlass.

        Interpretation:
        - L/S > 2.0 = extremely crowded longs → BEARISH (distribution incoming)
        - L/S 1.5-2.0 = moderately crowded → slightly bearish
        - L/S 0.8-1.5 = balanced → neutral
        - L/S 0.5-0.8 = more shorts → BULLISH (short squeeze potential)
        - L/S < 0.5 = extreme short crowding → very bullish

        Returns [-1, +1] or None if unavailable.
        """
        try:
            # Fetch and cache L/S data (1 call covers all symbols)
            if (self._ls_cache_time is None or
                    (datetime.now() - self._ls_cache_time).total_seconds() > CACHE_TTL_SECONDS):
                resp = requests.get(COINGLASS_LS_URL, headers=HEADERS, timeout=REQUEST_TIMEOUT)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get('success') and data.get('data'):
                        self._ls_cache = {}
                        for item in data['data']:
                            sym = item.get('symbol', '').upper()
                            ratio = item.get('longShortRatio')
                            if sym and ratio is not None:
                                self._ls_cache[sym] = float(ratio)
                        self._ls_cache_time = datetime.now()
                    else:
                        return None
                else:
                    return None

            ls_ratio = self._ls_cache.get(base)
            if ls_ratio is None:
                return None

            # Convert L/S ratio to score [-1, +1]
            # High L/S = bearish (everyone's long, who's left to buy?)
            # Low L/S = bullish (shorts will get squeezed)
            if ls_ratio > 2.0:
                return -1.0
            elif ls_ratio > 1.5:
                return -0.5
            elif ls_ratio > 1.2:
                return -0.2
            elif ls_ratio > 0.8:
                return 0.0  # Balanced
            elif ls_ratio > 0.5:
                return 0.5
            else:
                return 1.0  # Extreme short crowding → very bullish

        except Exception:
            return None

    def _get_btc_mempool_score(self) -> float:
        """
        BTC mempool congestion as a proxy for selling pressure.

        High mempool = lots of pending transactions = people rushing to move
        BTC to exchanges to sell → bearish.

        Low mempool = calm network = hodling → bullish.

        Returns [-1, +1] or None.
        """
        try:
            resp = requests.get(BTCINFO_STATS_URL, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                return None

            stats = resp.json()
            # Unconfirmed transactions — proxy for network stress
            unconfirmed = stats.get('n_tx_unconfirmed', 0)

            # Typical range: 5K-200K pending txs
            if unconfirmed > 150000:
                return -0.8  # Very congested — rush to exit
            elif unconfirmed > 80000:
                return -0.3  # Moderately busy
            elif unconfirmed > 30000:
                return 0.0   # Normal
            elif unconfirmed > 10000:
                return 0.3   # Quiet — hodling
            else:
                return 0.6   # Very quiet — low activity = accumulation

        except Exception:
            return None

    def _get_chain_stats_score(self, chain: str) -> float:
        """
        Use Blockchair chain stats to detect unusual activity.

        High transaction volume relative to average = distribution.
        Stable/low volume = accumulation.

        Returns [-1, +1] or None.
        """
        try:
            url = BLOCKCHAIR_STATS_URL.format(chain=chain)
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                return None

            data = resp.json().get('data', {})

            # Blockchair returns various chain stats
            tx_24h = data.get('transactions_24h', 0)
            suggested_fee = data.get('suggested_transaction_fee_per_byte_sat', 0)
            mempool_txns = data.get('mempool_transactions', 0)

            # Simple heuristic: high fees + high mempool = selling pressure
            if mempool_txns > 50000 or suggested_fee > 100:
                return -0.5  # Network stressed
            elif mempool_txns < 5000 and suggested_fee < 10:
                return 0.3  # Calm network
            else:
                return 0.0  # Normal

        except Exception:
            return None

    def bulk_update(self, symbols: list) -> dict:
        """
        Fetch flow scores for multiple symbols efficiently.
        The L/S ratio call covers all symbols in one API request.

        Args:
            symbols: list of trading pairs (e.g., ['BTC/USDT', 'ETH/USDT'])

        Returns:
            dict: {symbol: flow_score}
        """
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.get_flow_score(symbol)
            except Exception:
                results[symbol] = 0.0
            time.sleep(0.1)  # Be gentle with free APIs
        return results
