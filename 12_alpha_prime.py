#!/usr/bin/env python3
"""
ALPHA PRIME v2 — Multi-Factor Regime-Aware Capital Allocator

The closest thing to a Holy Grail in crypto trading.
Not prediction. Not buy-and-hope. Pure capital allocation to the best behavior.

5 LAYERS:
1. Market Regime Detector — Should I even be in risk?
2. Universe Filter — Which coins are worth attention?
3. Cross-Sectional Ranking — Who are today's winners?
4. Risk Engine — How much do I bet on each?
5. Execution Engine — When do I enter, exit, trim, or kill?
ALPHA PRIME v2
Multi-Exchange Momentum Trading Bot with Advanced Risk Management
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, jsonify, render_template_string, request
import threading
import time
import json
import os
import pickle
import warnings
from pathlib import Path
from multi_exchange import MultiExchangeScanner

# ML imports for HMM and Portfolio Optimizer
try:
    from hmmlearn.hmm import GaussianHMM
    from sklearn.preprocessing import StandardScaler
    from scipy.optimize import minimize
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("⚠️  hmmlearn/scipy/sklearn not installed. Run: pip install hmmlearn scipy scikit-learn")

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

STARTING_BALANCE = 100.0
# Dynamic position count by market regime
# Bullish market  → more bets, spread the alpha
# Uncertain/chop  → trim down, preserve capital
# Bearish/risk-off → cash only (handled by early return in slow_loop)
POSITIONS_BY_REGIME = {
    'RISK_ON':  10,   # Bull market  — hold up to 8 coins
    'CHOP':     4,   # Sideways     — hold up to 3 coins
    'RISK_OFF': 1,   # Bear market  — exit all, hold cash
    'UNKNOWN':  5,   # No signal    — conservative, max 2 coins
}
MAX_POSITION_PCT = 0.40       # Max 40% in one position
TARGET_INVESTMENT_PCT = 0.90  # Deploy up to 90% of portfolio; keep 10% as cash buffer
MIN_COIN_VOLUME_24H = 1_000_000  # Lowered to $1M to catch more opportunities
DISCORD_URL = ""              # Optional webhook

# Multi-Exchange Configuration
ENABLED_EXCHANGES = ['binance', 'kucoin', 'gateio', 'mexc', 'bybit', 'kraken', 'bitget']  # All 7 major exchanges

# 3-Speed Architecture
FAST_INTERVAL = 30      # 30 seconds - price observation
MEDIUM_INTERVAL = 300   # 5 minutes - exit checks
SLOW_INTERVAL = 1800    # 30 minutes - full rebalance

# Exit thresholds
RSI_TRIM_THRESHOLD = 80       # Trim when RSI > 80
TRIM_PERCENTAGE = 0.25        # Trim 25% of position
TIME_STOP_CYCLES = 3          # Exit if no movement after N slow cycles
MIN_RETURN_TO_HOLD = 0.02     # 2% minimum return to keep holding

# File paths
PORTFOLIO_FILE = "alpha_prime_portfolio.json"
TRADE_HISTORY_FILE = "alpha_prime_trades.csv"

exchange = ccxt.binance({'enableRateLimit': True})  # Keep for regime detection (BTC/ETH)
scanner = MultiExchangeScanner(ENABLED_EXCHANGES)   # Multi-exchange scanner

# Global state
app = Flask(__name__)
global_state = {
    'regime': 'UNKNOWN',
    'regime_details': {},
    'top_coins': [],
    'positions': {},
    'balance': STARTING_BALANCE,
    'portfolio_value': STARTING_BALANCE,
    'total_pnl': 0.0,
    'last_slow_update': None,
    'last_medium_update': None,
    'last_fast_update': None,
    # HMM Regime Classifier
    'hmm_regime': 'UNKNOWN',
    'hmm_confidence': 0.0,
    'hmm_state_history': [],   # list of {'timestamp', 'state'}
    'hmm_trained': False,
    # Portfolio Optimizer
    'optimizer_weights': {},   # symbol -> weight
    'optimizer_sharpe': 0.0,
    'optimizer_active': False,
}

# =============================================================================
# LAYER 1: MARKET REGIME DETECTOR
# =============================================================================

class RegimeDetector:
    """
    Determines if we should be in risk.
    Uses BTC + ETH as market thermometers.
    """
    
    @staticmethod
    def detect_regime():
        """
        Returns: ('RISK_ON' | 'CHOP' | 'RISK_OFF', details_dict)
        """
        try:
            # Fetch BTC data (4H candles, 200 periods for EMA200)
            btc_candles = exchange.fetch_ohlcv('BTC/USDT', '4h', limit=200)
            df = pd.DataFrame(btc_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            current_price = df['close'].iloc[-1]
            ema_200 = df['close'].ewm(span=200, adjust=False).mean().iloc[-1]
            
            # 30-day volatility (using daily returns approximation)
            returns = df['close'].pct_change().dropna()
            volatility_30d = returns.tail(180).std() * np.sqrt(365)  # Annualized
            
            # Market breadth: % of top coins above EMA 50
            breadth = RegimeDetector._calculate_market_breadth()
            
            details = {
                'btc_price': current_price,
                'btc_ema200': ema_200,
                'btc_vs_ema': ((current_price - ema_200) / ema_200) * 100,
                'volatility_30d': volatility_30d,
                'breadth_pct': breadth
            }
            
            # Regime logic
            if current_price < ema_200:
                return 'RISK_OFF', details
            elif breadth < 0.30:  # Less than 30% above EMA50
                return 'CHOP', details
            else:
                return 'RISK_ON', details
                
        except Exception as e:
            print(f"⚠️  Regime detection error: {e}")
            return 'UNKNOWN', {}
    
    @staticmethod
    def _calculate_market_breadth():
        """Calculate % of top 30 coins above their EMA 50"""
        top_symbols = [
            'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT',
            'AVAX/USDT', 'LINK/USDT', 'DOT/USDT', 'MATIC/USDT', 'SHIB/USDT',
            'LTC/USDT', 'BCH/USDT', 'ATOM/USDT', 'UNI/USDT', 'FIL/USDT',
            'APT/USDT', 'ARB/USDT', 'OP/USDT', 'INJ/USDT', 'SUI/USDT',
            'SEI/USDT', 'TIA/USDT', 'NEAR/USDT', 'RUNE/USDT', 'DOGE/USDT',
            'PEPE/USDT', 'BONK/USDT', 'WIF/USDT', 'ORDI/USDT', 'FTM/USDT'
        ]
        
        above_ema = 0
        total = 0
        
        for symbol in top_symbols:
            try:
                candles = exchange.fetch_ohlcv(symbol, '4h', limit=50)
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                current = df['close'].iloc[-1]
                ema_50 = df['close'].ewm(span=50, adjust=False).mean().iloc[-1]
                
                if current > ema_50:
                    above_ema += 1
                total += 1
                time.sleep(0.05)
            except:
                continue
        
        return above_ema / total if total > 0 else 0.5

# =============================================================================
# HMM MARKET REGIME CLASSIFIER  (Phase 1 Quant Upgrade)
# =============================================================================

class HMMRegimeClassifier:
    """
    Hidden Markov Model for market regime detection.

    Features fed to the model:
      - log_return   : daily log return of BTC
      - realized_vol : 7-day rolling std of log returns

    3 hidden states auto-labeled by mean return:
      highest mean  → BULL
      lowest mean   → BEAR
      middle mean   → SIDEWAYS

    Model is persisted to disk so it survives restarts.
    Falls back to rule-based RegimeDetector if HMM unavailable.
    """

    MODEL_FILE = 'hmm_regime_model.pkl'
    SCALER_FILE = 'hmm_scaler.pkl'

    def __init__(self, n_states=3, train_symbol='BTC/USDT',
                 train_timeframe='1d', train_limit=365):
        self.n_states = n_states
        self.train_symbol = train_symbol
        self.train_timeframe = train_timeframe
        self.train_limit = train_limit
        self.model = None
        self.scaler = None
        self.state_labels = {}   # state_int -> 'BULL'|'BEAR'|'SIDEWAYS'
        self.is_trained = False

    # ------------------------------------------------------------------
    def _fetch_training_data(self):
        """Fetch OHLCV data for training. Returns DataFrame or None."""
        try:
            candles = exchange.fetch_ohlcv(
                self.train_symbol, self.train_timeframe,
                limit=self.train_limit
            )
            df = pd.DataFrame(
                candles,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            print(f"   ⚠️  HMM: could not fetch training data: {e}")
            return None

    def _prepare_features(self, df):
        """Build [log_return, realized_vol] feature matrix."""
        df = df.copy()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['realized_vol'] = df['log_return'].rolling(7).std()
        df = df.dropna()
        return df[['log_return', 'realized_vol']].values, df

    def _label_states(self, model, X):
        """
        Assign human labels to HMM states based on mean log return.
        State with highest mean return → BULL
        State with lowest mean return  → BEAR
        Middle state                   → SIDEWAYS
        """
        state_seq = model.predict(X)
        means = {}
        for s in range(self.n_states):
            mask = state_seq == s
            if mask.sum() > 0:
                means[s] = X[mask, 0].mean()   # mean log return
            else:
                means[s] = 0.0

        sorted_states = sorted(means.keys(), key=lambda s: means[s])
        labels = {}
        labels[sorted_states[0]] = 'BEAR'
        labels[sorted_states[-1]] = 'BULL'
        for s in sorted_states[1:-1]:
            labels[s] = 'SIDEWAYS'
        return labels

    # ------------------------------------------------------------------
    def train(self):
        """Train the GaussianHMM on historical BTC data and persist to disk."""
        if not HMM_AVAILABLE:
            print("   ⚠️  HMM: hmmlearn not available, skipping training")
            return False

        print("   🧠 HMM: Training on historical BTC data...")
        df = self._fetch_training_data()
        if df is None or len(df) < 60:
            print("   ⚠️  HMM: insufficient data for training")
            return False

        X_raw, df_clean = self._prepare_features(df)

        # Scale features
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X_raw)

        # Fit HMM
        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type='full',
            n_iter=200,
            random_state=42
        )
        self.model.fit(X)

        # Label states
        self.state_labels = self._label_states(self.model, X)
        self.is_trained = True

        # Persist
        try:
            with open(self.MODEL_FILE, 'wb') as f:
                pickle.dump({'model': self.model,
                             'scaler': self.scaler,
                             'state_labels': self.state_labels}, f)
            print(f"   ✅ HMM: Trained & saved. States: {self.state_labels}")
        except Exception as e:
            print(f"   ⚠️  HMM: could not save model: {e}")

        return True

    def load_or_train(self):
        """Load model from disk if available, otherwise train fresh."""
        if not HMM_AVAILABLE:
            return

        if os.path.exists(self.MODEL_FILE):
            try:
                with open(self.MODEL_FILE, 'rb') as f:
                    data = pickle.load(f)
                self.model = data['model']
                self.scaler = data['scaler']
                self.state_labels = data['state_labels']
                self.is_trained = True
                print(f"   📂 HMM: Loaded model from disk. States: {self.state_labels}")
                return
            except Exception as e:
                print(f"   ⚠️  HMM: could not load model ({e}), retraining...")

        self.train()

    def retrain(self):
        """Force retrain (called weekly or on demand)."""
        self.train()

    # ------------------------------------------------------------------
    def predict(self, lookback_days=60):
        """
        Predict current market regime.

        Returns:
            regime      : 'BULL' | 'BEAR' | 'SIDEWAYS' | 'UNKNOWN'
            confidence  : float 0-1 (posterior probability of predicted state)
            history     : list of {'timestamp': str, 'state': str} for last lookback_days
        """
        if not self.is_trained or not HMM_AVAILABLE:
            return 'UNKNOWN', 0.0, []

        try:
            df = self._fetch_training_data()
            if df is None or len(df) < 30:
                return 'UNKNOWN', 0.0, []

            # Use recent data for prediction
            df_recent = df.tail(lookback_days + 10)
            X_raw, df_clean = self._prepare_features(df_recent)
            X = self.scaler.transform(X_raw)

            # Predict state sequence
            state_seq = self.model.predict(X)
            posteriors = self.model.predict_proba(X)

            # Current state = last element
            current_state_int = state_seq[-1]
            current_regime = self.state_labels.get(current_state_int, 'UNKNOWN')
            confidence = float(posteriors[-1, current_state_int])

            # Build history
            timestamps = df_clean['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S').tolist()
            history = [
                {'timestamp': timestamps[i],
                 'state': self.state_labels.get(state_seq[i], 'UNKNOWN')}
                for i in range(len(state_seq))
            ]

            return current_regime, confidence, history

        except Exception as e:
            print(f"   ⚠️  HMM predict error: {e}")
            return 'UNKNOWN', 0.0, []


# =============================================================================
# LAYER 2: UNIVERSE FILTER
# =============================================================================

class UniverseFilter:
    """
    Filters out dead/illiquid coins before scoring.
    Now scans ALL enabled exchanges.
    """
    
    @staticmethod
    def get_viable_coins():
        """
        Returns list of coin dicts from all exchanges that pass minimum viability checks.
        Each dict contains: symbol, base_currency, exchange, volume, price
        """
        try:
            # Use multi-exchange scanner
            viable_coins = scanner.get_all_viable_coins(MIN_COIN_VOLUME_24H)
            
            # Filter out stablecoins
            filtered = []
            for coin in viable_coins:
                base = coin['base_currency']
                if base not in ['USDC', 'BUSD', 'DAI', 'TUSD', 'USDT', 'FDUSD']:
                    filtered.append(coin)
            
            return filtered
        except Exception as e:
            print(f"⚠️  Universe filter error: {e}")
            return []

# =============================================================================
# SENTIMENT LOADER  (reads output from reddit_sentiment.py)
# =============================================================================

class SentimentLoader:
    """
    Loads the latest Reddit sentiment scores from reddit_sentiment_latest.json.
    Provides:
      get(symbol)             → float [-1, +1]  (subreddit-level sentiment)
      get_signal(symbol)      → 'BULLISH' | 'BEARISH' | 'NEUTRAL'
      get_verdict(symbol)     → 'SHOOT_UP' | 'SHOOT_DOWN' | 'WATCH' | 'LOW_DATA'
      get_confidence(symbol)  → 'HIGH' | 'MEDIUM' | 'LOW'
      get_mention_count(symbol) → int
      get_verdict_bonus(symbol) → float  (±0.1 bonus applied to alpha score)

    Cached in memory — call refresh() to reload from disk.
    """

    SENTIMENT_FILE = os.path.join('sentiment_data', 'reddit_sentiment_latest.json')

    def __init__(self):
        self._data = {}
        self.refresh()

    def refresh(self):
        """Reload sentiment data from disk."""
        try:
            if os.path.exists(self.SENTIMENT_FILE):
                with open(self.SENTIMENT_FILE, 'r') as f:
                    self._data = json.load(f)
                meta = self._data.get('_meta', {})
                coins_tracked = meta.get('coins_tracked', '?')
                print(f"   📰 Sentiment v{meta.get('version','?')} loaded: "
                      f"{meta.get('total_posts','?')} posts · "
                      f"{coins_tracked} coins tracked · {meta.get('date','?')}")
                # Print holdings correlation if available
                hc = self._data.get('_holdings_correlation', {})
                if hc.get('holdings'):
                    print(f"   🔗 Holdings correlation: {len(hc['holdings'])} positions · "
                          f"Market: {hc.get('market_overall','?')}")
                    for sym, info in hc['holdings'].items():
                        print(f"      {sym:<16} {info['recommendation']}")
            else:
                print("   ⚠️  Sentiment file not found — running without sentiment signal")
                self._data = {}
        except Exception as e:
            print(f"   ⚠️  Sentiment load error: {e}")
            self._data = {}

    def _entry(self, symbol: str) -> dict:
        """Internal: get the raw entry for a symbol."""
        base = symbol.split('/')[0].upper()  # 'SOL/USDT' → 'SOL'
        return self._data.get(base, {})

    def get(self, symbol: str) -> float:
        """
        Get sentiment score for a coin symbol (e.g. 'SOL/USDT' or 'SOL').
        Returns float in [-1.0, +1.0]. 0.0 = neutral / not found.
        """
        return float(self._entry(symbol).get('sentiment', 0.0))

    def get_signal(self, symbol: str) -> str:
        """Returns 'BULLISH', 'BEARISH', or 'NEUTRAL'."""
        return self._entry(symbol).get('signal', 'NEUTRAL')

    def get_verdict(self, symbol: str) -> str:
        """Returns cross-subreddit verdict: 'SHOOT_UP' | 'SHOOT_DOWN' | 'WATCH' | 'LOW_DATA'."""
        return self._entry(symbol).get('verdict', 'LOW_DATA')

    def get_confidence(self, symbol: str) -> str:
        """Returns verdict confidence: 'HIGH' | 'MEDIUM' | 'LOW'."""
        return self._entry(symbol).get('confidence', 'LOW')

    def get_mention_count(self, symbol: str) -> int:
        """Total number of times this coin was mentioned across all subreddits today."""
        return int(self._entry(symbol).get('mention_count', 0))

    def get_verdict_bonus(self, symbol: str) -> float:
        """
        Returns a small bonus/penalty to add on top of the alpha score:
          +0.10 if SHOOT_UP  with HIGH or MEDIUM confidence
          +0.05 if SHOOT_UP  with LOW confidence
          -0.10 if SHOOT_DOWN with HIGH or MEDIUM confidence
          -0.05 if SHOOT_DOWN with LOW confidence
           0.00 otherwise (WATCH / LOW_DATA)
        """
        verdict    = self.get_verdict(symbol)
        confidence = self.get_confidence(symbol)

        if verdict == 'SHOOT_UP':
            return 0.10 if confidence in ('HIGH', 'MEDIUM') else 0.05
        elif verdict == 'SHOOT_DOWN':
            return -0.10 if confidence in ('HIGH', 'MEDIUM') else -0.05
        return 0.0

    def get_holdings_correlation(self) -> dict:
        """Return the raw holdings correlation block from the JSON."""
        return self._data.get('_holdings_correlation', {})


# Global sentiment loader instance (refreshed each slow cycle)
sentiment_loader = SentimentLoader()

# =============================================================================
# LAYER 3: ALPHA SCORE ENGINE (Volatility-Adjusted Momentum + Sentiment)
# =============================================================================

class AlphaScorer:
    """
    Cross-sectional momentum with volatility adjustment + Reddit sentiment blend.
    """

    # Blending weights: how much each signal contributes to the final score
    MOMENTUM_WEIGHT  = 0.80   # 80% price-based momentum
    SENTIMENT_WEIGHT = 0.20   # 20% Reddit crowd sentiment

    @staticmethod
    def calculate_score(coin_data):
        """
        Calculate volatility-adjusted momentum score for a coin,
        blended with Reddit sentiment.

        Formula:
          RawMomentum     = 0.6 * Momentum_7d + 0.4 * Momentum_24h
          RiskAdjMomentum = RawMomentum / Volatility
          MomentumScore   = RiskAdjMomentum * TrendFilter

          SentimentScore  = reddit_sentiment[-1, +1]  (from SentimentLoader)

          FinalScore = 0.80 * MomentumScore + 0.20 * SentimentScore

        If sentiment data is unavailable, SentimentScore = 0 (neutral),
        so the formula degrades gracefully to pure momentum.
        """
        try:
            symbol = coin_data['symbol']
            exchange_name = coin_data['exchange']

            # Fetch 30 days of daily data for momentum + volatility
            candles = scanner.fetch_ohlcv(symbol, exchange_name, '1d', limit=30)
            if len(candles) < 30:
                return None, None

            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            current_price = df['close'].iloc[-1]
            price_7d_ago  = df['close'].iloc[-8]
            price_1d_ago  = df['close'].iloc[-2]

            # Momentum calculations
            momentum_7d  = (current_price - price_7d_ago) / price_7d_ago
            momentum_24h = (current_price - price_1d_ago) / price_1d_ago

            # Volatility (30-day stddev of log returns)
            log_returns = np.log(df['close'] / df['close'].shift(1)).dropna()
            volatility  = log_returns.std()

            if volatility == 0:
                return None, None

            # Trend filter: EMA 50 on 4H
            candles_4h = scanner.fetch_ohlcv(symbol, exchange_name, '4h', limit=50)
            df_4h      = pd.DataFrame(candles_4h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            ema_50     = df_4h['close'].ewm(span=50, adjust=False).mean().iloc[-1]
            trend_filter = 1 if current_price > ema_50 else 0

            # ── Momentum score (price-based) ──────────────────────────────
            raw_score      = 0.6 * momentum_7d + 0.4 * momentum_24h
            risk_adj_score = raw_score / volatility
            momentum_score = risk_adj_score * trend_filter

            # ── Sentiment score (Reddit crowd signal) ─────────────────────
            sentiment_score   = sentiment_loader.get(symbol)          # [-1, +1]
            sentiment_signal  = sentiment_loader.get_signal(symbol)
            verdict           = sentiment_loader.get_verdict(symbol)  # SHOOT_UP / DOWN / WATCH
            verdict_confidence= sentiment_loader.get_confidence(symbol)
            mention_count     = sentiment_loader.get_mention_count(symbol)
            verdict_bonus     = sentiment_loader.get_verdict_bonus(symbol)

            # ── Blended final score ───────────────────────────────────────
            # If trend filter is OFF (price below EMA50), kill the whole score
            # regardless of sentiment — we don't fight a downtrend.
            if trend_filter == 0:
                final_score = 0.0
            else:
                base_score = (
                    AlphaScorer.MOMENTUM_WEIGHT  * momentum_score +
                    AlphaScorer.SENTIMENT_WEIGHT * sentiment_score
                )
                # Apply verdict bonus: SHOOT_UP/DOWN at HIGH/MEDIUM confidence
                # nudges the score without overriding momentum signal.
                final_score = base_score + verdict_bonus

            details = {
                'symbol':              symbol,
                'exchange':            exchange_name,
                'price':               current_price,
                'momentum_7d':         momentum_7d,
                'momentum_24h':        momentum_24h,
                'volatility':          volatility,
                'ema_50':              ema_50,
                'trend_filter':        trend_filter,
                'raw_score':           raw_score,
                'momentum_score':      momentum_score,
                'sentiment_score':     sentiment_score,
                'sentiment_signal':    sentiment_signal,
                'reddit_verdict':      verdict,
                'verdict_confidence':  verdict_confidence,
                'reddit_mentions':     mention_count,
                'verdict_bonus':       verdict_bonus,
                'final_score':         final_score
            }

            return final_score, details

        except Exception as e:
            return None, None
    
    @staticmethod
    def rank_universe(coin_data_list):
        """
        Score all coins and return sorted by final_score.
        Now handles list of coin dicts (with exchange info).
        """
        results = []
        
        for coin_data in coin_data_list:
            score, details = AlphaScorer.calculate_score(coin_data)
            if score is not None:
                results.append(details)
            time.sleep(0.1)  # Rate limiting
        
        # Sort by final_score descending
        results.sort(key=lambda x: x['final_score'], reverse=True)
        return results

# =============================================================================
# LAYER 4: POSITION SIZING (Inverse Volatility)
# =============================================================================

class PositionSizer:
    """
    Allocates capital using inverse volatility weighting.
    """
    
    @staticmethod
    def calculate_weights(top_coins):
        """
        Input: List of coin details dicts with 'volatility' key
        Output: List of dicts with added 'weight' key
        """
        if not top_coins:
            return []
        
        # Inverse volatility
        inv_vols = [1.0 / coin['volatility'] for coin in top_coins]
        total_inv_vol = sum(inv_vols)
        
        # Normalize
        weights = [iv / total_inv_vol for iv in inv_vols]
        
        # Apply max position cap
        weights = [min(w, MAX_POSITION_PCT) for w in weights]
        
        # Renormalize after capping
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        
        # Add to coins
        for i, coin in enumerate(top_coins):
            coin['weight'] = weights[i]
        
        return top_coins

# =============================================================================
# PORTFOLIO OPTIMIZER — Max Sharpe (Mean-Variance)  (Phase 3 Quant Upgrade)
# =============================================================================

class PortfolioOptimizer:
    """
    Mean-Variance Portfolio Optimizer using Sharpe Ratio maximization.

    Algorithm:
      1. Fetch daily returns for each coin (lookback_days)
      2. Compute expected return vector μ and covariance matrix Σ
      3. Minimize  -(w·μ - rf) / sqrt(w·Σ·w)  subject to:
           sum(w) = 1,  0 ≤ w ≤ MAX_POSITION_PCT
      4. Return coins list with 'weight' key added

    Falls back to PositionSizer (inverse volatility) when:
      - hmmlearn/scipy not available
      - Fewer than MIN_HISTORY_DAYS of data available
      - Optimization fails to converge
    """

    MIN_HISTORY_DAYS = 14

    def __init__(self, risk_free_rate=0.0):
        self.risk_free_rate = risk_free_rate

    def _fetch_returns(self, coins, lookback_days=30):
        """
        Fetch daily log returns for each coin.
        Returns a DataFrame (days × coins) or None if insufficient data.
        """
        returns_dict = {}
        for coin in coins:
            symbol = coin['symbol']
            exchange_name = coin.get('exchange', 'binance')
            try:
                candles = scanner.fetch_ohlcv(symbol, exchange_name, '1d',
                                              limit=lookback_days + 5)
                if len(candles) < self.MIN_HISTORY_DAYS:
                    continue
                df = pd.DataFrame(
                    candles,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                log_ret = np.log(df['close'] / df['close'].shift(1)).dropna()
                returns_dict[symbol] = log_ret.values[-lookback_days:]
            except Exception:
                continue
            time.sleep(0.05)

        if len(returns_dict) < 2:
            return None

        # Align lengths
        min_len = min(len(v) for v in returns_dict.values())
        aligned = {k: v[-min_len:] for k, v in returns_dict.items()}
        return pd.DataFrame(aligned)

    def _neg_sharpe(self, weights, mu, cov):
        """Objective: negative Sharpe ratio (we minimize this)."""
        port_return = np.dot(weights, mu)
        port_vol = np.sqrt(np.dot(weights, np.dot(cov, weights)))
        if port_vol < 1e-10:
            return 0.0
        return -(port_return - self.risk_free_rate) / port_vol

    def optimize(self, coins):
        """
        Compute optimal weights for the given list of coins.
        Returns the same coins list with 'weight' key populated.
        Falls back to PositionSizer on any failure.
        """
        if not HMM_AVAILABLE or len(coins) < 2:
            return PositionSizer.calculate_weights(coins)

        try:
            returns_df = self._fetch_returns(coins)
            if returns_df is None or len(returns_df) < self.MIN_HISTORY_DAYS:
                print("   ⚠️  Optimizer: insufficient data, using inverse-vol fallback")
                return PositionSizer.calculate_weights(coins)

            # Only optimize coins we have data for
            available_symbols = list(returns_df.columns)
            opt_coins = [c for c in coins if c['symbol'] in available_symbols]
            fallback_coins = [c for c in coins if c['symbol'] not in available_symbols]

            mu = returns_df[available_symbols].mean().values          # expected returns
            cov = returns_df[available_symbols].cov().values          # covariance matrix

            n = len(available_symbols)
            w0 = np.ones(n) / n   # equal-weight starting point

            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
            bounds = [(0.0, MAX_POSITION_PCT)] * n

            result = minimize(
                self._neg_sharpe,
                w0,
                args=(mu, cov),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-9}
            )

            if not result.success:
                print(f"   ⚠️  Optimizer: did not converge ({result.message}), using inverse-vol")
                return PositionSizer.calculate_weights(coins)

            raw_weights = result.x
            # Renormalize (in case of floating-point drift)
            total_w = raw_weights.sum()
            if total_w > 0:
                raw_weights = raw_weights / total_w

            # Assign weights back to coins
            weight_map = dict(zip(available_symbols, raw_weights))
            for coin in opt_coins:
                coin['weight'] = float(weight_map[coin['symbol']])

            # Fallback coins get equal share of remaining weight
            if fallback_coins:
                remaining = 1.0 - sum(c['weight'] for c in opt_coins)
                per_coin = max(remaining / len(fallback_coins), 0.0)
                for coin in fallback_coins:
                    coin['weight'] = per_coin
                # Renormalize everything
                total = sum(c['weight'] for c in coins)
                if total > 0:
                    for coin in coins:
                        coin['weight'] /= total

            # Compute portfolio Sharpe for logging
            w_arr = np.array([weight_map.get(s, 0) for s in available_symbols])
            port_sharpe = -self._neg_sharpe(w_arr, mu, cov) * np.sqrt(252)

            weight_str = ' | '.join(
                f"{c['symbol'].split('/')[0]}={c['weight']:.1%}" for c in coins
            )
            print(f"   📐 OPTIMIZER: {weight_str} | Sharpe≈{port_sharpe:.2f}")
            event_logger.log('info',
                f'Portfolio Optimizer: {weight_str} | Annualized Sharpe≈{port_sharpe:.2f}',
                {'weights': {c['symbol']: round(c['weight'], 4) for c in coins},
                 'sharpe': round(port_sharpe, 4)}
            )

            return coins

        except Exception as e:
            print(f"   ⚠️  Optimizer error: {e}, using inverse-vol fallback")
            return PositionSizer.calculate_weights(coins)


# =============================================================================
# ANALYTICS: Portfolio & Execution Monitoring
# =============================================================================

class PortfolioAnalytics:
    """
    Calculate institutional-grade portfolio metrics.
    """
    
    def __init__(self, portfolio):
        self.portfolio = portfolio
        self.equity_history = []  # [(timestamp, portfolio_value, btc_price)]
        self.load_equity_curve()
    
    def load_equity_curve(self):
        """Load historical equity curve from CSV"""
        try:
            if os.path.exists('equity_curve.csv'):
                with open('equity_curve.csv', 'r') as f:
                    for line in f.readlines()[1:]:  # Skip header
                        parts = line.strip().split(',')
                        if len(parts) == 3:
                            self.equity_history.append({
                                'timestamp': parts[0],
                                'portfolio_value': float(parts[1]),
                                'btc_benchmark': float(parts[2])
                            })
        except Exception as e:
            print(f"   ⚠️  Could not load equity curve: {e}")
    
    def save_equity_point(self):
        """Save current portfolio value to equity curve"""
        try:
            portfolio_value = self.portfolio.get_total_value()
            btc_price = scanner.get_current_price('BTC/USDT', 'binance')
            
            # Normalize to starting balance (100)
            btc_normalized = 100 * (btc_price / self.get_initial_btc_price())
            
            timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
            
            # Append to CSV
            with open('equity_curve.csv', 'a') as f:
                f.write(f"{timestamp},{portfolio_value},{btc_normalized}\n")
            
            self.equity_history.append({
                'timestamp': timestamp,
                'portfolio_value': portfolio_value,
                'btc_benchmark': btc_normalized
            })
        except Exception as e:
            print(f"   ⚠️  Could not save equity point: {e}")
    
    def get_initial_btc_price(self):
        """Get BTC price at start (or use first saved price)"""
        if self.equity_history:
            return self.equity_history[0]['btc_benchmark'] / 100 * scanner.get_current_price('BTC/USDT', 'binance')
        return scanner.get_current_price('BTC/USDT', 'binance')
    
    def calculate_nlv(self):
        """Net Liquidation Value"""
        return self.portfolio.get_total_value()
    
    def calculate_daily_pnl(self):
        """Daily P&L in $ and %"""
        if len(self.equity_history) < 2:
            return {'pnl_dollar': 0, 'pnl_percent': 0}
        
        # Find value from 24h ago
        now = datetime.now()
        day_ago = now - timedelta(days=1)
        
        closest = None
        for point in self.equity_history:
            point_time = datetime.fromisoformat(point['timestamp'])
            if point_time >= day_ago:
                closest = point
                break
        
        if not closest:
            closest = self.equity_history[0]
        
        current_value = self.calculate_nlv()
        past_value = closest['portfolio_value']
        
        pnl_dollar = current_value - past_value
        pnl_percent = (pnl_dollar / past_value) * 100 if past_value > 0 else 0
        
        return {'pnl_dollar': pnl_dollar, 'pnl_percent': pnl_percent}
    
    def calculate_portfolio_beta(self):
        """Portfolio correlation to BTC (30-day rolling)"""
        if len(self.equity_history) < 30:
            return 0
        
        # Get last 30 data points
        recent = self.equity_history[-30:]
        
        portfolio_returns = []
        btc_returns = []
        
        for i in range(1, len(recent)):
            port_ret = (recent[i]['portfolio_value'] - recent[i-1]['portfolio_value']) / recent[i-1]['portfolio_value']
            btc_ret = (recent[i]['btc_benchmark'] - recent[i-1]['btc_benchmark']) / recent[i-1]['btc_benchmark']
            
            portfolio_returns.append(port_ret)
            btc_returns.append(btc_ret)
        
        if not portfolio_returns:
            return 0
        
        # Calculate covariance and variance
        port_mean = np.mean(portfolio_returns)
        btc_mean = np.mean(btc_returns)
        
        covariance = np.mean([(p - port_mean) * (b - btc_mean) for p, b in zip(portfolio_returns, btc_returns)])
        btc_variance = np.var(btc_returns)
        
        beta = covariance / btc_variance if btc_variance > 0 else 0
        return beta
    
    def calculate_sharpe(self):
        """Sharpe Ratio: Risk-adjusted return"""
        if len(self.equity_history) < 2:
            return 0
        
        # Use available data
        points = self.equity_history[-288:] if len(self.equity_history) > 288 else self.equity_history
        returns = []
        
        for i in range(1, len(points)):
            ret = (points[i]['portfolio_value'] - points[i-1]['portfolio_value']) / points[i-1]['portfolio_value']
            returns.append(ret)
        
        if not returns:
            return 0
        
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Assume risk-free rate = 0 for crypto
        sharpe = (avg_return / std_return) if std_return > 0 else 0
        
        # Annualize (assuming 5-min intervals → 288 per day)
        sharpe_annualized = sharpe * np.sqrt(288 * 365)
        
        return sharpe_annualized
    
    def calculate_var_95(self):
        """Value at Risk (95% confidence, 24h)"""
        if len(self.equity_history) < 2:
            return 0
        
        # Use available data, up to last 288 points (24h)
        recent = self.equity_history[-288:] if len(self.equity_history) > 288 else self.equity_history
        returns = []
        
        for i in range(1, len(points)):
            ret = (points[i]['portfolio_value'] - points[i-1]['portfolio_value']) / points[i-1]['portfolio_value']
            returns.append(ret)
        
        if not returns:
            return 0
        
        # 5th percentile (95% VaR)
        var_percentile = np.percentile(returns, 5)
        current_value = self.calculate_nlv()
        
        var_dollar = current_value * abs(var_percentile)
        
        return var_dollar
    
    def calculate_max_drawdown(self):
        """Maximum drawdown from peak"""
        if len(self.equity_history) < 2:
            return 0
        
        values = [point['portfolio_value'] for point in self.equity_history]
        peak = values[0]
        max_dd = 0
        
        for value in values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        
        return max_dd * 100  # Return as percentage
    
    def calculate_sortino(self):
        """Calculate Sortino Ratio (downside risk only)"""
        if len(self.equity_history) < 2:
            return 0
            
        points = self.equity_history[-288:] if len(self.equity_history) > 288 else self.equity_history
        returns = []
        for i in range(1, len(points)):
            ret = (points[i]['portfolio_value'] - points[i-1]['portfolio_value']) / points[i-1]['portfolio_value']
            returns.append(ret)
            
        if not returns:
            return 0
            
        mean_ret = np.mean(returns)
        target_return = 0
        downside_returns = [r - target_return for r in returns if r < target_return]
        
        if not downside_returns:
            return float('inf')  # No downside risk
            
        downside_deviation = np.std(downside_returns)
        if downside_deviation == 0:
            return float('inf')
            
        return (mean_ret * 288) / (downside_deviation * np.sqrt(288))

    def get_exposure_breakdown(self):
        """Long/Short/Net exposure"""
        total_long = sum(pos['current_value'] for pos in self.portfolio.positions.values())
        total_short = 0  # Paper trading = long only for now
        net_exposure = total_long - total_short
        
        nlv = self.calculate_nlv()
        
        return {
            'long_exposure': total_long,
            'short_exposure': total_short,
            'net_exposure': net_exposure,
            'long_pct': (total_long / nlv * 100) if nlv > 0 else 0,
            'cash_pct': (self.portfolio.balance / nlv * 100) if nlv > 0 else 0
        }


class ExecutionMonitor:
    """
    Track execution quality: slippage, fill rate, latency.
    """
    
    def __init__(self):
        self.orders = []  # List of order dicts
        self.api_latencies = []  # List of (timestamp, latency_ms)
    
    def log_order(self, symbol, side, expected_price, filled_price, quantity, exchange):
        """Log an executed order"""
        slippage = ((filled_price - expected_price) / expected_price) * 100 if expected_price > 0 else 0
        
        order = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'side': side,
            'expected_price': expected_price,
            'filled_price': filled_price,
            'quantity': quantity,
            'exchange': exchange,
            'slippage_pct': slippage
        }
        
        self.orders.append(order)
    
    def log_api_call(self, latency_ms):
        """Log API round-trip time"""
        self.api_latencies.append({
            'timestamp': datetime.now().isoformat(),
            'latency_ms': latency_ms
        })
        
        # Keep last 100 only
        if len(self.api_latencies) > 100:
            self.api_latencies = self.api_latencies[-100:]
    
    def get_fill_rate(self):
        """Percentage of orders filled (vs cancelled/rejected)"""
        if not self.orders:
            return 100.0
        
        # For now, all paper trades are filled
        return 100.0
    
    def get_avg_slippage(self):
        """Average slippage across all orders"""
        if not self.orders:
            return 0
        
        recent_orders = self.orders[-50:]  # Last 50 orders
        avg_slippage = np.mean([o['slippage_pct'] for o in recent_orders])
        
        return avg_slippage
    
    def get_avg_latency(self):
        """Average API latency in ms"""
        if not self.api_latencies:
            return 0
        
        recent = self.api_latencies[-20:]  # Last 20 calls
        avg_latency = np.mean([l['latency_ms'] for l in recent])
        
        return avg_latency

# =============================================================================
# EVENT LOGGER (Server-Side Log Stream)
# =============================================================================

class EventLogger:
    """Thread-safe event logger for streaming server-side events to dashboard."""
    
    def __init__(self, log_file='alpha_prime_events.jsonl', max_events=500):
        self.events = []  # List of {timestamp, type, message, details}
        self.max_events = max_events
        self.log_file = log_file
        self._lock = threading.Lock()
        self._event_id = 0
        self._load_from_file()
    
    def _load_from_file(self):
        """Load recent events from log file on startup"""
        if not os.path.exists(self.log_file):
            return
            
        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
                # Load last N lines to avoid reading massive files
                start_idx = max(0, len(lines) - self.max_events)
                
                for line in lines[start_idx:]:
                    try:
                        event = json.loads(line)
                        self.events.append(event)
                        # Keep ID counter in sync
                        self._event_id = max(self._event_id, event.get('id', 0))
                    except:
                        continue
                        
            print(f"📂 Loaded {len(self.events)} events from {self.log_file}")
            
        except Exception as e:
            print(f"⚠️  Could not load events: {e}")

    def log(self, event_type, message, details=None):
        """Log an event. Types: info, buy, sell, trim, regime, signal, warn, error"""
        with self._lock:
            self._event_id += 1
            event = {
                'id': self._event_id,
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'type': event_type,
                'message': message,
                'details': details or {}
            }
            
            # Memory
            self.events.append(event)
            if len(self.events) > self.max_events:
                self.events = self.events[-self.max_events:]
            
            # File Persistence
            try:
                with open(self.log_file, 'a') as f:
                    f.write(json.dumps(event) + "\n")
            except Exception as e:
                print(f"⚠️  Could not write event to file: {e}")
    
    def get_events(self, since_id=0, limit=100):
        """Get events after a given ID (for incremental polling)."""
        with self._lock:
            if since_id == 0:
                return self.events[-limit:]
            return [e for e in self.events if e['id'] > since_id][-limit:]
    
    def get_latest_id(self):
        with self._lock:
            return self._event_id

# Global event logger instance
event_logger = EventLogger()

# =============================================================================
# LAYER 5: EXECUTION ENGINE (Entry, Exit, Trim)
# =============================================================================

class ExecutionEngine:
    """
    Handles all trading logic: entry, exit, trim.
    """
    
    def __init__(self, portfolio):
        self.portfolio = portfolio
    
    def check_exit_conditions(self, symbol, position):
        """
        Check if position should be exited.
        Returns: (should_exit, reason)
        
        Exit conditions:
        - Price < EMA 50 (4H close) → KILL
        - Time stop: held for N cycles with low return
        """
        try:
            # Fetch 4H data for EMA50
            candles_4h = exchange.fetch_ohlcv(symbol, '4h', limit=50)
            df = pd.DataFrame(candles_4h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            current_price = df['close'].iloc[-1]
            ema_50 = df['close'].ewm(span=50, adjust=False).mean().iloc[-1]
            
            # Exit A: Trend Kill
            if current_price < ema_50:
                return True, "Trend Kill (< EMA50)"
            
            # Exit C: Time Stop
            cycles_held = position.get('cycles_held', 0)
            entry_price = position['entry_price']
            pnl_pct = (current_price - entry_price) / entry_price
            
            if cycles_held >= TIME_STOP_CYCLES and pnl_pct < MIN_RETURN_TO_HOLD:
                return True, f"Time Stop ({cycles_held} cycles, {pnl_pct*100:.1f}%)"
            
            return False, ""
            
        except Exception as e:
            return False, ""
    
    def check_trim_conditions(self, symbol):
        """
        Check if position should be trimmed.
        Returns: (should_trim, reason)
        
        Trim conditions:
        - RSI > 80
        - Price > Upper Bollinger Band
        """
        try:
            candles = exchange.fetch_ohlcv(symbol, '4h', limit=50)
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = (100 - (100 / (1 + rs))).iloc[-1]
            
            if rsi > RSI_TRIM_THRESHOLD:
                return True, f"Overbought (RSI {rsi:.0f})"
            
            # Bollinger Bands
            sma_20 = df['close'].rolling(window=20).mean().iloc[-1]
            std_20 = df['close'].rolling(window=20).std().iloc[-1]
            upper_band = sma_20 + (2 * std_20)
            current_price = df['close'].iloc[-1]
            
            if current_price > upper_band:
                return True, "Above Bollinger Upper Band"
            
            return False, ""
            
        except Exception as e:
            return False, ""

# =============================================================================
# PAPER TRADING PORTFOLIO
# =============================================================================

class PaperPortfolio:
    """
    Manages paper trading portfolio with persistence.
    """
    
    def __init__(self):
        self.load_portfolio()
    
    def load_portfolio(self):
        """Load or initialize portfolio"""
        if Path(PORTFOLIO_FILE).exists():
            with open(PORTFOLIO_FILE, 'r') as f:
                data = json.load(f)
                self.balance = data.get('balance', STARTING_BALANCE)
                self.positions = data.get('positions', {})
                self.total_trades = data.get('total_trades', 0)
                self.winning_trades = data.get('winning_trades', 0)
                # Fallback for old portfolio files
                self.realized_pnl = data.get('realized_pnl', data.get('total_pnl', 0.0))
                print(f"📂 Loaded portfolio: ${self.balance:.2f} balance, {len(self.positions)} positions")
        else:
            self.balance = STARTING_BALANCE
            self.positions = {}
            self.total_trades = 0
            self.winning_trades = 0
            self.realized_pnl = 0.0
            print(f"🆕 Starting fresh with ${STARTING_BALANCE}")
            self.save_portfolio()
    
    def save_portfolio(self):
        """Save portfolio to file"""
        data = {
            'balance': self.balance,
            'positions': self.positions,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'realized_pnl': self.realized_pnl,
            'last_updated': datetime.now().isoformat()
        }
        with open(PORTFOLIO_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    
    def log_trade(self, action, symbol, quantity, price, pnl=0, reason=""):
        """Log trade to CSV"""
        trade = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'value': quantity * price,
            'pnl': pnl,
            'reason': reason,
            'balance_after': self.balance
        }
        
        df = pd.DataFrame([trade])
        if Path(TRADE_HISTORY_FILE).exists():
            df.to_csv(TRADE_HISTORY_FILE, mode='a', header=False, index=False)
        else:
            df.to_csv(TRADE_HISTORY_FILE, index=False)
    
    def get_current_price(self, symbol, exchange_name='binance'):
        """Get current price for symbol from specific exchange"""
        try:
            price = scanner.get_current_price(symbol, exchange_name)
            return price
        except:
            return None
    
    def get_portfolio_value(self):
        """Calculate total portfolio value and update position stats"""
        total = self.balance
        for symbol, pos in self.positions.items():
            exchange_name = pos.get('exchange', 'binance')  # Get exchange from position
            price = self.get_current_price(symbol, exchange_name)
            if price:
                current_value = pos['quantity'] * price
                entry_value = pos['quantity'] * pos['entry_price']
                unrealized_pnl = current_value - entry_value
                unrealized_pnl_pct = (unrealized_pnl / entry_value) * 100 if entry_value > 0 else 0
                
                # Update position stats for dashboard
                pos['current_price'] = price
                pos['current_value'] = current_value
                pos['unrealized_pnl'] = unrealized_pnl
                pos['unrealized_pnl_pct'] = unrealized_pnl_pct
                
                total += current_value
        return total
    
    def get_total_value(self):
        """Alias for get_portfolio_value (used by analytics)"""
        return self.get_portfolio_value()
    
    def execute_buy(self, symbol, weight, details):
        """Execute buy order (exchange-aware)"""
        if symbol in self.positions:
            return False

        exchange_name = details.get('exchange', 'binance')
        price = self.get_current_price(symbol, exchange_name)
        if not price:
            return False

        # Calculate position size:
        # Each coin gets (weight × TARGET_INVESTMENT_PCT) of total portfolio value.
        # This guarantees we deploy TARGET_INVESTMENT_PCT in total, not whatever
        # the optimizer happened to sum to after renormalization + old multipliers.
        portfolio_value = self.get_portfolio_value()
        position_value  = portfolio_value * weight * TARGET_INVESTMENT_PCT

        # Safety: never spend more than we actually have
        if self.balance < position_value:
            position_value = self.balance * 0.95

        if position_value < 5:
            return False
        
        quantity = position_value / price
        
        self.balance -= position_value
        self.positions[symbol] = {
            'entry_price': price,
            'quantity': quantity,
            'entry_time': datetime.now().isoformat(),
            'entry_score': details['final_score'],
            'weight': weight,
            'cycles_held': 0,
            'exchange': exchange_name  # Track which exchange
        }
        
        self.log_trade('BUY', symbol, quantity, price, reason=f"Score: {details['final_score']:.2f} | {exchange_name.upper()}")
        self.save_portfolio()
        print(f"✅ BUY {symbol} on {exchange_name.upper()}: {quantity:.4f} @ ${price:.4f} (${position_value:.2f})")
        event_logger.log('buy', f'BUY {symbol} on {exchange_name.upper()}: {quantity:.4f} @ ${price:.4f} (${position_value:.2f})', {
            'symbol': symbol, 'exchange': exchange_name, 'price': price, 'quantity': quantity, 'value': position_value, 'score': details.get('final_score', 0)
        })
        return True
    
    def execute_sell(self, symbol, reason):
        """Execute sell order (exchange-aware)"""
        if symbol not in self.positions:
            return False
        
        position = self.positions[symbol]
        exchange_name = position.get('exchange', 'binance')  # Get exchange from position
        price = self.get_current_price(symbol, exchange_name)
        if not price:
            return False
        
        entry_price = position['entry_price']
        quantity = position['quantity']
        entry_value = quantity * entry_price
        exit_value = quantity * price
        pnl = exit_value - entry_value
        pnl_pct = (pnl / entry_value) * 100
        
        self.balance += exit_value
        del self.positions[symbol]
        
        # Update stats
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1
        self.realized_pnl += pnl  # Update realized P&L
        
        self.log_trade('SELL', symbol, quantity, price, pnl=pnl, reason=f"{reason} | {exchange_name.upper()}")
        self.save_portfolio()
        
        emoji = "🟢" if pnl > 0 else "🔴"
        print(f"{emoji} SELL {symbol} on {exchange_name.upper()}: {quantity:.4f} @ ${price:.4f} | P&L: ${pnl:.2f} ({pnl_pct:+.1f}%) | {reason}")
        event_logger.log('sell', f'SELL {symbol} on {exchange_name.upper()}: {quantity:.4f} @ ${price:.4f} | P&L: ${pnl:.2f} ({pnl_pct:+.1f}%) | {reason}', {
            'symbol': symbol, 'exchange': exchange_name, 'price': price, 'quantity': quantity, 'pnl': pnl, 'pnl_pct': pnl_pct, 'reason': reason
        })
        return True
    
    def execute_trim(self, symbol, reason):
        """Trim position by TRIM_PERCENTAGE"""
        if symbol not in self.positions:
            return False
        
        position = self.positions[symbol]
        price = self.get_current_price(symbol)
        if not price:
            return False
        
        trim_qty = position['quantity'] * TRIM_PERCENTAGE
        position['quantity'] -= trim_qty
        
        trim_value = trim_qty * price
        self.balance += trim_value
        
        entry_price = position['entry_price']
        pnl = (price - entry_price) * trim_qty
        pnl_pct = (price - entry_price) / entry_price * 100
        
        self.realized_pnl += pnl
        if pnl > 0:
            self.winning_trades += 0.25  # Partial win
        
        self.log_trade('TRIM', symbol, trim_qty, price, pnl=pnl, reason=reason)
        self.save_portfolio()
        
        print(f"📉 TRIM {symbol}: {TRIM_PERCENTAGE*100:.0f}% @ ${price:.4f} | Locked {pnl_pct:+.1f}% | {reason}")
        event_logger.log('trim', f'TRIM {symbol}: {TRIM_PERCENTAGE*100:.0f}% @ ${price:.4f} | Locked {pnl_pct:+.1f}% | {reason}', {
            'symbol': symbol, 'price': price, 'quantity': trim_qty, 'pnl': pnl, 'reason': reason
        })
        return True

# =============================================================================
# MAIN ALPHA PRIME ENGINE
# =============================================================================

class AlphaPrime:
    """
    Main engine orchestrating all 5 layers with 3-speed architecture.
    """
    
    def __init__(self):
        self.portfolio = PaperPortfolio()
        self.executor = ExecutionEngine(self.portfolio)
        
        self.last_slow_check = None
        self.last_medium_check = None
        self.last_fast_check = None
        
        self.current_regime = 'UNKNOWN'
        self.regime_details = {}
        self.top_ranked = []
        
        # Initialize analytics and monitoring
        self.analytics = PortfolioAnalytics(self.portfolio)
        self.execution_monitor = ExecutionMonitor()
        
        # ── NEW: HMM Regime Classifier ──────────────────────────────────
        self.hmm_classifier = HMMRegimeClassifier()
        print("\n🧠 Initializing HMM Regime Classifier...")
        self.hmm_classifier.load_or_train()
        global_state['hmm_trained'] = self.hmm_classifier.is_trained
        
        # ── NEW: Portfolio Optimizer ─────────────────────────────────────
        self.portfolio_optimizer = PortfolioOptimizer(risk_free_rate=0.0)
        print("📐 Portfolio Optimizer (Max-Sharpe) ready")
        
        # Store in global state for API access
        global_state['portfolio_obj'] = self.portfolio
        global_state['analytics'] = self.analytics
        global_state['execution_monitor'] = self.execution_monitor
        
        # Initialize equity curve CSV if doesn't exist
        if not os.path.exists('equity_curve.csv'):
            with open('equity_curve.csv', 'w') as f:
                f.write("timestamp,portfolio_value,btc_benchmark\n")
    
    def fast_loop(self):
        """
        30-second loop: Observe prices, update volatility, check emergencies.
        """
        print(f"\n⚡ FAST CHECK - {datetime.now().strftime('%H:%M:%S')}")
        
        # Update global state
        global_state['balance'] = self.portfolio.balance
        global_state['positions'] = self.portfolio.positions
        global_state['portfolio_value'] = self.portfolio.get_portfolio_value()
        global_state['realized_pnl'] = self.portfolio.realized_pnl
        global_state['last_fast_update'] = datetime.now().isoformat()
        
        # Emergency drawdown checks could go here
        # For now, just observe
        print(f"   Portfolio: ${global_state['portfolio_value']:.2f} | Cash: ${self.portfolio.balance:.2f}")
        event_logger.log('info', f'Portfolio: ${global_state["portfolio_value"]:.2f} | Cash: ${self.portfolio.balance:.2f} | {len(self.portfolio.positions)} positions')
    
    def medium_loop(self):
        """
        5-minute loop: Check trend breaks, stop-losses, trims.
        """
        print(f"\n⚙️  MEDIUM CHECK - {datetime.now().strftime('%H:%M:%S')}")
        event_logger.log('info', 'Medium cycle: checking exits and trims')
        
        # Check exit conditions for all positions
        for symbol in list(self.portfolio.positions.keys()):
            position = self.portfolio.positions[symbol]
            
            # Exit checks
            should_exit, exit_reason = self.executor.check_exit_conditions(symbol, position)
            if should_exit:
                self.portfolio.execute_sell(symbol, exit_reason)
                continue
            
            # Trim checks (only if not exiting)
            should_trim, trim_reason = self.executor.check_trim_conditions(symbol)
            if should_trim:
                self.portfolio.execute_trim(symbol, trim_reason)
        
        global_state['last_medium_update'] = datetime.now().isoformat()
    
    def slow_loop(self):
        """
        1-hour loop: Recalculate scores, rank universe, rotate capital.
        """
        print(f"\n{'='*80}")
        print(f"🧠 ALPHA PRIME SLOW CYCLE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")

        # Refresh Reddit sentiment data from disk (updated daily by reddit_sentiment.py)
        print("\n📰 REFRESHING SENTIMENT DATA")
        sentiment_loader.refresh()

        # LAYER 1: Regime Detection (Rule-Based + HMM)
        print("\n📊 LAYER 1: MARKET REGIME DETECTION")
        
        # --- Rule-Based Detector (existing, fast) ---
        regime, details = RegimeDetector.detect_regime()
        self.current_regime = regime
        self.regime_details = details
        
        print(f"   [Rule] Regime: {regime}")
        print(f"   BTC: ${details.get('btc_price', 0):.2f} vs EMA200 ${details.get('btc_ema200', 0):.2f} ({details.get('btc_vs_ema', 0):+.1f}%)")
        print(f"   Volatility: {details.get('volatility_30d', 0):.1%}")
        print(f"   Breadth: {details.get('breadth_pct', 0):.1%} above EMA50")
        
        # --- HMM Classifier (ML, primary signal) ---
        hmm_regime, hmm_confidence, hmm_history = self.hmm_classifier.predict()
        
        # Map HMM labels to execution logic
        # BULL → RISK_ON, BEAR → RISK_OFF, SIDEWAYS → CHOP
        hmm_to_rule = {'BULL': 'RISK_ON', 'BEAR': 'RISK_OFF', 'SIDEWAYS': 'CHOP', 'UNKNOWN': regime}
        hmm_rule_equiv = hmm_to_rule.get(hmm_regime, regime)
        
        if hmm_regime != 'UNKNOWN':
            print(f"   [HMM]  Regime: {hmm_regime} (confidence: {hmm_confidence:.1%})")
            # HMM is primary; rule-based is fallback
            effective_regime = hmm_rule_equiv
        else:
            print(f"   [HMM]  Not available — using rule-based regime")
            effective_regime = regime
        
        # Update global state with both signals
        global_state['regime'] = effective_regime
        global_state['regime_details'] = details
        global_state['hmm_regime'] = hmm_regime
        global_state['hmm_confidence'] = hmm_confidence
        global_state['hmm_state_history'] = hmm_history[-30:] if hmm_history else []
        
        event_logger.log('regime',
            f'Regime: {effective_regime} | HMM: {hmm_regime} ({hmm_confidence:.0%} conf) | Rule: {regime} | BTC: ${details.get("btc_price", 0):.0f}',
            {'regime': effective_regime, 'hmm_regime': hmm_regime, 'hmm_confidence': hmm_confidence,
             'rule_regime': regime, 'btc_price': details.get('btc_price', 0)}
        )
        
        # Strategy switching based on effective regime
        if effective_regime == 'RISK_OFF':
            print("   ⚠️  RISK-OFF / BEAR DETECTED → EXITING ALL POSITIONS (holding cash)")
            event_logger.log('warn', f'BEAR/RISK-OFF regime — exiting all positions')
            for symbol in list(self.portfolio.positions.keys()):
                self.portfolio.execute_sell(symbol, f"Bear Regime ({hmm_regime})")
            return
        
        # LAYER 2: Universe Filter
        print("\n🔍 LAYER 2: UNIVERSE FILTER")
        viable_coins = UniverseFilter.get_viable_coins()
        print(f"   Found {len(viable_coins)} viable coins (>${MIN_COIN_VOLUME_24H/1_000_000:.0f}M volume)")
        
        # LAYER 3: Alpha Score & Ranking
        print("\n📈 LAYER 3: ALPHA SCORE ENGINE")
        print(f"   Calculating volatility-adjusted momentum scores...")
        ranked = AlphaScorer.rank_universe(viable_coins)
        
        if not ranked:
            print("   ⚠️  No coins to rank")
            return
        
        # ── Dynamic position count based on regime ─────────────────────────
        active_n = POSITIONS_BY_REGIME.get(effective_regime, 2)
        top_n = ranked[:active_n]
        self.top_ranked = top_n

        print(f"\n   🏆 TOP {active_n} RANKED COINS (regime: {effective_regime}):")
        print(f"   {'Coin':<12} {'Score':<10} {'Mom7d':<8} {'Mom24h':<8} {'Vol':<8} {'Trend'}")
        print(f"   {'-'*70}")
        for coin in top_n:
            trend_status = "✅" if coin['trend_filter'] == 1 else "❌"
            print(f"   {coin['symbol']:<12} {coin['final_score']:>9.4f} {coin['momentum_7d']:>7.1%} {coin['momentum_24h']:>7.1%} {coin['volatility']:>7.4f} {trend_status}")

        global_state['top_coins'] = top_n

        # Log top coins as signal events
        for coin in top_n:
            event_logger.log('signal', f'RANKED #{top_n.index(coin)+1}: {coin["symbol"]} | Score: {coin["final_score"]:.4f} | Mom7d: {coin["momentum_7d"]:.1%} | {coin.get("exchange","binance").upper()}')

        # LAYER 4: Position Sizing (Portfolio Optimizer → Inverse-Vol fallback)
        print("\n💰 LAYER 4: POSITION SIZING (Max-Sharpe Optimizer)")

        # No additional regime slicing needed — active_n already accounts for regime
        top_n_weighted = self.portfolio_optimizer.optimize(top_n)
        
        for coin in top_n_weighted:
            print(f"   {coin['symbol']:<12} → {coin['weight']:>6.1%}")
        
        # Update optimizer state in global
        global_state['optimizer_weights'] = {c['symbol']: round(c['weight'], 4) for c in top_n_weighted}
        global_state['optimizer_active'] = HMM_AVAILABLE
        
        # Position count already controls regime risk — no need to also cut weights.
        # (Regime multiplier removed: it was causing ~70% of capital to idle in cash)
        
        # LAYER 5: Execution - Rotation
        print("\n🔄 LAYER 5: CAPITAL ROTATION")
        
        # Increment cycles_held for existing positions
        for symbol in self.portfolio.positions:
            self.portfolio.positions[symbol]['cycles_held'] = self.portfolio.positions[symbol].get('cycles_held', 0) + 1
        
        # Exit positions not in Top N
        top_symbols = {coin['symbol'] for coin in top_n_weighted}
        for symbol in list(self.portfolio.positions.keys()):
            if symbol not in top_symbols:
                self.portfolio.execute_sell(symbol, "Rotated out of Top N")
        
        # Enter new positions in Top N
        for coin in top_n_weighted:
            if coin['symbol'] not in self.portfolio.positions:
                self.portfolio.execute_buy(coin['symbol'], coin['weight'], coin)
        
        # Save equity curve point
        self.analytics.save_equity_point()
        
        global_state['last_slow_update'] = datetime.now().isoformat()
        
        # Summary
        portfolio_value = self.portfolio.get_portfolio_value()
        total_return = portfolio_value - STARTING_BALANCE
        total_return_pct = (total_return / STARTING_BALANCE) * 100
        win_rate = (self.portfolio.winning_trades / self.portfolio.total_trades * 100) if self.portfolio.total_trades > 0 else 0
        
        print(f"\n{'='*80}")
        print(f"📊 PORTFOLIO SUMMARY")
        print(f"   Value: ${portfolio_value:.2f} | Return: {total_return_pct:+.1f}%")
        print(f"   Cash: ${self.portfolio.balance:.2f} | Positions: {len(self.portfolio.positions)}")
        print(f"   Trades: {self.portfolio.total_trades} | Win Rate: {win_rate:.0f}%")
        print(f"{'='*80}\n")
        
        event_logger.log('info', f'Slow cycle complete — NLV: ${portfolio_value:.2f} ({total_return_pct:+.1f}%) | {len(self.portfolio.positions)} positions | Win rate: {win_rate:.0f}%')
    
    def run(self):
        """
        Main execution loop with 3-speed architecture.
        """
        # 3-Speed Architecture
        FAST_INTERVAL = 30      # 30 seconds - price observation
        MEDIUM_INTERVAL = 300   # 5 minutes - exit checks
        SLOW_INTERVAL = 1800    # 30 minutes - full rebalance
        print("="*80)
        print("🚀 ALPHA PRIME v2 — STARTING")
        print("="*80)
        print("📊 5 Layers: Regime → Filter → Score → Size → Execute")
        print("⏱️  3 Speeds: 30s (observe) / 5min (exits) / 1h (rotate)")
        print("="*80)
        
        # Initial slow cycle
        self.slow_loop()
        self.last_slow_check = time.time()
        self.last_medium_check = time.time()
        self.last_fast_check = time.time()
        
        try:
            while True:
                current_time = time.time()
                
                # Check if it's time for slow cycle
                if current_time - self.last_slow_check >= SLOW_INTERVAL:
                    self.slow_loop()
                    self.last_slow_check = current_time
                
                # Check if it's time for medium cycle
                elif current_time - self.last_medium_check >= MEDIUM_INTERVAL:
                    self.medium_loop()
                    self.last_medium_check = current_time
                
                # Check if it's time for fast cycle
                elif current_time - self.last_fast_check >= FAST_INTERVAL:
                    self.fast_loop()
                    self.last_fast_check = current_time
                
                time.sleep(5)  # Check every 5 seconds
                
        except KeyboardInterrupt:
            print("\n🛑 Alpha Prime stopped.")
            portfolio_value = self.portfolio.get_portfolio_value()
            print(f"   Final Portfolio Value: ${portfolio_value:.2f}")

# =============================================================================
# EMBEDDED FLASK DASHBOARD
# =============================================================================

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Alpha Prime v2</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        :root {
            --background: 0 0% 3.9%;
            --foreground: 0 0% 98%;
            --card: 0 0% 6.5%;
            --card-foreground: 0 0% 98%;
            --popover: 0 0% 3.9%;
            --popover-foreground: 0 0% 98%;
            --primary: 0 0% 98%;
            --primary-foreground: 0 0% 9%;
            --secondary: 0 0% 14.9%;
            --secondary-foreground: 0 0% 98%;
            --muted: 0 0% 14.9%;
            --muted-foreground: 0 0% 63.9%;
            --accent: 0 0% 14.9%;
            --accent-foreground: 0 0% 98%;
            --border: 0 0% 14.9%;
            --ring: 0 0% 83.1%;
            --radius: 0.5rem;

            --green: #22c55e;
            --green-muted: rgba(34,197,94,0.15);
            --red: #ef4444;
            --red-muted: rgba(239,68,68,0.15);
            --yellow: #eab308;
            --yellow-muted: rgba(234,179,8,0.12);
            --blue: #3b82f6;
            --blue-muted: rgba(59,130,246,0.12);
        }

        * { margin:0; padding:0; box-sizing:border-box; }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: hsl(var(--background));
            color: hsl(var(--foreground));
            min-height: 100vh;
            -webkit-font-smoothing: antialiased;
        }

        /* ─── HUD BAR (Sticky) ─── */
        .hud-bar {
            position: sticky; top: 0; z-index: 50;
            display: flex; align-items: center; gap: 1px;
            background: hsl(0 0% 5%);
            border-bottom: 1px solid hsl(var(--border));
            padding: 0;
            backdrop-filter: blur(12px);
        }
        .hud-item {
            flex: 1;
            display: flex; flex-direction: column; align-items: center; justify-content: center;
            padding: 10px 12px;
            border-right: 1px solid hsl(var(--border));
            min-height: 60px;
        }
        .hud-item:last-child { border-right: none; }
        .hud-label {
            font-size: 10px; font-weight: 500;
            text-transform: uppercase; letter-spacing: 0.05em;
            color: hsl(var(--muted-foreground));
            margin-bottom: 3px;
        }
        .hud-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 14px; font-weight: 600;
            font-variant-numeric: tabular-nums;
        }
        .hud-sub {
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
            color: hsl(var(--muted-foreground));
        }

        /* ─── MAIN GRID ─── */
        .main-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: auto auto auto;
            gap: 1px;
            background: hsl(var(--border));
            margin-top: 0;
        }

        /* ─── CARD ─── */
        .card {
            background: hsl(var(--card));
            padding: 16px 20px;
        }
        .card-header {
            display: flex; align-items: center; justify-content: space-between;
            margin-bottom: 14px;
        }
        .card-title {
            font-size: 11px; font-weight: 600;
            text-transform: uppercase; letter-spacing: 0.06em;
            color: hsl(var(--muted-foreground));
        }
        .card-badge {
            font-family: 'JetBrains Mono', monospace;
            font-size: 9px; font-weight: 500;
            padding: 2px 8px; border-radius: 9999px;
            background: hsl(var(--secondary));
            color: hsl(var(--muted-foreground));
        }

        /* ─── FULL WIDTH SECTIONS ─── */
        .full-width { grid-column: 1 / -1; }

        /* ─── RISK CARDS ROW ─── */
        .risk-row {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1px;
            background: hsl(var(--border));
        }
        .risk-card {
            background: hsl(var(--card));
            padding: 14px 16px;
            display: flex; flex-direction: column;
        }
        .risk-label {
            font-size: 10px; font-weight: 500;
            text-transform: uppercase; letter-spacing: 0.05em;
            color: hsl(var(--muted-foreground));
            margin-bottom: 6px;
        }
        .risk-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 20px; font-weight: 600;
            font-variant-numeric: tabular-nums;
        }
        .risk-sub {
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
            color: hsl(var(--muted-foreground));
            margin-top: 2px;
        }

        /* ─── TABLE ─── */
        .data-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }
        .data-table thead th {
            font-size: 10px; font-weight: 500;
            text-transform: uppercase; letter-spacing: 0.05em;
            color: hsl(var(--muted-foreground));
            text-align: left;
            padding: 8px 10px;
            border-bottom: 1px solid hsl(var(--border));
        }
        .data-table tbody td {
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            padding: 9px 10px;
            border-bottom: 1px solid hsl(0 0% 10%);
            font-variant-numeric: tabular-nums;
        }
        .data-table tbody tr:hover {
            background: hsl(0 0% 8%);
        }
        .data-table tbody tr:last-child td {
            border-bottom: none;
        }

        /* ─── CHART CONTAINER ─── */
        .chart-container {
            position: relative;
            height: 220px;
            width: 100%;
        }

        /* ─── EXPOSURE BAR ─── */
        .exposure-bar-container {
            margin-top: 12px;
        }
        .exposure-bar-track {
            width: 100%;
            height: 6px;
            background: hsl(var(--secondary));
            border-radius: 3px;
            overflow: hidden;
            display: flex;
        }
        .exposure-bar-fill {
            height: 100%;
            transition: width 0.5s ease;
        }
        .exposure-labels {
            display: flex; justify-content: space-between;
            margin-top: 6px;
        }
        .exposure-label {
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
            color: hsl(var(--muted-foreground));
        }

        /* ─── EXECUTION GRID ─── */
        .exec-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
        }
        .exec-item {
            display: flex; flex-direction: column;
        }
        .exec-label {
            font-size: 10px; font-weight: 500;
            text-transform: uppercase; letter-spacing: 0.05em;
            color: hsl(var(--muted-foreground));
            margin-bottom: 4px;
        }
        .exec-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 16px; font-weight: 600;
            font-variant-numeric: tabular-nums;
        }

        /* ─── LOG TERMINAL ─── */
        .log-terminal {
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            line-height: 1.6;
            height: 160px;
            overflow-y: auto;
            color: hsl(var(--muted-foreground));
            scrollbar-width: thin;
            scrollbar-color: hsl(var(--border)) transparent;
        }
        .log-terminal::-webkit-scrollbar { width: 4px; }
        .log-terminal::-webkit-scrollbar-track { background: transparent; }
        .log-terminal::-webkit-scrollbar-thumb { background: hsl(var(--border)); border-radius: 2px; }
        .log-line { padding: 1px 0; }
        .log-info { color: hsl(var(--muted-foreground)); }
        .log-action { color: var(--blue); }
        .log-buy { color: var(--green); }
        .log-sell { color: var(--red); }
        .log-warn { color: var(--yellow); }
        .log-ts { color: hsl(0 0% 30%); }

        /* ─── STATUS DOTS ─── */
        .status-dot {
            display: inline-block;
            width: 6px; height: 6px;
            border-radius: 50%;
            margin-right: 5px;
            position: relative;
        }
        .status-dot.green { background: var(--green); }
        .status-dot.red { background: var(--red); }
        .status-dot.yellow { background: var(--yellow); }
        .status-dot.green::after {
            content: '';
            position: absolute; inset: -2px;
            border-radius: 50%;
            background: var(--green);
            opacity: 0.3;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { transform: scale(1); opacity: 0.3; }
            50% { transform: scale(1.8); opacity: 0; }
        }

        /* ─── COLORS ─── */
        .text-green { color: var(--green); }
        .text-red { color: var(--red); }
        .text-yellow { color: var(--yellow); }
        .text-muted { color: hsl(var(--muted-foreground)); }
        .text-blue { color: var(--blue); }
        .bg-green-muted { background: var(--green-muted); color: var(--green); }
        .bg-red-muted { background: var(--red-muted); color: var(--red); }

        /* ─── BADGES ─── */
        .badge {
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px; font-weight: 500;
            padding: 2px 7px; border-radius: 4px;
            display: inline-block;
        }
        .badge-long { background: var(--green-muted); color: var(--green); }
        .badge-regime {
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px; font-weight: 600;
            padding: 2px 8px; border-radius: 4px;
        }
        .regime-risk-on { background: var(--green-muted); color: var(--green); }
        .regime-risk-off { background: var(--red-muted); color: var(--red); }
        .regime-unknown { background: var(--yellow-muted); color: var(--yellow); }
        .regime-chop { background: var(--blue-muted); color: var(--blue); }

        /* ─── NO DATA STATE ─── */
        .no-data {
            display: flex; align-items: center; justify-content: center;
            height: 100px;
            color: hsl(var(--muted-foreground));
            font-size: 12px;
        }
    </style>
</head>
<body>

    <!-- ═══ HUD BAR ═══ -->
    <div class="hud-bar" id="hudBar">
        <div class="hud-item">
            <div class="hud-label">Net Liquidation</div>
            <div class="hud-value" id="hudNlv">$0.00</div>
        </div>
        <div class="hud-item">
            <div class="hud-label">Realized P&L</div>
            <div class="hud-value" id="hudRealizedPnl">$0.00</div>
            <div class="hud-sub">Closed trades</div>
        </div>
        <div class="hud-item">
            <div class="hud-label">Unrealized P&L</div>
            <div class="hud-value" id="hudUnrealizedPnl">$0.00</div>
            <div class="hud-sub">Open positions</div>
        </div>
        <div class="hud-item">
            <div class="hud-label">Buying Power</div>
            <div class="hud-value" id="hudBuyingPower">$0.00</div>
        </div>
        <div class="hud-item">
            <div class="hud-label">Margin Util.</div>
            <div class="hud-value" id="hudMarginUtil">0.0%</div>
        </div>
        <div class="hud-item">
            <div class="hud-label">Portfolio Beta</div>
            <div class="hud-value" id="hudBeta">0.00</div>
        </div>
        <div class="hud-item">
            <div class="hud-label">HMM Regime</div>
            <div id="hudHmmRegime"><span class="badge-regime regime-unknown">UNKNOWN</span></div>
            <div class="hud-sub" id="hudHmmConf">—</div>
        </div>
        <div class="hud-item">
            <div class="hud-label">Rule Regime</div>
            <div id="hudRegime"><span class="badge-regime regime-unknown">UNKNOWN</span></div>
        </div>
        <div class="hud-item">
            <div class="hud-label">API Latency</div>
            <div class="hud-value text-green" id="hudLatency">0ms</div>
        </div>
        <div class="hud-item">
            <div class="hud-label">Connection</div>
            <div class="hud-value"><span class="status-dot green"></span><span id="hudConnection" style="font-size:11px">LIVE</span></div>
        </div>
    </div>

    <!-- ═══ MAIN BENTO GRID ═══ -->
    <div class="main-grid">

        <!-- ─── COMMAND CENTER: Positions ─── -->
        <div class="card">
            <div class="card-header">
                <div class="card-title">Active Positions</div>
                <div class="card-badge" id="posCount">0 open</div>
            </div>
            <div id="positionsContainer">
                <table class="data-table" id="positionsTable">
                    <thead>
                        <tr>
                            <th>Symbol</th>
                            <th>Side</th>
                            <th>Entry</th>
                            <th>Mark</th>
                            <th>Size</th>
                            <th>P&L</th>
                            <th>ROE %</th>
                            <th>Time</th>
                        </tr>
                    </thead>
                    <tbody id="positionsBody">
                    </tbody>
                </table>
                <div class="no-data" id="noPositions">No active positions</div>
            </div>
        </div>

        <!-- ─── ATTRIBUTION: Equity Curve ─── -->
        <div class="card">
            <div class="card-header">
                <div class="card-title">Equity Curve</div>
                <div class="card-badge" id="equityAlpha">Alpha: —</div>
            </div>
            <div class="chart-container">
                <canvas id="equityChart"></canvas>
            </div>
            <div class="no-data" id="noEquity" style="display:none">Collecting data points...</div>
        </div>

        <!-- ─── RISK METRICS (Full Width) ─── -->
        <div class="full-width">
            <div class="risk-row">
                <div class="risk-card">
                    <div class="risk-label">VaR (95%, 24h)</div>
                    <div class="risk-value" id="riskVar">$0.00</div>
                    <div class="risk-sub">Max expected loss</div>
                </div>
                <div class="risk-card">
                    <div class="risk-label">Sharpe Ratio</div>
                    <div class="risk-value" id="riskSharpe">0.00</div>
                    <div class="risk-sub">Risk-adjusted return</div>
                </div>
                <div class="risk-card">
                    <div class="risk-label">Sortino Ratio</div>
                    <div class="risk-value" id="riskSortino">0.00</div>
                    <div class="risk-sub">Downside-penalized</div>
                </div>
                <div class="risk-card">
                    <div class="risk-label">Max Drawdown</div>
                    <div class="risk-value" id="riskMDD">0.00%</div>
                    <div class="risk-sub">Peak-to-trough</div>
                </div>
            </div>
        </div>

        <!-- ─── EXPOSURE BREAKDOWN ─── -->
        <div class="card">
            <div class="card-header">
                <div class="card-title">Exposure Breakdown</div>
            </div>
            <div class="exposure-bar-container">
                <div class="exposure-bar-track">
                    <div class="exposure-bar-fill" id="expLongBar" style="width:0%;background:var(--green);"></div>
                    <div class="exposure-bar-fill" id="expCashBar" style="width:100%;background:hsl(var(--secondary));"></div>
                </div>
                <div class="exposure-labels">
                    <div class="exposure-label"><span class="text-green">●</span> Long: <span id="expLong">0.0%</span></div>
                    <div class="exposure-label"><span class="text-muted">●</span> Cash: <span id="expCash">100.0%</span></div>
                    <div class="exposure-label">Net: <span id="expNet">$0.00</span></div>
                </div>
            </div>
            <div style="margin-top:16px;">
                <div class="card-title" style="margin-bottom:10px;">Concentration</div>
                <div id="concentrationBars"></div>
            </div>
        </div>

        <!-- ─── TOP RANKED UNIVERSE ─── -->
        <div class="card">
            <div class="card-header">
                <div class="card-title">Alpha Rankings</div>
                <div class="card-badge" id="rankCount">0 coins</div>
            </div>
            <div id="rankingsContainer">
                <table class="data-table" id="rankingsTable">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Coin</th>
                            <th>Score</th>
                            <th>Mom 7d</th>
                            <th>Mom 24h</th>
                            <th>Vol</th>
                            <th>Trend</th>
                        </tr>
                    </thead>
                    <tbody id="rankingsBody">
                    </tbody>
                </table>
                <div class="no-data" id="noRankings">Waiting for first scan...</div>
            </div>
        </div>

        <!-- ─── EXECUTION QUALITY ─── -->
        <div class="card">
            <div class="card-header">
                <div class="card-title">Execution Quality</div>
                <div class="card-badge" id="totalOrders">0 orders</div>
            </div>
            <div class="exec-grid">
                <div class="exec-item">
                    <div class="exec-label">Fill Rate</div>
                    <div class="exec-value text-green" id="execFillRate">100%</div>
                </div>
                <div class="exec-item">
                    <div class="exec-label">Avg Slippage</div>
                    <div class="exec-value" id="execSlippage">0.00%</div>
                </div>
                <div class="exec-item">
                    <div class="exec-label">Avg Latency</div>
                    <div class="exec-value" id="execLatency">0ms</div>
                </div>
                <div class="exec-item">
                    <div class="exec-label">Status</div>
                    <div class="exec-value"><span class="status-dot green"></span><span style="font-size:12px" id="execStatus">RUNNING</span></div>
                </div>
            </div>
            <div style="margin-top: 16px;">
                <div class="card-title" style="margin-bottom:8px;">Win Rate & Expectancy</div>
                <div style="display:flex; gap:20px; margin-top:6px;">
                    <div>
                        <div class="exec-label">Win Rate</div>
                        <div class="exec-value" id="winRate">—</div>
                    </div>
                    <div>
                        <div class="exec-label">Profit Factor</div>
                        <div class="exec-value" id="profitFactor">—</div>
                    </div>
                    <div>
                        <div class="exec-label">Total P&L</div>
                        <div class="exec-value" id="totalPnl">$0.00</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- ─── LOGS (Full Width) ─── -->
        <div class="card full-width">
            <div class="card-header">
                <div class="card-title"><span class="status-dot green"></span> System Log</div>
                <div class="card-badge" id="logCount">0 events</div>
            </div>
            <div class="log-terminal" id="logTerminal">
                <div class="log-line log-info"><span class="log-ts">[--:--:--]</span> Connecting to Alpha Prime v2...</div>
            </div>
        </div>

    </div>

    <script>
        // ═══ CHART SETUP ═══
        let equityChart = null;

        function initChart() {
            const ctx = document.getElementById('equityChart').getContext('2d');
            equityChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        {
                            label: 'Portfolio',
                            data: [],
                            borderColor: '#3b82f6',
                            backgroundColor: 'rgba(59,130,246,0.05)',
                            borderWidth: 1.5,
                            fill: true,
                            tension: 0.3,
                            pointRadius: 0,
                            pointHoverRadius: 3,
                        },
                        {
                            label: 'BTC Buy & Hold',
                            data: [],
                            borderColor: 'rgba(255,255,255,0.15)',
                            borderWidth: 1,
                            borderDash: [4, 3],
                            fill: false,
                            tension: 0.3,
                            pointRadius: 0,
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: { mode: 'index', intersect: false },
                    plugins: {
                        legend: {
                            display: true,
                            position: 'top',
                            align: 'end',
                            labels: {
                                color: 'rgba(255,255,255,0.5)',
                                font: { family: 'JetBrains Mono', size: 10 },
                                boxWidth: 12, boxHeight: 1, padding: 12,
                                usePointStyle: false,
                            }
                        },
                        tooltip: {
                            backgroundColor: 'hsl(0,0%,8%)',
                            borderColor: 'hsl(0,0%,15%)',
                            borderWidth: 1,
                            titleFont: { family: 'JetBrains Mono', size: 10 },
                            bodyFont: { family: 'JetBrains Mono', size: 11 },
                            padding: 8,
                            displayColors: false,
                            callbacks: {
                                label: function(ctx) {
                                    return ctx.dataset.label + ': $' + ctx.parsed.y.toFixed(2);
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            display: true,
                            grid: { color: 'rgba(255,255,255,0.03)' },
                            ticks: {
                                color: 'rgba(255,255,255,0.25)',
                                font: { family: 'JetBrains Mono', size: 9 },
                                maxTicksLimit: 6,
                            },
                        },
                        y: {
                            display: true,
                            position: 'right',
                            grid: { color: 'rgba(255,255,255,0.03)' },
                            ticks: {
                                color: 'rgba(255,255,255,0.25)',
                                font: { family: 'JetBrains Mono', size: 9 },
                                callback: v => '$' + v.toFixed(0),
                            }
                        }
                    }
                }
            });
        }

        // ═══ DATA FETCHING & RENDERING ═══

        const logs = [];
        let lastLogId = 0;

        function colorVal(value, invert) {
            if (value > 0) return invert ? 'text-red' : 'text-green';
            if (value < 0) return invert ? 'text-green' : 'text-red';
            return 'text-muted';
        }

        async function fetchJSON(url) {
            try {
                const r = await fetch(url);
                if (!r.ok) return null;
                return await r.json();
            } catch(e) { return null; }
        }

        async function updateHUD() {
            const d = await fetchJSON('/api/hud');
            if (!d || d.error) return;

            document.getElementById('hudNlv').textContent = '$' + d.nlv.toFixed(2);

            const realizedPnlEl = document.getElementById('hudRealizedPnl');
            realizedPnlEl.textContent = (d.realized_pnl >= 0 ? '+$' : '-$') + Math.abs(d.realized_pnl).toFixed(2);
            realizedPnlEl.className = 'hud-value ' + colorVal(d.realized_pnl);

            const unrealizedPnlEl = document.getElementById('hudUnrealizedPnl');
            unrealizedPnlEl.textContent = (d.unrealized_pnl >= 0 ? '+$' : '-$') + Math.abs(d.unrealized_pnl).toFixed(2);
            unrealizedPnlEl.className = 'hud-value ' + colorVal(d.unrealized_pnl);


            document.getElementById('hudBuyingPower').textContent = '$' + (d.buying_power || d.nlv).toFixed(2);
            document.getElementById('hudMarginUtil').textContent = d.margin_utilization_pct.toFixed(1) + '%';
            document.getElementById('hudBeta').textContent = d.beta.toFixed(2);

            const regimeEl = document.getElementById('hudRegime');
            const regime = d.system_status || 'UNKNOWN';
            let regimeClass = 'regime-unknown';
            if (regime === 'RISK_ON') regimeClass = 'regime-risk-on';
            else if (regime === 'RISK_OFF') regimeClass = 'regime-risk-off';
            regimeEl.innerHTML = '<span class="badge-regime ' + regimeClass + '">' + regime + '</span>';

            // HMM Regime
            const hmmRegimeEl = document.getElementById('hudHmmRegime');
            const hmmConfEl = document.getElementById('hudHmmConf');
            const hmmRegime = d.hmm_regime || 'UNKNOWN';
            const hmmConf = d.hmm_confidence || 0;
            let hmmClass = 'regime-unknown';
            if (hmmRegime === 'BULL') hmmClass = 'regime-risk-on';
            else if (hmmRegime === 'BEAR') hmmClass = 'regime-risk-off';
            else if (hmmRegime === 'SIDEWAYS') hmmClass = 'regime-chop';
            hmmRegimeEl.innerHTML = '<span class="badge-regime ' + hmmClass + '">' + hmmRegime + '</span>';
            hmmConfEl.textContent = hmmRegime !== 'UNKNOWN' ? (hmmConf * 100).toFixed(0) + '% conf' : '—';

            const lat = d.api_latency_ms;
            const latEl = document.getElementById('hudLatency');
            latEl.textContent = lat.toFixed(0) + 'ms';
            latEl.className = 'hud-value ' + (lat < 100 ? 'text-green' : lat < 500 ? 'text-yellow' : 'text-red');
        }

        async function updatePositions() {
            const d = await fetchJSON('/api/positions');
            if (!d || d.error) return;

            const positions = d.positions || [];
            document.getElementById('posCount').textContent = positions.length + ' open';

            const tbody = document.getElementById('positionsBody');
            const noData = document.getElementById('noPositions');

            if (positions.length === 0) {
                tbody.innerHTML = '';
                noData.style.display = 'flex';
                return;
            }
            noData.style.display = 'none';

            let html = '';
            positions.forEach(p => {
                const pnlClass = p.pnl_dollar >= 0 ? 'text-green' : 'text-red';
                const pnlSign = p.pnl_dollar >= 0 ? '+' : '';
                html += '<tr>';
                html += '<td style="color:hsl(var(--foreground));font-weight:500">' + p.symbol.replace('/USDT','') + '<span class="text-muted" style="font-size:10px;margin-left:3px">' + (p.exchange||'').toUpperCase().slice(0,3) + '</span></td>';
                html += '<td><span class="badge badge-long">LONG</span></td>';
                html += '<td>$' + p.entry_price.toFixed(4) + '</td>';
                html += '<td>$' + p.current_price.toFixed(4) + '</td>';
                html += '<td>$' + p.current_value.toFixed(2) + '</td>';
                html += '<td class="' + pnlClass + '">' + pnlSign + '$' + Math.abs(p.pnl_dollar).toFixed(2) + '</td>';
                html += '<td class="' + pnlClass + '">' + pnlSign + p.pnl_percent.toFixed(1) + '%</td>';
                html += '<td class="text-muted">' + p.cycles_held + 'c</td>';
                html += '</tr>';
            });
            tbody.innerHTML = html;
        }

        async function updateRisk() {
            const d = await fetchJSON('/api/risk');
            if (!d || d.error) return;

            document.getElementById('riskVar').textContent = '$' + d.var_95.toFixed(2);
            document.getElementById('riskSharpe').textContent = d.sharpe_ratio.toFixed(2);
            document.getElementById('riskSortino').textContent = d.sortino_ratio === Infinity ? '∞' : d.sortino_ratio.toFixed(2);
            document.getElementById('riskMDD').textContent = d.max_drawdown_pct.toFixed(2) + '%';

            // Color code
            const sharpeEl = document.getElementById('riskSharpe');
            sharpeEl.className = 'risk-value ' + (d.sharpe_ratio > 1 ? 'text-green' : d.sharpe_ratio > 0 ? 'text-muted' : 'text-red');

            const mddEl = document.getElementById('riskMDD');
            mddEl.className = 'risk-value ' + (d.max_drawdown_pct > 10 ? 'text-red' : d.max_drawdown_pct > 5 ? 'text-yellow' : 'text-muted');

            // Exposure
            if (d.exposure) {
                const exp = d.exposure;
                document.getElementById('expLong').textContent = exp.long_pct.toFixed(1) + '%';
                document.getElementById('expCash').textContent = exp.cash_pct.toFixed(1) + '%';
                document.getElementById('expNet').textContent = '$' + exp.net_exposure.toFixed(2);
                document.getElementById('expLongBar').style.width = exp.long_pct + '%';
                document.getElementById('expCashBar').style.width = exp.cash_pct + '%';
            }
        }

        async function updateExecution() {
            const d = await fetchJSON('/api/execution');
            if (!d || d.error) return;

            document.getElementById('execFillRate').textContent = d.fill_rate_pct.toFixed(0) + '%';
            document.getElementById('execSlippage').textContent = d.avg_slippage_pct.toFixed(2) + '%';

            const latEl = document.getElementById('execLatency');
            latEl.textContent = d.avg_latency_ms.toFixed(0) + 'ms';
            latEl.className = 'exec-value ' + (d.avg_latency_ms < 100 ? 'text-green' : d.avg_latency_ms < 500 ? 'text-yellow' : 'text-red');

            document.getElementById('totalOrders').textContent = d.total_orders + ' orders';

            const slipEl = document.getElementById('execSlippage');
            slipEl.className = 'exec-value ' + (d.avg_slippage_pct <= 0 ? 'text-green' : 'text-red');
        }

        async function updateEquityCurve() {
            const d = await fetchJSON('/api/equity_curve');
            if (!d || d.error || !equityChart) return;

            if (!d.data || d.data.length === 0) {
                document.getElementById('noEquity').style.display = 'flex';
                return;
            }
            document.getElementById('noEquity').style.display = 'none';

            const labels = d.data.map(p => {
                const t = p.timestamp.split('T')[1] || p.timestamp;
                return t.slice(0,5);
            });
            const portData = d.data.map(p => p.portfolio_value);
            const btcData = d.data.map(p => p.btc_benchmark);

            equityChart.data.labels = labels;
            equityChart.data.datasets[0].data = portData;
            equityChart.data.datasets[1].data = btcData;
            equityChart.update('none');

            // Alpha calculation
            if (portData.length > 1) {
                const portReturn = ((portData[portData.length-1] - portData[0]) / portData[0] * 100);
                const btcReturn = btcData.length > 1 ? ((btcData[btcData.length-1] - btcData[0]) / btcData[0] * 100) : 0;
                const alpha = portReturn - btcReturn;
                const alphaEl = document.getElementById('equityAlpha');
                alphaEl.textContent = 'Alpha: ' + (alpha >= 0 ? '+' : '') + alpha.toFixed(2) + '%';
                alphaEl.className = 'card-badge ' + (alpha >= 0 ? 'text-green' : 'text-red');
            }
        }

        async function updateRankings() {
            const d = await fetchJSON('/api/top_ranked');
            if (!d || d.error) return;

            const ranked = d.ranked || [];
            document.getElementById('rankCount').textContent = ranked.length + ' coins';

            const tbody = document.getElementById('rankingsBody');
            const noData = document.getElementById('noRankings');

            if (ranked.length === 0) {
                tbody.innerHTML = '';
                noData.style.display = 'flex';
                return;
            }
            noData.style.display = 'none';

            let html = '';
            ranked.forEach(c => {
                const trendIcon = c.trend_filter === 1 ? '<span class="text-green">▲</span>' : '<span class="text-red">▼</span>';
                const momClass7 = c.momentum_7d >= 0 ? 'text-green' : 'text-red';
                const momClass24 = c.momentum_24h >= 0 ? 'text-green' : 'text-red';
                html += '<tr>';
                html += '<td class="text-muted">' + c.rank + '</td>';
                html += '<td style="color:hsl(var(--foreground));font-weight:500">' + c.symbol.replace('/USDT','') + '<span class="text-muted" style="font-size:10px;margin-left:3px">' + c.exchange.toUpperCase().slice(0,3) + '</span></td>';
                html += '<td class="text-blue">' + c.final_score.toFixed(3) + '</td>';
                html += '<td class="' + momClass7 + '">' + (c.momentum_7d * 100).toFixed(1) + '%</td>';
                html += '<td class="' + momClass24 + '">' + (c.momentum_24h * 100).toFixed(1) + '%</td>';
                html += '<td>' + c.volatility.toFixed(4) + '</td>';
                html += '<td>' + trendIcon + '</td>';
                html += '</tr>';
            });
            tbody.innerHTML = html;
        }

        async function updateStats() {
            const d = await fetchJSON('/api/state');
            if (!d) return;

            const total = d.total_trades || 0;
            const wins = d.winning_trades || 0;
            const pnl = d.total_pnl || 0;

            document.getElementById('winRate').textContent = total > 0 ? (wins/total*100).toFixed(0) + '%' : '—';
            document.getElementById('totalPnl').textContent = (pnl >= 0 ? '+$' : '-$') + Math.abs(pnl).toFixed(2);
            document.getElementById('totalPnl').className = 'exec-value ' + colorVal(pnl);

            // Profit factor
            if (total > 0 && pnl !== 0) {
                const grossWin = pnl > 0 ? pnl : 0;
                const grossLoss = pnl < 0 ? Math.abs(pnl) : 0.01;
                document.getElementById('profitFactor').textContent = (grossWin / grossLoss).toFixed(2);
            }

            // Concentration bars
            if (d.portfolio && d.portfolio.positions) {
                const positions = d.portfolio.positions;
                const total = Object.values(positions).reduce((s,p) => s + (p.current_value||0), 0);
                if (total > 0) {
                    let bars = '';
                    Object.entries(positions).forEach(([sym, p]) => {
                        const pct = ((p.current_value||0) / total * 100);
                        bars += '<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">';
                        bars += '<span style="font-family:JetBrains Mono;font-size:10px;color:hsl(var(--muted-foreground));width:60px">' + sym.replace('/USDT','') + '</span>';
                        bars += '<div style="flex:1;height:4px;background:hsl(var(--secondary));border-radius:2px;overflow:hidden">';
                        bars += '<div style="height:100%;width:' + pct + '%;background:var(--blue);border-radius:2px"></div></div>';
                        bars += '<span style="font-family:JetBrains Mono;font-size:10px;color:hsl(var(--muted-foreground))">' + pct.toFixed(0) + '%</span>';
                        bars += '</div>';
                    });
                    document.getElementById('concentrationBars').innerHTML = bars;
                }
            }
        }

        async function updateLogs() {
            const d = await fetchJSON('/api/logs?since_id=' + lastLogId + '&limit=100');
            if (!d || d.error) return;

            if (d.events && d.events.length > 0) {
                d.events.forEach(e => {
                    // Convert server time (ISO) to Local Browser Time
                    let ts = '--:--:--';
                    if (e.timestamp) {
                        try {
                            const date = new Date(e.timestamp);
                            ts = date.toLocaleTimeString([], { hour12: false });
                        } catch(err) {
                            ts = e.timestamp.split('T')[1].slice(0,8);
                        }
                    }

                    let logType = 'info';
                    if (e.type === 'buy') logType = 'buy';
                    else if (e.type === 'sell') logType = 'sell';
                    else if (e.type === 'trim') logType = 'sell';
                    else if (e.type === 'regime') logType = 'action';
                    else if (e.type === 'signal') logType = 'action';
                    else if (e.type === 'warn' || e.type === 'error') logType = 'warn';
                    logs.push({type: logType, msg: e.message, ts});
                });
                if (logs.length > 200) logs.splice(0, logs.length - 200);
                lastLogId = d.latest_id;
            }

            // Render logs
            const el = document.getElementById('logTerminal');
            const recent = logs.slice(-50);
            el.innerHTML = recent.map(l =>
                '<div class="log-line log-' + l.type + '"><span class="log-ts">[' + l.ts + ']</span> ' + l.msg + '</div>'
            ).join('');
            el.scrollTop = el.scrollHeight;
            document.getElementById('logCount').textContent = logs.length + ' events';
        }

        // ═══ MAIN LOOP ═══
        async function tick() {
            await Promise.all([
                updateHUD(),
                updatePositions(),
                updateRisk(),
                updateExecution(),
                updateEquityCurve(),
                updateRankings(),
                updateStats(),
                updateLogs(),
            ]);
        }

        initChart();
        tick();
        setInterval(tick, 5000);
    </script>
</body>
</html>
"""



@app.route('/')
def dashboard():
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/state')
def api_state():
    """Return JSON-serializable subset of global state"""
    # Filter out non-serializable objects (classes)
    safe_state = {
        'regime': global_state.get('regime', 'UNKNOWN'),
        'regime_details': global_state.get('regime_details', {}),
        'portfolio': global_state.get('portfolio', {}),
        'positions': global_state.get('positions', {}),
        'top_ranked': global_state.get('top_ranked', []),
        'last_slow_update': global_state.get('last_slow_update'),
        'last_medium_update': global_state.get('last_medium_update'),
        'last_fast_update': global_state.get('last_fast_update'),
        # HMM Regime Classifier
        'hmm_regime': global_state.get('hmm_regime', 'UNKNOWN'),
        'hmm_confidence': global_state.get('hmm_confidence', 0.0),
        'hmm_state_history': global_state.get('hmm_state_history', []),
        'hmm_trained': global_state.get('hmm_trained', False),
        # Portfolio Optimizer
        'optimizer_weights': global_state.get('optimizer_weights', {}),
        'optimizer_active': global_state.get('optimizer_active', False),
    }
    return jsonify(safe_state)

@app.route('/api/hud')
def api_hud():
    """HUD bar metrics: NLV, Daily P&L, Beta, Status, Latency"""
    try:
        analytics = global_state.get('analytics')
        execution_monitor = global_state.get('execution_monitor')
        portfolio = global_state.get('portfolio_obj')
        
        if not analytics:
            return jsonify({'error': 'Analytics not initialized'}), 500
        
        nlv = analytics.calculate_nlv()
        daily_pnl = analytics.calculate_daily_pnl()
        beta = analytics.calculate_portfolio_beta()
        exposure = analytics.get_exposure_breakdown()
        
        latency = execution_monitor.get_avg_latency() if execution_monitor else 0
        
        # Calculate realized and unrealized P&L
        realized_pnl = portfolio.realized_pnl if portfolio else 0  # From closed trades
        
        # Sum up unrealized P&L from all open positions
        unrealized_pnl = 0
        if portfolio:
            for symbol, pos in portfolio.positions.items():
                unrealized_pnl += pos.get('unrealized_pnl', 0)
        
        return jsonify({
            'nlv': nlv,
            'daily_pnl_dollar': daily_pnl['pnl_dollar'],
            'daily_pnl_percent': daily_pnl['pnl_percent'],
            'beta': beta,
            'buying_power': global_state.get('portfolio', {}).get('balance', 0),
            'margin_utilization_pct': exposure['long_pct'],
            'system_status': global_state.get('regime', 'UNKNOWN'),
            'api_latency_ms': latency,
            'connection_status': 'CONNECTED',
            'realized_pnl': realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            # HMM fields
            'hmm_regime': global_state.get('hmm_regime', 'UNKNOWN'),
            'hmm_confidence': global_state.get('hmm_confidence', 0.0),
            'hmm_trained': global_state.get('hmm_trained', False),
            # Optimizer fields
            'optimizer_weights': global_state.get('optimizer_weights', {}),
            'optimizer_active': global_state.get('optimizer_active', False),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/risk')
def api_risk():
    """Risk metrics: Sharpe, Sortino, VaR, Max Drawdown"""
    try:
        analytics = global_state.get('analytics')
        
        if not analytics:
            return jsonify({'error': 'Analytics not initialized'}), 500
        
        return jsonify({
            'sharpe_ratio': analytics.calculate_sharpe(),
            'sortino_ratio': analytics.calculate_sortino(),
            'var_95': analytics.calculate_var_95(),
            'max_drawdown_pct': analytics.calculate_max_drawdown(),
            'exposure': analytics.get_exposure_breakdown()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/execution')
def api_execution():
    """Execution quality stats"""
    try:
        execution_monitor = global_state.get('execution_monitor')
        
        if not execution_monitor:
            return jsonify({'error': 'Execution monitor not initialized'}), 500
        
        return jsonify({
            'fill_rate_pct': execution_monitor.get_fill_rate(),
            'avg_slippage_pct': execution_monitor.get_avg_slippage(),
            'avg_latency_ms': execution_monitor.get_avg_latency(),
            'total_orders': len(execution_monitor.orders)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/equity_curve')
def api_equity_curve():
    """Historical equity curve data"""
    try:
        analytics = global_state.get('analytics')
        
        if not analytics:
            return jsonify({'error': 'Analytics not initialized'}), 500
        
        # Return last 100 points (or all if less)
        history = analytics.equity_history[-100:]
        
        return jsonify({
            'data': history,
            'total_points': len(analytics.equity_history)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/positions')
def api_positions():
    """Enhanced position data with risk metrics"""
    try:
        portfolio = global_state.get('portfolio_obj')
        
        if not portfolio:
            return jsonify({'positions': []})
        
        positions_list = []
        for symbol, pos in portfolio.positions.items():
            try:
                exchange = pos.get('exchange', 'binance')
                current_price = portfolio.get_current_price(symbol, exchange)
                
                if not current_price:
                    continue
                
                entry_price = pos.get('entry_price', 0)
                quantity = pos.get('quantity', 0)
                current_value = quantity * current_price
                entry_value = pos.get('entry_value', quantity * entry_price)
                pnl = current_value - entry_value
                pnl_pct = (pnl / entry_value * 100) if entry_value > 0 else 0
                
                positions_list.append({
                    'symbol': symbol,
                    'exchange': exchange,
                    'side': pos.get('side', 'LONG'),
                    'quantity': quantity,
                    'entry_price': entry_price,
                    'current_price': current_price,
                    'entry_value': entry_value,
                    'current_value': current_value,
                    'pnl_dollar': pnl,
                    'pnl_percent': pnl_pct,
                    'cycles_held': pos.get('cycles_held', 0),
                    'entry_time': pos.get('entry_time', '')
                })
            except Exception as e:
                # Skip this position if there's an error
                print(f"Error processing position {symbol}: {e}")
                continue
        
        return jsonify({'positions': positions_list})
    except Exception as e:
        print(f"Error in api_positions: {e}")
        return jsonify({'positions': []})

@app.route('/api/logs')
def api_logs():
    """Server-side event log stream with incremental polling support."""
    try:
        since_id = int(request.args.get('since_id', 0))
        limit = int(request.args.get('limit', 100))
        events = event_logger.get_events(since_id=since_id, limit=limit)
        return jsonify({
            'events': events,
            'latest_id': event_logger.get_latest_id()
        })
    except Exception as e:
        return jsonify({'error': str(e), 'events': []}), 500

@app.route('/api/top_ranked')
def api_top_ranked():
    """Current top-ranked coins from the latest scoring cycle."""
    try:
        top_coins = global_state.get('top_coins', [])
        # Build safe list (some coin detail values may not be JSON-safe)
        ranked = []
        for i, coin in enumerate(top_coins):
            ranked.append({
                'rank': i + 1,
                'symbol': coin.get('symbol', ''),
                'exchange': coin.get('exchange', 'binance'),
                'final_score': float(coin.get('final_score', 0)),
                'momentum_7d': float(coin.get('momentum_7d', 0)),
                'momentum_24h': float(coin.get('momentum_24h', 0)),
                'volatility': float(coin.get('volatility', 0)),
                'trend_filter': int(coin.get('trend_filter', 0)),
                'weight': float(coin.get('weight', 0)),
            })
        return jsonify({
            'ranked': ranked,
            'regime': global_state.get('regime', 'UNKNOWN'),
            'total_scanned': len(global_state.get('top_coins', [])),
            'last_update': global_state.get('last_slow_update')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def run_flask():
    app.run(host='0.0.0.0', port=5050, debug=False, use_reloader=False)

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Start Flask in background thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    event_logger.log('info', 'Alpha Prime v2 engine starting — 3-speed architecture active')
    print("\n🌐 Dashboard running at http://localhost:5050\n")
    time.sleep(2)
    
    # Run Alpha Prime
    engine = AlphaPrime()
    engine.run()
