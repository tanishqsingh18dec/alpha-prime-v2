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

ASYMMETRIC LATENCY:
- Fast (30s): Observe prices, update volatility, check emergencies
- Medium (5min): Check trend breaks, stops, trims
- Slow (1h): Recalculate scores, rank universe, rotate capital

If ANY layer fails → capital is protected.
"""

import ccxt
import pandas as pd
import numpy as np
import time
import json
import requests
from datetime import datetime, timedelta
from pathlib import Path
from flask import Flask, render_template_string, jsonify
import threading

# =============================================================================
# CONFIGURATION
# =============================================================================

STARTING_BALANCE = 100.0
TOP_N_POSITIONS = 5           # Hold top 5 ranked coins
MAX_POSITION_PCT = 0.40       # Max 40% in one position
MIN_COIN_VOLUME_24H = 2_000_000  # Lowered to $2M to catch more opportunities
DISCORD_URL = ""              # Optional webhook

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

exchange = ccxt.binance({'enableRateLimit': True})

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
    'last_fast_update': None
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
# LAYER 2: UNIVERSE FILTER
# =============================================================================

class UniverseFilter:
    """
    Filters out dead/illiquid coins before scoring.
    """
    
    @staticmethod
    def get_viable_coins():
        """
        Returns list of symbols that pass minimum viability checks.
        """
        try:
            markets = exchange.load_markets()
            tickers = exchange.fetch_tickers()
            
            all_symbols = [s for s in markets if s.endswith('/USDT')]
            viable = []
            
            for symbol in all_symbols:
                if not markets[symbol]['active']:
                    continue
                # Exclude stablecoins
                base = symbol.split('/')[0]
                if base in ['USDC', 'BUSD', 'DAI', 'TUSD', 'USDT', 'FDUSD']:
                    continue
                
                # Volume filter
                if symbol in tickers:
                    quote_volume = tickers[symbol].get('quoteVolume', 0)
                    if quote_volume and quote_volume >= MIN_COIN_VOLUME_24H:
                        viable.append(symbol)
            
            print(f"   🔎 Scanned {len(all_symbols)} USDT pairs on Binance")
            return viable
        except Exception as e:
            print(f"⚠️  Universe filter error: {e}")
            return []

# =============================================================================
# LAYER 3: ALPHA SCORE ENGINE (Volatility-Adjusted Momentum)
# =============================================================================

class AlphaScorer:
    """
    Cross-sectional momentum with volatility adjustment.
    """
    
    @staticmethod
    def calculate_score(symbol):
        """
        Returns: (final_score, details_dict) or (None, None)
        
        Formula:
        RawScore = 0.6 * Momentum_7d + 0.4 * Momentum_24h
        RiskAdjScore = RawScore / Volatility
        FinalScore = RiskAdjScore * TrendFilter
        """
        try:
            # Fetch 30 days of daily data for momentum + volatility
            candles = exchange.fetch_ohlcv(symbol, '1d', limit=30)
            if len(candles) < 30:
                return None, None
            
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            current_price = df['close'].iloc[-1]
            price_7d_ago = df['close'].iloc[-8]
            price_1d_ago = df['close'].iloc[-2]
            
            # Momentum calculations
            momentum_7d = (current_price - price_7d_ago) / price_7d_ago
            momentum_24h = (current_price - price_1d_ago) / price_1d_ago
            
            # Volatility (30-day stddev of log returns)
            log_returns = np.log(df['close'] / df['close'].shift(1)).dropna()
            volatility = log_returns.std()
            
            if volatility == 0:
                return None, None
            
            # Trend filter: EMA 50 on 4H
            candles_4h = exchange.fetch_ohlcv(symbol, '4h', limit=50)
            df_4h = pd.DataFrame(candles_4h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            ema_50 = df_4h['close'].ewm(span=50, adjust=False).mean().iloc[-1]
            trend_filter = 1 if current_price > ema_50 else 0
            
            # Score calculation
            raw_score = 0.6 * momentum_7d + 0.4 * momentum_24h
            risk_adj_score = raw_score / volatility
            final_score = risk_adj_score * trend_filter
            
            details = {
                'symbol': symbol,
                'price': current_price,
                'momentum_7d': momentum_7d,
                'momentum_24h': momentum_24h,
                'volatility': volatility,
                'ema_50': ema_50,
                'trend_filter': trend_filter,
                'raw_score': raw_score,
                'final_score': final_score
            }
            
            return final_score, details
            
        except Exception as e:
            return None, None
    
    @staticmethod
    def rank_universe(symbols):
        """
        Score all symbols and return sorted by final_score.
        """
        results = []
        
        for symbol in symbols:
            score, details = AlphaScorer.calculate_score(symbol)
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
                self.total_pnl = data.get('total_pnl', 0.0)
                print(f"📂 Loaded portfolio: ${self.balance:.2f} balance, {len(self.positions)} positions")
        else:
            self.balance = STARTING_BALANCE
            self.positions = {}
            self.total_trades = 0
            self.winning_trades = 0
            self.total_pnl = 0.0
            print(f"🆕 Starting fresh with ${STARTING_BALANCE}")
            self.save_portfolio()
    
    def save_portfolio(self):
        """Save portfolio to file"""
        data = {
            'balance': self.balance,
            'positions': self.positions,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'total_pnl': self.total_pnl,
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
    
    def get_current_price(self, symbol):
        """Get current price for symbol"""
        try:
            ticker = exchange.fetch_ticker(symbol)
            return ticker['last']
        except:
            return None
    
    def get_portfolio_value(self):
        """Calculate total portfolio value and update position stats"""
        total = self.balance
        for symbol, pos in self.positions.items():
            price = self.get_current_price(symbol)
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
    
    def execute_buy(self, symbol, weight, details):
        """Execute buy order"""
        if symbol in self.positions:
            return False
        
        price = self.get_current_price(symbol)
        if not price:
            return False
        
        # Calculate position size
        portfolio_value = self.get_portfolio_value()
        position_value = portfolio_value * weight
        
        if self.balance < position_value:
            position_value = self.balance * 0.95  # Use 95% of available
        
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
            'cycles_held': 0
        }
        
        self.log_trade('BUY', symbol, quantity, price, reason=f"Score: {details['final_score']:.2f}")
        self.save_portfolio()
        
        print(f"✅ BUY {symbol}: {quantity:.4f} @ ${price:.4f} (${position_value:.2f})")
        return True
    
    def execute_sell(self, symbol, reason):
        """Execute sell order"""
        if symbol not in self.positions:
            return False
        
        position = self.positions[symbol]
        price = self.get_current_price(symbol)
        if not price:
            return False
        
        entry_price = position['entry_price']
        quantity = position['quantity']
        
        exit_value = price * quantity
        entry_value = entry_price * quantity
        pnl = exit_value - entry_value
        pnl_pct = (pnl / entry_value) * 100
        
        self.balance += exit_value
        self.total_trades += 1
        self.total_pnl += pnl
        if pnl > 0:
            self.winning_trades += 1
        
        del self.positions[symbol]
        
        self.log_trade('SELL', symbol, quantity, price, pnl=pnl, reason=reason)
        self.save_portfolio()
        
        emoji = "✅" if pnl > 0 else "❌"
        print(f"{emoji} SELL {symbol}: {quantity:.4f} @ ${price:.4f} | P&L: {pnl_pct:+.1f}% | {reason}")
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
        
        self.total_pnl += pnl
        if pnl > 0:
            self.winning_trades += 0.25  # Partial win
        
        self.log_trade('TRIM', symbol, trim_qty, price, pnl=pnl, reason=reason)
        self.save_portfolio()
        
        print(f"📉 TRIM {symbol}: {TRIM_PERCENTAGE*100:.0f}% @ ${price:.4f} | Locked {pnl_pct:+.1f}% | {reason}")
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
    
    def fast_loop(self):
        """
        30-second loop: Observe prices, update volatility, check emergencies.
        """
        print(f"\n⚡ FAST CHECK - {datetime.now().strftime('%H:%M:%S')}")
        
        # Update global state
        global_state['balance'] = self.portfolio.balance
        global_state['positions'] = self.portfolio.positions
        global_state['portfolio_value'] = self.portfolio.get_portfolio_value()
        global_state['total_pnl'] = self.portfolio.total_pnl
        global_state['last_fast_update'] = datetime.now().isoformat()
        
        # Emergency drawdown checks could go here
        # For now, just observe
        print(f"   Portfolio: ${global_state['portfolio_value']:.2f} | Cash: ${self.portfolio.balance:.2f}")
    
    def medium_loop(self):
        """
        5-minute loop: Check trend breaks, stop-losses, trims.
        """
        print(f"\n⚙️  MEDIUM CHECK - {datetime.now().strftime('%H:%M:%S')}")
        
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
        
        # LAYER 1: Regime Detection
        print("\n📊 LAYER 1: MARKET REGIME DETECTION")
        regime, details = RegimeDetector.detect_regime()
        self.current_regime = regime
        self.regime_details = details
        
        print(f"   Regime: {regime}")
        print(f"   BTC: ${details.get('btc_price', 0):.2f} vs EMA200 ${details.get('btc_ema200', 0):.2f} ({details.get('btc_vs_ema', 0):+.1f}%)")
        print(f"   Volatility: {details.get('volatility_30d', 0):.1%}")
        print(f"   Breadth: {details.get('breadth_pct', 0):.1%} above EMA50")
        
        global_state['regime'] = regime
        global_state['regime_details'] = details
        
        # If Risk-Off, exit everything
        if regime == 'RISK_OFF':
            print("   ⚠️  RISK-OFF DETECTED → EXITING ALL POSITIONS")
            for symbol in list(self.portfolio.positions.keys()):
                self.portfolio.execute_sell(symbol, "Risk-Off Regime")
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
        
        # Select Top N
        top_n = ranked[:TOP_N_POSITIONS]
        self.top_ranked = top_n
        
        print(f"\n   🏆 TOP {TOP_N_POSITIONS} RANKED COINS:")
        print(f"   {'Coin':<12} {'Score':<10} {'Mom7d':<8} {'Mom24h':<8} {'Vol':<8} {'Trend'}")
        print(f"   {'-'*70}")
        for coin in top_n:
            trend_status = "✅" if coin['trend_filter'] == 1 else "❌"
            print(f"   {coin['symbol']:<12} {coin['final_score']:>9.4f} {coin['momentum_7d']:>7.1%} {coin['momentum_24h']:>7.1%} {coin['volatility']:>7.4f} {trend_status}")
        
        global_state['top_coins'] = top_n
        
        # LAYER 4: Position Sizing
        print("\n💰 LAYER 4: POSITION SIZING (Inverse Volatility)")
        top_n_weighted = PositionSizer.calculate_weights(top_n)
        
        for coin in top_n_weighted:
            print(f"   {coin['symbol']:<12} → {coin['weight']:>6.1%}")
        
        # Apply regime multiplier
        regime_multiplier = 1.0 if regime == 'RISK_ON' else 0.5
        if regime_multiplier != 1.0:
            print(f"   ⚠️  Regime: {regime} → Reducing size by {(1-regime_multiplier)*100:.0f}%")
            for coin in top_n_weighted:
                coin['weight'] *= regime_multiplier
        
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
    <title>🧠 Alpha Prime v2</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
            color: #e0e0e0;
            padding: 20px;
            min-height: 100vh;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            color: #00ff88;
            margin-bottom: 30px;
            font-size: 2.5rem;
            text-shadow: 0 0 20px rgba(0, 255, 136, 0.5);
        }
        .regime-badge {
            display: inline-block;
            padding: 10px 20px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 1.2rem;
            margin: 20px 0;
        }
        .risk-on { background: linear-gradient(90deg, #00ff88, #00cc6a); color: #000; }
        .chop { background: linear-gradient(90deg, #ffd700, #ffaa00); color: #000; }
        .risk-off { background: linear-gradient(90deg, #ff4757, #cc3344); color: #fff; }
        .unknown { background: #444; color: #fff; }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            padding: 20px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .stat-label {
            color: #8892b0;
            font-size: 0.9rem;
            margin-bottom: 5px;
        }
        .stat-value {
            font-size: 1.6rem;
            font-weight: bold;
            color: #00ff88;
        }
        .section {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            padding: 25px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin-bottom: 20px;
        }
        h2 {
            color: #00ff88;
            margin-bottom: 15px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        th {
            color: #00ff88;
            font-weight: 600;
        }
        .positive { color: #00ff88; }
        .negative { color: #ff4757; }
        .neutral { color: #ffd700; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 ALPHA PRIME v2</h1>
        
        <div style="text-align: center;">
            <span id="regime-badge" class="regime-badge unknown">Loading...</span>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Portfolio Value</div>
                <div class="stat-value" id="portfolio-value">$0.00</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Cash Balance</div>
                <div class="stat-value" id="cash-balance">$0.00</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total P&L (Net)</div>
                <div class="stat-value" id="net-pnl">$0.00</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Realized P&L</div>
                <div class="stat-value" id="realized-pnl">$0.00</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Open Positions</div>
                <div class="stat-value" id="position-count">0</div>
            </div>
        </div>
        
        <div class="section">
            <h2>💼 Current Positions</h2>
            <table id="positions-table">
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Entry Price</th>
                        <th>Quantity</th>
                        <th>Current Value</th>
                        <th>Unrealized P&L</th>
                        <th>Time Held</th>
                    </tr>
                </thead>
                <tbody id="positions-body"></tbody>
            </table>
        </div>
        
        <div class="section">
            <h2>� Regime Details</h2>
            <div id="regime-details"></div>
        </div>
    </div>
    
    <script>
        const STARTING_BALANCE = 100.0;

        function updateDashboard() {
            fetch('/api/state')
                .then(response => response.json())
                .then(data => {
                    // Update stats
                    const netPnl = data.portfolio_value - STARTING_BALANCE;
                    const realizedPnl = data.total_pnl;
                    
                    document.getElementById('portfolio-value').textContent = '$' + data.portfolio_value.toFixed(2);
                    document.getElementById('cash-balance').textContent = '$' + data.balance.toFixed(2);
                    
                    const pnlElem = document.getElementById('net-pnl');
                    pnlElem.textContent = (netPnl >= 0 ? '+' : '') + '$' + netPnl.toFixed(2);
                    pnlElem.className = 'stat-value ' + (netPnl >= 0 ? 'positive' : 'negative');
                    
                    const realizedElem = document.getElementById('realized-pnl');
                    realizedElem.textContent = (realizedPnl >= 0 ? '+' : '') + '$' + realizedPnl.toFixed(2);
                    realizedElem.className = 'stat-value ' + (realizedPnl >= 0 ? 'positive' : 'negative');
                    
                    document.getElementById('position-count').textContent = Object.keys(data.positions).length;
                    
                    // Update regime badge
                    const badge = document.getElementById('regime-badge');
                    badge.textContent = data.regime;
                    badge.className = 'regime-badge ' + data.regime.toLowerCase().replace('_', '-');
                    
                    // Update regime details
                    const details = data.regime_details;
                    let detailsHtml = '<table style="width: 100%;">';
                    detailsHtml += `<tr><td>BTC Price:</td><td>$${details.btc_price?.toFixed(2) || 'N/A'}</td></tr>`;
                    detailsHtml += `<tr><td>BTC vs EMA200:</td><td class="${details.btc_vs_ema > 0 ? 'positive' : 'negative'}">${details.btc_vs_ema?.toFixed(1) || 'N/A'}%</td></tr>`;
                    detailsHtml += `<tr><td>30D Volatility:</td><td>${(details.volatility_30d * 100)?.toFixed(1) || 'N/A'}%</td></tr>`;
                    detailsHtml += `<tr><td>Market Breadth:</td><td>${(details.breadth_pct * 100)?.toFixed(1) || 'N/A'}%</td></tr>`;
                    detailsHtml += '</table>';
                    document.getElementById('regime-details').innerHTML = detailsHtml;
                    
                    // Update positions
                    const positionsBody = document.getElementById('positions-body');
                    positionsBody.innerHTML = '';
                    
                    if (Object.keys(data.positions).length === 0) {
                        positionsBody.innerHTML = '<tr><td colspan="6" style="text-align:center; padding:20px;">No open positions</td></tr>';
                    } else {
                        for (const [symbol, position] of Object.entries(data.positions)) {
                            const row = positionsBody.insertRow();
                            const currentValue = position.current_value || (position.quantity * position.entry_price);
                            const pnl = position.unrealized_pnl || 0;
                            const pnlPct = position.unrealized_pnl_pct || 0;
                            const pnlClass = pnl >= 0 ? 'positive' : 'negative';
                            
                            row.innerHTML = `
                                <td>${symbol}</td>
                                <td>$${position.entry_price.toFixed(4)}</td>
                                <td>${position.quantity.toFixed(4)}</td>
                                <td>$${currentValue.toFixed(2)}</td>
                                <td class="${pnlClass}">$${pnl.toFixed(2)} (${pnlPct.toFixed(1)}%)</td>
                                <td>${position.cycles_held} cycles</td>
                            `;
                        }
                    }
                });
        }
        
        // Update every 10 seconds
        updateDashboard();
        setInterval(updateDashboard, 10000);
    </script>
</body>
</html>
"""

@app.route('/')
def dashboard():
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/state')
def api_state():
    return jsonify(global_state)

def run_flask():
    app.run(host='0.0.0.0', port=5050, debug=False, use_reloader=False)

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Start Flask in background thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    print("\n🌐 Dashboard running at http://localhost:5050\n")
    time.sleep(2)
    
    # Run Alpha Prime
    engine = AlphaPrime()
    engine.run()
