"""
Multi-Exchange Scanner for Alpha Prime v2
Scans multiple cryptocurrency exchanges and aggregates viable trading opportunities.
"""

import ccxt
from datetime import datetime

class MultiExchangeScanner:
    """
    Scans multiple exchanges and provides unified interface for:
    - Market data fetching
    - Coin filtering
    - Price lookups
    - Order execution routing
    """
    
    def __init__(self, exchanges=['binance', 'kucoin', 'gateio', 'mexc', 'bybit', 'kraken', 'bitget']):
        """
        Initialize scanner with specified exchanges.
        
        Args:
            exchanges: List of exchange names to enable (default: all 4)
        """
        self.exchanges = {}
        self.enabled_exchanges = exchanges
        
        print(f"\n{'='*80}")
        print(f"🌐 MULTI-EXCHANGE SCANNER INITIALIZATION")
        print(f"{'='*80}")
        
        for name in exchanges:
            try:
                self.exchanges[name] = self._init_exchange(name)
                print(f"   ✅ {name.upper()} connected")
            except Exception as e:
                print(f"   ⚠️  {name.upper()} failed to connect: {e}")
                print(f"      Continuing without {name}...")
        
        print(f"{'='*80}\n")
    
    def _init_exchange(self, name):
        """Initialize a single exchange connection."""
        exchange_class = getattr(ccxt, name)
        exchange = exchange_class({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
            }
        })
        return exchange
    
    def get_all_viable_coins(self, min_volume=2_000_000):
        """
        Fetch and filter coins from all enabled exchanges.
        
        Args:
            min_volume: Minimum 24h volume in USD (default: $2M)
            
        Returns:
            List of coin dicts with symbol, exchange, volume, price
        """
        all_coins = []
        
        for exchange_name, exchange in self.exchanges.items():
            try:
                coins = self._scan_exchange(exchange, exchange_name, min_volume)
                all_coins.extend(coins)
                print(f"   🔎 Scanned {len(coins)} coins on {exchange_name.upper()}")
            except Exception as e:
                print(f"   ⚠️  Error scanning {exchange_name}: {e}")
        
        # Remove duplicates (prefer exchange with higher volume)
        unique_coins = self._deduplicate_coins(all_coins)
        
        print(f"   📊 Total unique coins: {len(unique_coins)}")
        return unique_coins
    
    def _scan_exchange(self, exchange, exchange_name, min_volume):
        """Scan a single exchange for viable coins."""
        viable = []
        
        try:
            markets = exchange.load_markets()
            tickers = exchange.fetch_tickers()
            
            # Filter for USDT pairs only
            usdt_symbols = [s for s in markets if s.endswith('/USDT') or s.endswith('USDT')]
            
            for symbol in usdt_symbols:
                # Normalize symbol format
                normalized_symbol = symbol if '/' in symbol else symbol.replace('USDT', '/USDT')
                
                # Check if market exists and is active
                if normalized_symbol not in markets:
                    continue
                if not markets[normalized_symbol].get('active', False):
                    continue
                
                # Check volume threshold
                if normalized_symbol in tickers:
                    ticker = tickers[normalized_symbol]
                    volume = ticker.get('quoteVolume', 0)
                    price = ticker.get('last', 0)
                    
                    if volume >= min_volume and price > 0:
                        # Extract base currency (e.g., BTC from BTC/USDT)
                        base_currency = normalized_symbol.split('/')[0]
                        
                        viable.append({
                            'symbol': normalized_symbol,
                            'base_currency': base_currency,
                            'exchange': exchange_name,
                            'volume': volume,
                            'price': price
                        })
        
        except Exception as e:
            print(f"      Error during {exchange_name} scan: {e}")
        
        return viable
    
    def _deduplicate_coins(self, all_coins):
        """
        Remove duplicate coins across exchanges.
        If same coin exists on multiple exchanges, keep the one with higher volume.
        """
        coin_map = {}
        
        for coin in all_coins:
            base = coin['base_currency']
            
            if base not in coin_map:
                coin_map[base] = coin
            else:
                # Keep the exchange with higher volume
                if coin['volume'] > coin_map[base]['volume']:
                    coin_map[base] = coin
        
        return list(coin_map.values())
    
    def get_current_price(self, symbol, exchange_name):
        """
        Fetch current price from specific exchange.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            exchange_name: Exchange to query
            
        Returns:
            Current price or None if error
        """
        try:
            if exchange_name not in self.exchanges:
                print(f"   ⚠️  Exchange {exchange_name} not available")
                return None
            
            exchange = self.exchanges[exchange_name]
            ticker = exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            print(f"   ⚠️  Error fetching price for {symbol} on {exchange_name}: {e}")
            return None
    
    def fetch_ohlcv(self, symbol, exchange_name, timeframe='1d', limit=30):
        """
        Fetch historical OHLCV data from specific exchange.
        
        Args:
            symbol: Trading pair
            exchange_name: Exchange to query
            timeframe: Candle timeframe (default: 1d)
            limit: Number of candles (default: 30)
            
        Returns:
            List of OHLCV candles or empty list if error
        """
        try:
            if exchange_name not in self.exchanges:
                return []
            
            exchange = self.exchanges[exchange_name]
            candles = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            return candles
        except Exception as e:
            print(f"   ⚠️  Error fetching OHLCV for {symbol} on {exchange_name}: {e}")
            return []
    
    def execute_order(self, symbol, exchange_name, side, amount):
        """
        Execute order on specific exchange (paper trading simulation).
        
        Args:
            symbol: Trading pair
            exchange_name: Exchange to execute on
            side: 'buy' or 'sell'
            amount: Order amount in USDT
            
        Returns:
            Simulated order dict
        """
        # For paper trading, just return simulated order
        return {
            'symbol': symbol,
            'exchange': exchange_name,
            'side': side,
            'amount': amount,
            'timestamp': datetime.now().isoformat()
        }
