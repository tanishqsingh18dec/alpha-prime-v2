"""
Multi-Exchange Scanner for Alpha Prime v2
Scans multiple cryptocurrency exchanges and aggregates viable trading opportunities.
Includes derivatives data (funding rates, OI) and order book microstructure.

Architecture:
  - Sync connections  → used by exit checks, trims, single-coin lookups
  - Async connections → used by batch_prefetch_scoring_data() for 10x faster scoring
"""

import ccxt
import ccxt.async_support as ccxt_async
import asyncio
from datetime import datetime

# Exchanges that support perpetual futures via CCXT (used for funding/OI)
SWAP_CAPABLE = {'binance', 'bybit', 'gateio', 'mexc', 'bitget', 'kucoin'}

# Max concurrent requests per exchange to avoid rate limits
MAX_CONCURRENT_PER_EXCHANGE = 5


class MultiExchangeScanner:
    """
    Scans multiple exchanges and provides unified interface for:
    - Market data fetching (spot + derivatives)
    - Coin filtering
    - Price lookups
    - Order book microstructure (spread, depth, imbalance)
    - Funding rates & Open Interest (via swap connections)
    - Batch async prefetching for fast scoring cycles
    """

    def __init__(self, exchanges=['binance', 'kucoin', 'gateio', 'mexc', 'bybit', 'kraken', 'bitget']):
        """
        Initialize scanner with spot + swap connections per exchange.

        Args:
            exchanges: List of exchange names to enable
        """
        self.exchanges = {}       # Sync spot connections
        self.swap_exchanges = {}  # Sync swap connections (for funding + OI)
        self.enabled_exchanges = exchanges

        print(f"\n{'='*80}")
        print(f"🌐 MULTI-EXCHANGE SCANNER INITIALIZATION")
        print(f"{'='*80}")

        for name in exchanges:
            try:
                self.exchanges[name] = self._init_exchange(name, 'spot')
                print(f"   ✅ {name.upper()} spot connected")
            except Exception as e:
                print(f"   ⚠️  {name.upper()} spot failed: {e}")

            # Initialize swap connection for derivatives data (funding, OI)
            if name in SWAP_CAPABLE:
                try:
                    self.swap_exchanges[name] = self._init_exchange(name, 'swap')
                    print(f"   ✅ {name.upper()} swap connected (derivatives)")
                except Exception:
                    pass  # Non-critical — derivatives data is optional

        print(f"{'='*80}\n")

    def _init_exchange(self, name, market_type='spot'):
        """Initialize a single sync exchange connection."""
        exchange_class = getattr(ccxt, name)
        exchange = exchange_class({
            'enableRateLimit': True,
            'options': {
                'defaultType': market_type,
            }
        })
        return exchange

    def _init_async_exchange(self, name, market_type='spot'):
        """Initialize a single async exchange connection (separate instance)."""
        exchange_class = getattr(ccxt_async, name)
        exchange = exchange_class({
            'enableRateLimit': True,
            'options': {
                'defaultType': market_type,
            }
        })
        return exchange

    # =========================================================================
    # ASYNC BATCH PREFETCHING — the 10x speedup
    # =========================================================================

    def prefetch_scoring_data(self, coin_data_list):
        """
        Sync wrapper: Fetch all scoring data (OHLCV 1d, OHLCV 4h, funding rate,
        order book) for ALL coins concurrently.

        Returns:
            dict: { symbol: { 'ohlcv_1d': [...], 'ohlcv_4h': [...],
                               'funding_rate': float|None,
                               'order_book': dict|None } }
        """
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self._async_batch_prefetch(coin_data_list)
            )
            loop.run_until_complete(self._close_async_exchanges())
            loop.close()
            return result
        except Exception as e:
            print(f"   ⚠️  Async prefetch failed, returning empty: {e}")
            return {}

    async def _close_async_exchanges(self):
        """Close all async exchange connections to prevent resource leaks."""
        tasks = []
        for ex in self._async_spot_pool.values():
            tasks.append(ex.close())
        for ex in self._async_swap_pool.values():
            tasks.append(ex.close())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._async_spot_pool = {}
        self._async_swap_pool = {}

    async def _async_batch_prefetch(self, coin_data_list):
        """
        Core async method: creates fresh async exchange instances,
        fires all fetch tasks concurrently with per-exchange semaphores,
        then returns aggregated results.
        """
        # Create fresh async instances (each coin fetch gets its own safe path)
        self._async_spot_pool = {}
        self._async_swap_pool = {}
        exchange_semaphores = {}

        for name in self.enabled_exchanges:
            if name in self.exchanges:
                self._async_spot_pool[name] = self._init_async_exchange(name, 'spot')
            if name in self.swap_exchanges:
                self._async_swap_pool[name] = self._init_async_exchange(name, 'swap')
            exchange_semaphores[name] = asyncio.Semaphore(MAX_CONCURRENT_PER_EXCHANGE)

        # Build all fetch tasks
        results = {}
        tasks = []

        for coin in coin_data_list:
            symbol = coin['symbol']
            exchange_name = coin.get('exchange', 'binance')
            results[symbol] = {
                'ohlcv_1d': [],
                'ohlcv_4h': [],
                'funding_rate': None,
                'order_book': None,
            }
            sem = exchange_semaphores.get(exchange_name, asyncio.Semaphore(3))

            # 4 fetches per coin, all concurrent
            tasks.append(self._safe_fetch(
                sem, self._async_fetch_ohlcv,
                symbol, exchange_name, '1d', 30,
                results, symbol, 'ohlcv_1d'))
            tasks.append(self._safe_fetch(
                sem, self._async_fetch_ohlcv,
                symbol, exchange_name, '4h', 50,
                results, symbol, 'ohlcv_4h'))
            tasks.append(self._safe_fetch(
                sem, self._async_fetch_funding_rate,
                symbol, exchange_name,
                results, symbol, 'funding_rate'))
            tasks.append(self._safe_fetch(
                sem, self._async_fetch_order_book_summary,
                symbol, exchange_name,
                results, symbol, 'order_book'))

        # Fire all at once (semaphores limit per-exchange concurrency)
        await asyncio.gather(*tasks, return_exceptions=True)
        return results

    async def _safe_fetch(self, semaphore, fetch_fn, *args):
        """
        Rate-limited fetch wrapper: acquires per-exchange semaphore
        before calling the async fetch function.
        Last 3 args are always: results_dict, symbol, key
        """
        results_dict = args[-3]
        symbol = args[-2]
        key = args[-1]
        fetch_args = args[:-3]

        async with semaphore:
            try:
                value = await fetch_fn(*fetch_args)
                results_dict[symbol][key] = value
            except Exception:
                pass  # Keep default (empty list / None)

    async def _async_fetch_ohlcv(self, symbol, exchange_name, timeframe, limit):
        """Async OHLCV fetch."""
        ex = self._async_spot_pool.get(exchange_name)
        if not ex:
            return []
        try:
            candles = await ex.fetch_ohlcv(symbol, timeframe, limit=limit)
            return candles
        except Exception:
            return []

    async def _async_fetch_funding_rate(self, symbol, exchange_name):
        """Async funding rate fetch."""
        ex = self._async_swap_pool.get(exchange_name)
        if not ex:
            return None
        try:
            swap_symbol = symbol.replace('/USDT', '/USDT:USDT')
            result = await ex.fetch_funding_rate(swap_symbol)
            return result.get('fundingRate')
        except Exception:
            return None

    async def _async_fetch_order_book_summary(self, symbol, exchange_name):
        """Async order book fetch + microstructure calculation."""
        ex = self._async_spot_pool.get(exchange_name)
        if not ex:
            return None
        try:
            book = await ex.fetch_order_book(symbol, limit=20)

            if not book['bids'] or not book['asks']:
                return None

            best_bid = book['bids'][0][0]
            best_ask = book['asks'][0][0]
            bid_vol  = book['bids'][0][1]  # Volume at best bid
            ask_vol  = book['asks'][0][1]  # Volume at best ask
            mid = (best_bid + best_ask) / 2
            spread_pct = ((best_ask - best_bid) / mid) * 100 if mid > 0 else 0

            bid_depth = sum(price * qty for price, qty in book['bids'])
            ask_depth = sum(price * qty for price, qty in book['asks'])
            total = bid_depth + ask_depth
            imbalance = (bid_depth - ask_depth) / total if total > 0 else 0

            return {
                'spread_pct': spread_pct,
                'bid_depth': bid_depth,
                'ask_depth': ask_depth,
                'imbalance': imbalance,
                'best_bid': best_bid,
                'best_ask': best_ask,
                'bid_vol': bid_vol,
                'ask_vol': ask_vol,
            }
        except Exception:
            return None

    # =========================================================================
    # SYNC METHODS (unchanged — used by exit checks, trims, etc.)
    # =========================================================================

    # ─── Derivatives Data (Funding Rates, Open Interest) ─────────────────

    def fetch_funding_rate(self, symbol, exchange_name):
        """
        Fetch the current perpetual funding rate for a symbol.

        Returns:
            Funding rate as a float (e.g., 0.0003 = 0.03%) or None if unavailable.
        """
        try:
            swap_ex = self.swap_exchanges.get(exchange_name)
            if not swap_ex:
                return None

            # CCXT expects the swap symbol format (e.g., BTC/USDT:USDT)
            swap_symbol = symbol.replace('/USDT', '/USDT:USDT')
            result = swap_ex.fetch_funding_rate(swap_symbol)
            return result.get('fundingRate')
        except Exception:
            return None

    def fetch_open_interest(self, symbol, exchange_name):
        """
        Fetch the current open interest for a symbol's perpetual contract.

        Returns:
            Open interest value (USDT notional) or None if unavailable.
        """
        try:
            swap_ex = self.swap_exchanges.get(exchange_name)
            if not swap_ex:
                return None

            swap_symbol = symbol.replace('/USDT', '/USDT:USDT')
            result = swap_ex.fetch_open_interest(swap_symbol)
            return result.get('openInterestAmount') or result.get('openInterest')
        except Exception:
            return None

    # ─── Order Book Microstructure ───────────────────────────────────────

    def fetch_order_book_summary(self, symbol, exchange_name, depth=20):
        """
        Fetch top-N levels of the order book and compute microstructure metrics.

        Returns:
            dict with:
              spread_pct  — bid-ask spread as percentage of mid-price
              bid_depth   — total USDT liquidity in top bids
              ask_depth   — total USDT liquidity in top asks
              imbalance   — (bid_depth - ask_depth) / (bid_depth + ask_depth)
              best_bid    — best bid price
              best_ask    — best ask price
              bid_vol     — volume at best bid
              ask_vol     — volume at best ask
            or None if error.
        """
        try:
            ex = self.exchanges.get(exchange_name)
            if not ex:
                return None

            book = ex.fetch_order_book(symbol, limit=depth)

            if not book['bids'] or not book['asks']:
                return None

            best_bid = book['bids'][0][0]
            best_ask = book['asks'][0][0]
            bid_vol  = book['bids'][0][1]
            ask_vol  = book['asks'][0][1]
            mid = (best_bid + best_ask) / 2
            spread_pct = ((best_ask - best_bid) / mid) * 100 if mid > 0 else 0

            bid_depth = sum(price * qty for price, qty in book['bids'])
            ask_depth = sum(price * qty for price, qty in book['asks'])
            total = bid_depth + ask_depth
            imbalance = (bid_depth - ask_depth) / total if total > 0 else 0

            return {
                'spread_pct': spread_pct,
                'bid_depth': bid_depth,
                'ask_depth': ask_depth,
                'imbalance': imbalance,
                'best_bid': best_bid,
                'best_ask': best_ask,
                'bid_vol': bid_vol,
                'ask_vol': ask_vol,
            }
        except Exception:
            return None

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

                # BLACKLIST: Filter out Leveraged Tokens (3L, 5L, UP, DOWN, BEAR, BULL)
                base = normalized_symbol.split('/')[0]
                if any(base.endswith(s) for s in ['3L', '3S', '5L', '5S', 'UP', 'DOWN', 'BEAR', 'BULL']):
                    continue
                if 'ETF' in base:  # Some exchanges label them as ETF
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
