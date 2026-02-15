import ccxt

exchange = ccxt.binance({'enableRateLimit': True})
markets = exchange.load_markets()
tickers = exchange.fetch_tickers()

# Coins from screenshot
coins_to_check = ['PIPPIN', 'RIVER', 'ASTER', 'H', 'LEO', 'ZRO', 'NEXO', 'XDC', 'QNT']

print("="*80)
print("CHECKING COINS FROM YOUR LIST")
print("="*80)

for coin in coins_to_check:
    symbol = f"{coin}/USDT"
    
    if symbol in markets:
        if symbol in tickers:
            volume = tickers[symbol].get('quoteVolume', 0)
            price = tickers[symbol].get('last', 0)
            active = markets[symbol]['active']
            
            status = "✅ ON RADAR" if (volume >= 2_000_000 and active) else "❌ TOO LOW VOLUME"
            
            print(f"\n{coin}:")
            print(f"  Symbol: {symbol}")
            print(f"  Price: ${price:.4f}")
            print(f"  Volume (24h): ${volume:,.0f}")
            print(f"  Status: {status}")
        else:
            print(f"\n{coin}: ⚠️  Exists but no ticker data")
    else:
        print(f"\n{coin}: ❌ NOT AVAILABLE on Binance USDT pairs")

print("\n" + "="*80)
