#!/usr/bin/env python3
"""
Test script for multi-exchange integration.
Verifies that all exchanges connect and coins are scanned correctly.
"""

from multi_exchange import MultiExchangeScanner

print("="*80)
print("🧪 MULTI-EXCHANGE INTEGRATION TEST")
print("="*80)

# Test 1: Initialize scanner
print("\n📌 TEST 1: Initializing multi-exchange scanner...")
scanner = MultiExchangeScanner(['binance', 'kucoin', 'gateio', 'mexc'])
print(f"✅ Scanner initialized with {len(scanner.exchanges)} exchanges\n")

# Test 2: Scan all exchanges
print("📌 TEST 2: Scanning all exchanges for viable coins ($2M+ volume)...")
viable_coins = scanner.get_all_viable_coins(min_volume=2_000_000)
print(f"✅ Total coins found: {len(viable_coins)}\n")

# Test 3: Breakdown by exchange
print("📌 TEST 3: Breakdown by exchange:")
exchange_counts = {}
for coin in viable_coins:
    ex = coin['exchange']
    exchange_counts[ex] = exchange_counts.get(ex, 0) + 1

for exchange_name, count in sorted(exchange_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"   {exchange_name.upper()}: {count} coins")

# Test 4: Show top 10 coins by volume
print("\n📌 TEST 4: Top 10 coins by volume:")
sorted_coins = sorted(viable_coins, key=lambda x: x['volume'], reverse=True)[:10]
for i, coin in enumerate(sorted_coins, 1):
    print(f"   {i}. {coin['base_currency']:8} | {coin['exchange']:8} | ${coin['volume']/1_000_000:.1f}M volume | ${coin['price']:.4f}")

# Test 5: Check specific coins
print("\n📌 TEST 5: Checking specific coins...")
test_coins = ['BTC', 'ETH', 'ASTER', 'ZRO']
for base in test_coins:
    found = [c for c in viable_coins if c['base_currency'] == base]
    if found:
        c = found[0]
        print(f"   ✅ {base:6} found on {c['exchange'].upper()} | ${c['price']:.4f} | ${c['volume']/1_000_000:.1f}M volume")
    else:
        print(f"   ❌ {base} not found")

# Test 6: Price fetching
print("\n📌 TEST 6: Testing price fetching from specific exchanges...")
test_pairs = [
    ('BTC/USDT', 'binance'),
    ('ETH/USDT', 'kucoin'),
]
for symbol, exchange_name in test_pairs:
    if exchange_name in scanner.exchanges:
        price = scanner.get_current_price(symbol, exchange_name)
        if price:
            print(f"   ✅ {symbol:12} on {exchange_name.upper():8} = ${price:,.2f}")
        else:
            print(f"   ⚠️  {symbol} price fetch failed on {exchange_name}")

print("\n" + "="*80)
print("🎉 ALL TESTS COMPLETED!")
print("="*80)
