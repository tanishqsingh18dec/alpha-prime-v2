#!/usr/bin/env python3
"""
Check most visited 7d coins from latest screenshot.
"""

from multi_exchange import MultiExchangeScanner

# Coins from "Most Visited 7d" screenshot
most_visited_coins = [
    'PIPPIN',
    'OWB',
    'BERA',     # Berachain
    'BNKR',     # BankrCoin
    'SIREN',
    'WAR',
    'ASTER',
    'RIVER',
    'KITE',
    'ZRO',      # LayerZero
    'ZBCN',     # Zebec Network
    'XDC',
    'NIGHT',    # Midnight
]

print("="*80)
print("🔍 CHECKING 'MOST VISITED 7D' COINS")
print("="*80)

scanner = MultiExchangeScanner(['binance', 'kucoin', 'gateio', 'mexc'])
viable_coins = scanner.get_all_viable_coins(min_volume=1_000_000)

print(f"\n✅ Total coins scanned with $1M filter: {len(viable_coins)}")
print(f"📋 Checking {len(most_visited_coins)} coins from screenshot...\n")

found_count = 0
missing = []

for coin in most_visited_coins:
    matches = [c for c in viable_coins if c['base_currency'].upper() == coin.upper()]
    
    if matches:
        c = matches[0]
        found_count += 1
        print(f"✅ {coin:10} | {c['exchange'].upper():8} | ${c['price']:.6f} | ${c['volume']/1_000_000:.1f}M vol")
    else:
        missing.append(coin)
        print(f"❌ {coin:10} | NOT FOUND with $1M filter")

print(f"\n📊 SUMMARY: Found {found_count}/{len(most_visited_coins)} coins ({found_count*100//len(most_visited_coins)}%)")

# Check missing with lower filter
if missing:
    print("\n" + "="*80)
    print(f"🔬 CHECKING {len(missing)} MISSING COINS WITH $500K FILTER")
    print("="*80)
    
    viable_500k = scanner.get_all_viable_coins(min_volume=500_000)
    print(f"\nTotal coins with $500k+ volume: {len(viable_500k)}\n")
    
    for coin in missing:
        matches = [c for c in viable_500k if c['base_currency'].upper() == coin.upper()]
        
        if matches:
            c = matches[0]
            print(f"🟡 {coin:10} | {c['exchange'].upper():8} | ${c['price']:.6f} | ${c['volume']/1_000_000:.2f}M vol | BELOW $1M")
        else:
            print(f"❌ {coin:10} | NOT ON ANY EXCHANGE or <$500k volume")

print("\n" + "="*80)
