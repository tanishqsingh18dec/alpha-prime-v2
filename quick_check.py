#!/usr/bin/env python3
"""Quick check of coin count with $1M filter"""

from multi_exchange import MultiExchangeScanner

scanner = MultiExchangeScanner(['binance', 'kucoin', 'gateio', 'mexc'])
viable_coins = scanner.get_all_viable_coins(min_volume=1_000_000)

print(f"\n{'='*80}")
print(f"📊 COIN COVERAGE WITH $1M FILTER")
print(f"{'='*80}")
print(f"\n✅ Total unique coins: {len(viable_coins)}")

# Breakdown by exchange
exchange_counts = {}
for coin in viable_coins:
    ex = coin['exchange']
    exchange_counts[ex] = exchange_counts.get(ex, 0) + 1

print(f"\nBreakdown by exchange:")
for exchange_name, count in sorted(exchange_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"   {exchange_name.upper():8}: {count} coins")

# Check screenshot coins
screenshot_coins = ['PIPPIN', 'RIVER', 'ASTER', 'H', 'ZRO', 'XDC', 'NEXO', 'QNT', 'SKY', 'NIGHT', 'HBAR']
found = [coin for coin in screenshot_coins if coin in [c['base_currency'] for c in viable_coins]]
print(f"\n✅ Screenshot coins found: {len(found)}/{len(screenshot_coins)}")
print(f"   {', '.join(found)}")

missing = [coin for coin in screenshot_coins if coin not in [c['base_currency'] for c in viable_coins]]
if missing:
    print(f"❌ Still missing: {', '.join(missing)}")

print(f"{'='*80}\n")
