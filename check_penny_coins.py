#!/usr/bin/env python3
"""
Check penny/micro-cap coins from trending 24h screenshot.
"""

from multi_exchange import MultiExchangeScanner

# Coins from trending 24h screenshot (penny/micro-caps)
trending_coins = [
    ('hoodrat', 'HOODRAT'),
    ('MooNutPeng', 'The Fastest Runner'),
    ('BUIDL', '$BUIDL DeFiUS'),
    ('HTC', 'OPTIMISTic Engine HTC'),
    ('YI', 'Yi'),
    ('BTR', 'Bitlayer'),
    ('MEga', 'MEGA_ETH_COIN'),
    ('TAKE', 'OVERTAKE'),
    ('ME', 'Magic Eden'),
    ('SIREN', 'siren'),
]

print("="*80)
print("🔍 CHECKING PENNY/MICRO-CAP COINS FROM TRENDING 24H")
print("="*80)

scanner = MultiExchangeScanner(['binance', 'kucoin', 'gateio', 'mexc'])

# Check with $1M filter (current setting)
print("\n📌 WITH CURRENT $1M FILTER:")
viable_1m = scanner.get_all_viable_coins(min_volume=1_000_000)
print(f"Total coins: {len(viable_1m)}\n")

found_1m = 0
missing_1m = []

for symbol, name in trending_coins:
    matches = [c for c in viable_1m if c['base_currency'].upper() == symbol.upper()]
    if matches:
        c = matches[0]
        found_1m += 1
        print(f"✅ {symbol:12} ({name[:20]:20}) | {c['exchange'].upper():8} | ${c['price']:.6f} | ${c['volume']/1_000_000:.1f}M")
    else:
        missing_1m.append((symbol, name))
        print(f"❌ {symbol:12} ({name[:20]:20}) | NOT FOUND")

print(f"\n📊 Found {found_1m}/{len(trending_coins)} coins with $1M filter ({found_1m*100//len(trending_coins)}%)")

# Check with lower filters
if missing_1m:
    print("\n" + "="*80)
    print("🔬 CHECKING MISSING COINS WITH $500K FILTER:")
    print("="*80)
    
    viable_500k = scanner.get_all_viable_coins(min_volume=500_000)
    print(f"Total coins: {len(viable_500k)}\n")
    
    found_500k = 0
    still_missing = []
    
    for symbol, name in missing_1m:
        matches = [c for c in viable_500k if c['base_currency'].upper() == symbol.upper()]
        if matches:
            c = matches[0]
            found_500k += 1
            print(f"🟡 {symbol:12} ({name[:20]:20}) | {c['exchange'].upper():8} | ${c['price']:.6f} | ${c['volume']/1_000_000:.1f}M")
        else:
            still_missing.append((symbol, name))
    
    if still_missing:
        print("\n" + "="*80)
        print("🔬 CHECKING WITH $100K FILTER (VERY RISKY):")
        print("="*80)
        
        viable_100k = scanner.get_all_viable_coins(min_volume=100_000)
        print(f"Total coins: {len(viable_100k)}\n")
        
        for symbol, name in still_missing:
            matches = [c for c in viable_100k if c['base_currency'].upper() == symbol.upper()]
            if matches:
                c = matches[0]
                print(f"🔴 {symbol:12} ({name[:20]:20}) | {c['exchange'].upper():8} | ${c['price']:.6f} | ${c['volume']/1_000_000:.2f}M | VERY LOW LIQ")
            else:
                print(f"❌ {symbol:12} ({name[:20]:20}) | NOT ON ANY EXCHANGE or <$100k volume")

print("\n" + "="*80)
print("⚠️  RISK WARNING: Penny coins with <$1M volume are EXTREMELY risky!")
print("="*80)
