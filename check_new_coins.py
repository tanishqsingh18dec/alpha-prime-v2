#!/usr/bin/env python3
"""
Check if we're fetching the coins from the user's NEW screenshot.
"""

from multi_exchange import MultiExchangeScanner

# Coins from the NEW screenshot (top 12)
new_screenshot_coins = [
    'RIVER',
    'KITE',
    'STABLE',
    'POL',  # Polygon (prev. MATIC)
    'VIRTUAL',  # Virtuals Protocol
    'SEI',
    'ENA',  # Ethena
    'TON',  # Toncoin
    'NIGHT',  # Midnight
    'WLFI',  # World Liberty Financial
    'PENGU',  # Pudgy Penguins
    'TAO'  # Bittensor
]

print("="*80)
print("🔍 CHECKING NEW SCREENSHOT COINS")
print("="*80)

# Initialize scanner
scanner = MultiExchangeScanner(['binance', 'kucoin', 'gateio', 'mexc'])

# Get all viable coins with $1M filter
viable_coins = scanner.get_all_viable_coins(min_volume=1_000_000)

print(f"\n✅ Total coins scanned: {len(viable_coins)}")
print(f"📋 Checking {len(new_screenshot_coins)} coins from NEW screenshot...\n")

found_count = 0
missing_coins = []

for base in new_screenshot_coins:
    matches = [c for c in viable_coins if c['base_currency'] == base]
    
    if matches:
        c = matches[0]
        found_count += 1
        print(f"✅ {base:10} | {c['exchange'].upper():8} | ${c['price']:.4f} | ${c['volume']/1_000_000:.1f}M vol")
    else:
        missing_coins.append(base)
        print(f"❌ {base:10} | NOT FOUND with $1M filter")

print(f"\n📊 SUMMARY: Found {found_count}/{len(new_screenshot_coins)} coins ({found_count*100//len(new_screenshot_coins)}%)")

# Check missing coins with lower filter
if missing_coins:
    print("\n" + "="*80)
    print("🔬 CHECKING MISSING COINS WITH $500K FILTER")
    print("="*80)
    
    viable_low = scanner.get_all_viable_coins(min_volume=500_000)
    print(f"\n✅ Total coins with $500k+ volume: {len(viable_low)}")
    print(f"📋 Re-checking {len(missing_coins)} missing coins...\n")
    
    for base in missing_coins:
        matches = [c for c in viable_low if c['base_currency'] == base]
        
        if matches:
            c = matches[0]
            print(f"🟡 {base:10} | {c['exchange'].upper():8} | ${c['price']:.4f} | ${c['volume']/1_000_000:.1f}M vol | BELOW $1M")
        else:
            print(f"❌ {base:10} | NOT ON ANY EXCHANGE")

print("\n" + "="*80)
