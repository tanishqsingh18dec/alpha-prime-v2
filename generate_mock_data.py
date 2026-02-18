
import pandas as pd
import numpy as np
import json
import random
from datetime import datetime, timedelta
import os

# Configuration
DAYS_HISTORY = 30 # Generate 30 days of 4H data for equity curve? Or just 100 points?
# Dashboard expects up to 100 points.
# Equity curve format: timestamp,portfolio_value,btc_benchmark

def generate_equity_curve():
    print("Generating equity curve...")
    start_value = 100.0
    btc_start = 50000.0
    
    points = []
    current_val = start_value
    current_btc = btc_start
    
    # Generate 100 points (e.g., last 4 days hourly)
    start_time = datetime.now() - timedelta(hours=100)
    
    for i in range(101):
        timestamp = (start_time + timedelta(hours=i)).isoformat()
        
        # Random walk
        val_change = np.random.normal(0, 0.01) # 1% std dev
        btc_change = np.random.normal(0, 0.015) 
        
        current_val *= (1 + val_change)
        current_btc *= (1 + btc_change)
        
        # Normalize BTC to 100 base
        btc_normalized = 100 * (current_btc / btc_start)
        
        points.append({
            'timestamp': timestamp,
            'portfolio_value': round(current_val, 2),
            'btc_benchmark': round(btc_normalized, 2)
        })
    
    df = pd.DataFrame(points)
    df.to_csv('equity_curve.csv', index=False)
    print(f"✅ Generated {len(points)} equity points")

def generate_portfolio():
    print("Generating portfolio...")
    portfolio = {
        "balance": 45.30,
        "positions": {
            "BTC/USDT": {
                "symbol": "BTC/USDT", "exchange": "binance", "entry_price": 65000, "quantity": 0.0005, 
                "current_price": 67000, "current_value": 33.5, "unrealized_pnl": 1.0, 
                "entry_time": datetime.now().isoformat(), "cycles_held": 5
            },
            "ETH/USDT": {
                "symbol": "ETH/USDT", "exchange": "kucoin", "entry_price": 3000, "quantity": 0.01, 
                "current_price": 3100, "current_value": 31.0, "unrealized_pnl": 1.0, 
                "entry_time": datetime.now().isoformat(), "cycles_held": 2
            }
        },
        "total_trades": 12,
        "winning_trades": 8,
        "total_pnl": 15.40,
        "last_updated": datetime.now().isoformat()
    }
    
    with open('alpha_prime_portfolio.json', 'w') as f:
        json.dump(portfolio, f, indent=2)
    print("✅ Generated portfolio")

def generate_logs():
    print("Generating logs...")
    events = []
    types = ['info', 'buy', 'sell', 'regime', 'signal']
    
    start_time = datetime.now() - timedelta(hours=5)
    
    for i in range(20):
        t = (start_time + timedelta(minutes=i*15)).isoformat()
        evt_type = random.choice(types)
        msg = f"Simulated event {i}"
        
        if evt_type == 'buy':
            msg = f"BUY ETH/USDT on KUCOIN: 0.01 @ $3000"
        elif evt_type == 'sell':
            msg = f"SELL SOL/USDT on BINANCE: P&L +$2.50"
        elif evt_type == 'regime':
            msg = f"Regime change: RISK_ON"
            
        event = {
            "id": i+1,
            "timestamp": t,
            "type": evt_type,
            "message": msg,
            "details": {}
        }
        events.append(event)
        
    with open('alpha_prime_events.jsonl', 'w') as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    print("✅ Generated logs")

if __name__ == "__main__":
    generate_equity_curve()
    generate_portfolio()
    generate_logs()
