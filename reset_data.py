import os
import json

def reset_data():
    files_to_remove = [
        "alpha_prime_portfolio.json",
        "alpha_prime_trades.csv",
        "equity_curve.csv",
        "alpha_prime_events.jsonl"
    ]
    
    print("🧹 Cleaning up Alpha Prime v2 data...")
    
    for file in files_to_remove:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"✅ Deleted {file}")
            except Exception as e:
                print(f"❌ Error deleting {file}: {e}")
        else:
            print(f"ℹ️  {file} not found (already clean)")
            
    print("\n✨ Data reset complete. Ready for fresh start at $1000.00.")

if __name__ == "__main__":
    confirm = input("Are you sure you want to DELETE all trading history and reset to $1000? (y/n): ")
    if confirm.lower() == 'y':
        reset_data()
    else:
        print("❌ Reset cancelled.")
