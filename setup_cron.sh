#!/bin/bash
# =============================================================================
# setup_cron.sh — Automates daily Reddit sentiment scraping for Alpha Prime
# Run this ONCE: bash setup_cron.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$SCRIPT_DIR/venv/bin/python3"
SCRAPER="$SCRIPT_DIR/reddit_sentiment.py"
LOG_FILE="$SCRIPT_DIR/sentiment_data/cron.log"

echo "======================================"
echo "  Alpha Prime — Cron Setup"
echo "======================================"
echo "Script dir : $SCRIPT_DIR"
echo "Python     : $PYTHON"
echo "Scraper    : $SCRAPER"
echo "Log file   : $LOG_FILE"
echo ""

# Verify python and scraper exist
if [ ! -f "$PYTHON" ]; then
    echo "❌ Python not found at $PYTHON"
    echo "   Make sure your venv is set up: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

if [ ! -f "$SCRAPER" ]; then
    echo "❌ Scraper not found at $SCRAPER"
    exit 1
fi

# The cron job line: runs at 6:00 AM every day
# Format: minute hour day month weekday command
CRON_JOB="0 6 * * * $PYTHON $SCRAPER >> $LOG_FILE 2>&1"

# Check if cron job already exists
EXISTING=$(crontab -l 2>/dev/null | grep -F "$SCRAPER")

if [ -n "$EXISTING" ]; then
    echo "✅ Cron job already exists:"
    echo "   $EXISTING"
    echo ""
    echo "No changes made."
else
    # Add the new cron job
    (crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -
    echo "✅ Cron job added successfully!"
    echo ""
    echo "   Schedule: Every day at 6:00 AM"
    echo "   Command : $CRON_JOB"
    echo "   Logs    : $LOG_FILE"
fi

echo ""
echo "======================================"
echo "  Current crontab:"
echo "======================================"
crontab -l
echo ""
echo "Done. The scraper will now run automatically every morning at 6 AM."
echo "To check logs: tail -f $LOG_FILE"
