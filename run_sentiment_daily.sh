#!/bin/bash
# =============================================================================
# run_sentiment_daily.sh
# Smart daily runner for Reddit Sentiment Scraper (Alpha Prime v2)
#
# Strategy:
#   - Called by launchd every 30 minutes
#   - Checks if today's run has already happened  → skip if yes
#   - Checks internet connectivity               → skip if offline
#   - Runs the scraper exactly once per day      → whenever laptop is first on
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$SCRIPT_DIR/venv/bin/python3"
SCRAPER="$SCRIPT_DIR/reddit_sentiment.py"
LOG_FILE="$SCRIPT_DIR/sentiment_data/cron.log"
LAST_RUN_FILE="$SCRIPT_DIR/sentiment_data/.last_run_date"

TODAY=$(date +%Y-%m-%d)

# ── Ensure log directory exists ───────────────────────────────────────────────
mkdir -p "$SCRIPT_DIR/sentiment_data"

# ── Log helper ────────────────────────────────────────────────────────────────
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# ── 1. Already ran today? ─────────────────────────────────────────────────────
if [ -f "$LAST_RUN_FILE" ]; then
    LAST_RUN=$(cat "$LAST_RUN_FILE")
    if [ "$LAST_RUN" = "$TODAY" ]; then
        log "✅ Already ran today ($TODAY). Skipping."
        exit 0
    fi
fi

# ── 2. Internet connectivity check ────────────────────────────────────────────
# Try reaching Reddit — if it fails, we're offline; try again next poll
if ! curl -s --max-time 5 --head "https://www.reddit.com" > /dev/null 2>&1; then
    log "🌐 No internet connection. Will retry next interval."
    exit 0
fi

# ── 3. Run the scraper ────────────────────────────────────────────────────────
log "🚀 Starting Reddit sentiment scrape for $TODAY ..."

cd "$SCRIPT_DIR" || exit 1
"$PYTHON" "$SCRAPER" >> "$LOG_FILE" 2>&1
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    # Mark today as done so we don't run again until tomorrow
    echo "$TODAY" > "$LAST_RUN_FILE"
    log "✅ Scrape completed successfully for $TODAY."
else
    log "❌ Scrape FAILED (exit code $EXIT_CODE). Will retry next interval."
    # Note: we do NOT write LAST_RUN_FILE on failure so it retries next poll
fi

exit $EXIT_CODE
