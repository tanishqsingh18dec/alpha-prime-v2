#!/usr/bin/env python3
"""
Reddit Sentiment Scraper for Alpha Prime v2
============================================
Fetches top 15 posts (from TODAY) + top 20 comments per post
from all 50 major crypto subreddits.

ENHANCED: Now scans ALL text for ANY coin mention, scores each mention
in context, and computes:
  - Per-coin verdict: SHOOT_UP / SHOOT_DOWN / WATCH / LOW_DATA
  - Holdings Correlation: aligns Reddit signal against your live portfolio

Two layers of scoring:
  1. Subreddit Score: sentiment of the whole post (existing logic)
  2. Mention Score:  per-coin context scoring across ALL subreddits

Output:
  reddit_sentiment_YYYY-MM-DD.csv  (raw per-post data)
  reddit_sentiment_latest.json     (rich per-coin signals + holdings correlation)

Run daily via cron or manually:
    python3 reddit_sentiment.py
"""

import requests
import time
import json
import os
import re
from datetime import datetime, timezone
from collections import defaultdict
import pandas as pd

# ─── CONFIG ──────────────────────────────────────────────────────────────────

POSTS_PER_SUB     = 15    # Top N posts per subreddit (today only)
COMMENTS_PER_POST = 20    # Top N comments per post
SLEEP_BETWEEN_CALLS = 1.5 # Seconds between API calls (avoid rate limit)

VERDICT_MIN_MENTIONS = 5   # Need at least this many mentions for a verdict
CONFIDENCE_HIGH      = 20  # Mentions >= this → HIGH confidence
CONFIDENCE_MEDIUM    = 10  # Mentions >= this → MEDIUM confidence

PORTFOLIO_FILE = "alpha_prime_portfolio.json"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/120.0.0.0 Safari/537.36 crypto-sentiment-bot/2.0'
}

OUTPUT_DIR = "sentiment_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── SUBREDDIT → PRIMARY COIN MAPPING ────────────────────────────────────────

SUBREDDITS = [
    # (subreddit_name, coin_symbol)
    ("CryptoCurrency",       "MARKET"),   # General market sentiment
    ("Bitcoin",              "BTC"),
    ("ethereum",             "ETH"),
    ("ethtrader",            "ETH"),
    ("dogecoin",             "DOGE"),
    ("CryptoMoonShots",      "MARKET"),
    ("NFT",                  "MARKET"),
    ("CryptoTechnology",     "MARKET"),
    ("btc",                  "BTC"),
    ("BitcoinBeginners",     "BTC"),
    ("binance",              "BNB"),
    ("cardano",              "ADA"),
    ("SHIBArmy",             "SHIB"),
    ("Ripple",               "XRP"),
    ("litecoin",             "LTC"),
    ("cryptocurrencymemes",  "MARKET"),
    ("SafeMoon",             "SAFEMOON"),
    ("Monero",               "XMR"),
    ("NFTsMarketplace",      "MARKET"),
    ("CoinBase",             "MARKET"),
    ("altcoin",              "MARKET"),
    ("Crypto_com",           "CRO"),
    ("opensea",              "MARKET"),
    ("CryptoCurrencyTrading","MARKET"),
    ("solana",               "SOL"),
    ("Shibainucoin",         "SHIB"),
    ("Iota",                 "MIOTA"),
    ("tronix",               "TRX"),
    ("nanocurrency",         "XNO"),
    ("CryptoMoon",           "MARKET"),
    ("defi",                 "MARKET"),
    ("AxieInfinity",         "AXS"),
    ("ledgerwallet",         "MARKET"),
    ("loopringorg",          "LRC"),
    ("BATProject",           "BAT"),
    ("decentraland",         "MANA"),
    ("terraluna",            "LUNA"),
    ("cosmosnetwork",        "ATOM"),
    ("algorand",             "ALGO"),
    ("Crypto_General",       "MARKET"),
    ("tezos",                "XTZ"),
    ("UniSwap",              "UNI"),
    ("0xPolygon",            "MATIC"),
    ("ico",                  "MARKET"),
    ("TREZOR",               "MARKET"),
    ("CelsiusNetwork",       "CEL"),
    ("StepN",                "GMT"),
    ("FantomFoundation",     "FTM"),
    ("dot",                  "DOT"),
    ("Avax",                 "AVAX"),
]

# ─── COIN ALIAS DICTIONARY ───────────────────────────────────────────────────
# Maps name/ticker variations → canonical symbol
# Add your holdings here too for best detection

COIN_ALIASES = {
    # Major coins
    "bitcoin": "BTC",    "btc": "BTC",       "$btc": "BTC",
    "ethereum": "ETH",   "eth": "ETH",       "$eth": "ETH",   "ether": "ETH",
    "solana": "SOL",     "sol": "SOL",        "$sol": "SOL",
    "ripple": "XRP",     "xrp": "XRP",        "$xrp": "XRP",
    "cardano": "ADA",    "ada": "ADA",        "$ada": "ADA",
    "dogecoin": "DOGE",  "doge": "DOGE",      "$doge": "DOGE",
    "shib": "SHIB",      "shiba": "SHIB",     "shibainuoshi": "SHIB",
    "bnb": "BNB",        "binance coin": "BNB",
    "avax": "AVAX",      "avalanche": "AVAX",
    "matic": "MATIC",    "polygon": "MATIC",
    "dot": "DOT",        "polkadot": "DOT",
    "ltc": "LTC",        "litecoin": "LTC",
    "atom": "ATOM",      "cosmos": "ATOM",
    "uni": "UNI",        "uniswap": "UNI",
    "link": "LINK",      "chainlink": "LINK",
    "ftm": "FTM",        "fantom": "FTM",
    "algo": "ALGO",      "algorand": "ALGO",
    "luna": "LUNA",      "terra": "LUNA",
    "xmr": "XMR",        "monero": "XMR",
    "trx": "TRX",        "tron": "TRX",
    "mana": "MANA",      "decentraland": "MANA",
    "axs": "AXS",        "axie": "AXS",
    "bat": "BAT",        "basic attention": "BAT",
    "lrc": "LRC",        "loopring": "LRC",
    "cro": "CRO",        "crypto.com": "CRO",
    "gmt": "GMT",        "stepn": "GMT",
    "xtz": "XTZ",        "tezos": "XTZ",
    "cel": "CEL",        "celsius": "CEL",
    "xno": "XNO",        "nano": "XNO",
    "miota": "MIOTA",    "iota": "MIOTA",
    "pepe": "PEPE",
    "wif": "WIF",        "dogwifhat": "WIF",
    "floki": "FLOKI",
    "bonk": "BONK",
    "sui": "SUI",
    "aptos": "APT",      "apt": "APT",
    "arb": "ARB",        "arbitrum": "ARB",
    "op": "OP",          "optimism": "OP",
    "inj": "INJ",        "injective": "INJ",
    "sei": "SEI",
    "fet": "FET",        "fetch.ai": "FET",   "fetchai": "FET",
    "render": "RNDR",    "rndr": "RNDR",
    "gala": "GALA",
    "sand": "SAND",      "sandbox": "SAND",
    # NOTE: Do NOT hardcode specific holdings here.
    # Current holdings are injected dynamically at runtime — see build_coin_regex().
}


def _inject_holdings_into_aliases():
    """
    Read the live portfolio file and auto-add any held coin's base ticker
    (lowercase) → SYMBOL (uppercase) into COIN_ALIASES.
    This runs every time build_coin_regex() is called so tomorrow's holdings
    are picked up automatically without touching this file.
    """
    if not os.path.exists(PORTFOLIO_FILE):
        return
    try:
        with open(PORTFOLIO_FILE, 'r') as f:
            data = json.load(f)
        positions = data.get('positions', {})
        added = []
        for full_symbol in positions:
            # "INTER/BIT" → base = "INTER"
            base = full_symbol.split('/')[0].upper()
            key  = base.lower()
            if key not in COIN_ALIASES:
                COIN_ALIASES[key] = base
                added.append(base)
        if added:
            print(f"   🔄 Auto-added holdings to coin alias map: {', '.join(added)}")
    except Exception as e:
        print(f"   ⚠️  Could not inject holdings into alias map: {e}")


def build_coin_regex():
    """
    Inject current holdings into COIN_ALIASES, then compile the detection regex.
    Call this at the start of scrape_all() so it always reflects live holdings.
    """
    global COIN_MENTION_RE
    _inject_holdings_into_aliases()
    # Sort by length descending so longer aliases match before shorter substrings
    _sorted = sorted(COIN_ALIASES.keys(), key=len, reverse=True)
    _escaped = [re.escape(k) for k in _sorted]
    COIN_MENTION_RE = re.compile(
        r'(?<![a-zA-Z0-9])(' + '|'.join(_escaped) + r')(?![a-zA-Z0-9])',
        re.IGNORECASE
    )
    return COIN_MENTION_RE


# Build at module load time (for unit-test imports etc.)
# Will be refreshed at the start of scrape_all() with live holdings.
_sorted_aliases = sorted(COIN_ALIASES.keys(), key=len, reverse=True)
_escaped = [re.escape(k) for k in _sorted_aliases]
COIN_MENTION_RE = re.compile(
    r'(?<![a-zA-Z0-9])(' + '|'.join(_escaped) + r')(?![a-zA-Z0-9])',
    re.IGNORECASE
)

# ─── SENTIMENT SCORING: FinBERT (primary) → Keyword Fallback ─────────────────

BULLISH_WORDS = [
    "moon", "bullish", "pump", "buy", "buying", "long", "up", "surge",
    "rally", "breakout", "ath", "all time high", "undervalued", "gem",
    "accumulate", "hodl", "hold", "strong", "growth", "potential",
    "launch", "partnership", "upgrade", "adoption", "positive", "profit",
    "gains", "green", "🚀", "💎", "🟢", "📈", "fire", "🔥",
    "bullrun", "bull run", "uptrend", "accumulating", "bottom", "reversal",
    "support", "breakout", "golden cross", "buy the dip", "dip",
]

BEARISH_WORDS = [
    "dump", "crash", "sell", "selling", "short", "down", "drop", "fall",
    "bearish", "scam", "rug", "rugpull", "dead", "overvalued", "avoid",
    "warning", "fraud", "hack", "exploit", "fear", "loss", "red",
    "correction", "capitulate", "panic", "bubble", "worthless",
    "🔴", "📉", "💀", "🗑️", "dumping", "bleeding", "rekt", "exit",
    "resistance", "death cross", "breakdown", "fud", "downtrend",
]


def _keyword_score(text: str) -> float:
    """Fallback: keyword-based sentiment scoring [-1, +1]."""
    if not text:
        return 0.0
    text_lower = text.lower()
    bull = sum(1 for w in BULLISH_WORDS if w in text_lower)
    bear = sum(1 for w in BEARISH_WORDS if w in text_lower)
    total = bull + bear
    if total == 0:
        return 0.0
    return (bull - bear) / total


# ─── FinBERT Scorer (lazy-loaded) ────────────────────────────────────────────
# Uses ProsusAI/finbert — a BERT model fine-tuned on financial text.
# Falls back to keyword scoring if transformers/torch are not installed.

_finbert_pipeline = None
_finbert_available = None  # None = not checked yet

def _load_finbert():
    """Lazy-load FinBERT pipeline. Returns True if loaded, False otherwise."""
    global _finbert_pipeline, _finbert_available
    if _finbert_available is not None:
        return _finbert_available
    try:
        from transformers import pipeline as hf_pipeline
        print("   🧠 Loading FinBERT sentiment model (first run may download ~500MB)...")
        _finbert_pipeline = hf_pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            device=-1,  # CPU (use 0 for GPU)
            truncation=True,
            max_length=512,
        )
        _finbert_available = True
        print("   ✅ FinBERT loaded — using ML sentiment scoring")
        return True
    except Exception as e:
        _finbert_available = False
        print(f"   ⚠️  FinBERT not available ({e}) — using keyword fallback")
        return False


def score_text(text: str) -> float:
    """
    Returns a sentiment score between -1.0 (very bearish) and +1.0 (very bullish).

    Uses FinBERT (ProsusAI/finbert) if available, otherwise falls back to keyword counting.
    FinBERT is ~3-5x more accurate on financial text but requires transformers + torch.
    """
    if not text:
        return 0.0

    # Try FinBERT first
    if _load_finbert() and _finbert_pipeline:
        try:
            # Truncate to avoid token limits
            truncated = text[:1024]
            result = _finbert_pipeline(truncated)[0]
            label = result['label'].lower()
            confidence = result['score']

            if label == 'positive':
                return confidence        # e.g., +0.85
            elif label == 'negative':
                return -confidence       # e.g., -0.90
            else:  # neutral
                return 0.0
        except Exception:
            pass  # Fall through to keyword scoring

    # Fallback: keyword scoring
    return _keyword_score(text)


# ─── COIN MENTION EXTRACTION ─────────────────────────────────────────────────

def extract_coin_mentions(text: str) -> list:
    """
    Scan any text and return list of (canonical_symbol, context_window) tuples.
    Context window = 120 chars around the match for sentiment scoring.
    """
    if not text:
        return []

    mentions = []
    for match in COIN_MENTION_RE.finditer(text):
        raw = match.group(0).lower()
        symbol = COIN_ALIASES.get(raw)
        if not symbol:
            continue

        # Extract context window (60 chars each side)
        start = max(0, match.start() - 60)
        end   = min(len(text), match.end() + 60)
        context = text[start:end]

        mentions.append((symbol, context))

    return mentions


def score_mention_in_context(symbol: str, context: str) -> float:
    """
    Score a specific coin mention using the surrounding context words.
    Returns: +1.0 (very bullish) ... -1.0 (very bearish)
    """
    return score_text(context)


# ─── REDDIT FETCHERS ─────────────────────────────────────────────────────────

def fetch_today_posts(subreddit: str, limit: int = POSTS_PER_SUB):
    """
    Fetch top posts from TODAY only using the 'new' sort + date filter.
    Falls back to 'hot' if new returns nothing from today.
    """
    today_start = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    ).timestamp()

    posts = []

    for sort in ['new', 'hot']:
        url = f"https://www.reddit.com/r/{subreddit}/{sort}.json?limit=50"
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            if r.status_code == 429:
                print(f"   ⚠️  Rate limited on r/{subreddit}, sleeping 10s...")
                time.sleep(10)
                continue
            if r.status_code != 200:
                continue

            children = r.json()['data']['children']
            for child in children:
                d = child['data']
                if d.get('created_utc', 0) >= today_start:
                    posts.append({
                        'post_id':      d.get('id', ''),
                        'title':        d.get('title', ''),
                        'body':         d.get('selftext', ''),
                        'score':        d.get('score', 0),
                        'upvote_ratio': d.get('upvote_ratio', 0.5),
                        'num_comments': d.get('num_comments', 0),
                        'created_utc':  d.get('created_utc', 0),
                        'url':          d.get('url', ''),
                    })

            if posts:
                break  # Got today's posts, no need for fallback

        except Exception as e:
            print(f"   ⚠️  Error fetching r/{subreddit}: {e}")

        time.sleep(SLEEP_BETWEEN_CALLS)

    # Sort by score (upvotes) and take top N
    posts.sort(key=lambda x: x['score'], reverse=True)
    return posts[:limit]


def fetch_comments(subreddit: str, post_id: str, limit: int = COMMENTS_PER_POST):
    """Fetch top comments for a specific post."""
    url = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.json?limit={limit}&sort=top"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 429:
            print(f"   ⚠️  Rate limited fetching comments, sleeping 10s...")
            time.sleep(10)
            return []
        if r.status_code != 200:
            return []

        data = r.json()
        if len(data) < 2:
            return []

        comments = data[1]['data']['children']
        return [
            c['data'].get('body', '')
            for c in comments
            if c.get('kind') == 't1' and c['data'].get('body', '') not in ['[deleted]', '[removed]', '']
        ][:limit]

    except Exception as e:
        print(f"   ⚠️  Error fetching comments for {post_id}: {e}")
        return []


# ─── HOLDINGS CORRELATION ─────────────────────────────────────────────────────

def load_current_holdings() -> dict:
    """Load current positions from the bot's portfolio file."""
    if not os.path.exists(PORTFOLIO_FILE):
        return {}
    try:
        with open(PORTFOLIO_FILE, 'r') as f:
            data = json.load(f)
        return data.get('positions', {})
    except Exception as e:
        print(f"   ⚠️  Could not load portfolio: {e}")
        return {}


def compute_holdings_correlation(coin_data: dict, holdings: dict) -> dict:
    """
    For each position in the portfolio, look up its sentiment verdict and
    produce a plain-English recommendation.

    coin_data: the final per-coin sentiment dict (keyed by symbol like "SOL")
    holdings:  dict of {symbol: position_dict} e.g. {"INTER/BIT": {...}}
    """
    correlation = {}

    for full_symbol, position in holdings.items():
        # Strip exchange suffix: "INTER/BIT" → "INTER", "AIX/MEX" → "AIX"
        base = full_symbol.split('/')[0].upper()

        sentiment_entry = coin_data.get(base, {})
        verdict     = sentiment_entry.get('verdict', 'LOW_DATA')
        confidence  = sentiment_entry.get('confidence', 'LOW')
        sentiment   = sentiment_entry.get('sentiment', 0.0)
        mention_cnt = sentiment_entry.get('mention_count', 0)
        mentions_labeled = mention_cnt

        unrealized_pnl = position.get('unrealized_pnl', 0)
        side = 'LONG'  # Bot is long-only

        # Build recommendation
        if verdict == 'SHOOT_UP' and confidence in ('HIGH', 'MEDIUM'):
            rec = "HOLD — Reddit strongly agrees 🟢"
        elif verdict == 'SHOOT_UP' and confidence == 'LOW':
            rec = "HOLD — weak bullish signal 🟡"
        elif verdict == 'SHOOT_DOWN' and confidence in ('HIGH', 'MEDIUM'):
            rec = "CAUTION — Reddit bearish on this coin 🔴"
        elif verdict == 'SHOOT_DOWN' and confidence == 'LOW':
            rec = "WATCH — slight bearish signal 🟡"
        elif verdict == 'WATCH':
            rec = "NEUTRAL — no strong Reddit signal ⚪"
        else:
            rec = "LOW DATA — not enough Reddit mentions ⚪"

        correlation[full_symbol] = {
            'side':           side,
            'unrealized_pnl': round(unrealized_pnl, 4),
            'reddit_verdict': verdict,
            'confidence':     confidence,
            'sentiment_score': round(sentiment, 4),
            'mention_count':  mentions_labeled,
            'recommendation': rec,
        }

    return correlation


def compute_market_overall(coin_data: dict) -> str:
    """Compute overall market sentiment from the MARKET bucket and BTC/ETH."""
    scores = []
    for key in ['MARKET', 'BTC', 'ETH']:
        if key in coin_data:
            scores.append(coin_data[key].get('sentiment', 0.0))
    if not scores:
        return 'NEUTRAL'
    avg = sum(scores) / len(scores)
    if avg > 0.1:
        return 'BULLISH'
    elif avg < -0.1:
        return 'BEARISH'
    return 'NEUTRAL'


# ─── VERDICT ENGINE ──────────────────────────────────────────────────────────

def compute_verdict(mention_count: int, bull: int, bear: int,
                    subreddit_sentiment: float) -> tuple:
    """
    Compute final verdict and confidence for a coin.

    Returns: (verdict, confidence, mention_score)
      verdict:    SHOOT_UP | SHOOT_DOWN | WATCH | LOW_DATA
      confidence: HIGH | MEDIUM | LOW
      mention_score: float in [-1, +1] from mention context alone
    """
    if mention_count < VERDICT_MIN_MENTIONS:
        return 'LOW_DATA', 'LOW', 0.0

    total = bull + bear
    mention_score = (bull - bear) / total if total > 0 else 0.0

    # Blend: 60% mention context score, 40% subreddit-level sentiment
    blended = 0.6 * mention_score + 0.4 * subreddit_sentiment

    if blended >= 0.2:
        verdict = 'SHOOT_UP'
    elif blended <= -0.2:
        verdict = 'SHOOT_DOWN'
    else:
        verdict = 'WATCH'

    if mention_count >= CONFIDENCE_HIGH:
        confidence = 'HIGH'
    elif mention_count >= CONFIDENCE_MEDIUM:
        confidence = 'MEDIUM'
    else:
        confidence = 'LOW'

    return verdict, confidence, round(mention_score, 4)


# ─── MAIN SCRAPER ─────────────────────────────────────────────────────────────

def scrape_all():
    today_str = datetime.now().strftime('%Y-%m-%d')

    # Always rebuild the coin regex with today's live holdings first
    build_coin_regex()

    print(f"\n{'='*70}")
    print(f"🔍 REDDIT SENTIMENT SCRAPER v3 — {today_str}")
    print(f"   Subreddits: {len(SUBREDDITS)} | Posts/sub: {POSTS_PER_SUB} | "
          f"Comments/post: {COMMENTS_PER_POST}")
    print(f"   Coin aliases tracked: {len(set(COIN_ALIASES.values()))}")
    print(f"{'='*70}\n")

    all_rows = []

    # Global mention accumulator: coin → {bull, bear, neutral, total}
    mention_tally = defaultdict(lambda: {'bull': 0, 'bear': 0, 'neutral': 0, 'total': 0})

    def _tally_text(text: str, weight: float = 1.0):
        """Extract mentions from text, score them, and tally results."""
        mentions = extract_coin_mentions(text)
        for symbol, context in mentions:
            score = score_mention_in_context(symbol, context)
            mention_tally[symbol]['total'] += 1
            if score > 0.05:
                mention_tally[symbol]['bull'] += weight
            elif score < -0.05:
                mention_tally[symbol]['bear'] += weight
            else:
                mention_tally[symbol]['neutral'] += weight

    for subreddit, coin in SUBREDDITS:
        print(f"📡 r/{subreddit} → primary:{coin}")

        posts = fetch_today_posts(subreddit, POSTS_PER_SUB)
        if not posts:
            print(f"   ⚠️  No posts found today, skipping.")
            time.sleep(SLEEP_BETWEEN_CALLS)
            continue

        print(f"   Found {len(posts)} posts today")

        for i, post in enumerate(posts):
            # Fetch comments
            comments = fetch_comments(subreddit, post['post_id'])
            time.sleep(SLEEP_BETWEEN_CALLS)

            # ── Layer 1: Subreddit-level score (existing logic) ────────────
            title_text   = post['title'] + ' ' + post['body']
            title_score  = score_text(title_text)
            comment_scores = [score_text(c) for c in comments if c]
            avg_comment_score = (
                sum(comment_scores) / len(comment_scores)
                if comment_scores else 0.0
            )
            combined_score = (0.4 * title_score) + (0.6 * avg_comment_score)
            upvote_weight  = post['upvote_ratio']
            final_score    = combined_score * upvote_weight

            all_rows.append({
                'date':              today_str,
                'subreddit':         subreddit,
                'coin':              coin,
                'post_id':           post['post_id'],
                'title':             post['title'][:120],
                'post_score':        post['score'],
                'upvote_ratio':      post['upvote_ratio'],
                'num_comments':      post['num_comments'],
                'title_sentiment':   round(title_score, 4),
                'comment_sentiment': round(avg_comment_score, 4),
                'combined_score':    round(combined_score, 4),
                'final_score':       round(final_score, 4),
                'comments_fetched':  len(comments),
            })

            # ── Layer 2: Cross-subreddit mention extraction ───────────────
            # Title (weight 1.5 — more intentional than comments)
            _tally_text(post['title'] + ' ' + post['body'], weight=1.5)
            # Each comment (weight 1.0)
            for comment in comments:
                _tally_text(comment, weight=1.0)

            print(f"   [{i+1:02d}] {post['title'][:55]:<55} | Sub:{final_score:+.3f}")

        print()

    # ── Save raw CSV ───────────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(OUTPUT_DIR, f"reddit_sentiment_{today_str}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Raw data saved → {csv_path}  ({len(df)} rows)")

    # ── Aggregate per coin (subreddit-level) ──────────────────────────────────
    coin_sentiment = (
        df.groupby('coin')
          .agg(
              avg_sentiment=('final_score', 'mean'),
              total_posts=('post_id', 'count'),
              total_upvotes=('post_score', 'sum'),
          )
          .reset_index()
          .sort_values('avg_sentiment', ascending=False)
    )
    subreddit_scores = {
        row['coin']: row['avg_sentiment']
        for _, row in coin_sentiment.iterrows()
    }

    # ── Build rich per-coin output dict ───────────────────────────────────────
    sentiment_dict = {}

    # First, for coins with subreddit tracking
    for _, row in coin_sentiment.iterrows():
        coin = row['coin']
        sub_sentiment = row['avg_sentiment']
        tally    = mention_tally.get(coin, {})
        bull_cnt = tally.get('bull', 0)
        bear_cnt = tally.get('bear', 0)
        total    = tally.get('total', 0)

        verdict, confidence, mention_score = compute_verdict(
            total, bull_cnt, bear_cnt, sub_sentiment
        )

        sentiment_dict[coin] = {
            'sentiment':       round(sub_sentiment, 4),
            'posts_analyzed':  int(row['total_posts']),
            'total_upvotes':   int(row['total_upvotes']),
            'signal':          ('BULLISH' if sub_sentiment > 0.1
                                else 'BEARISH' if sub_sentiment < -0.1
                                else 'NEUTRAL'),
            'mention_count':   total,
            'bull_mentions':   round(bull_cnt),
            'bear_mentions':   round(bear_cnt),
            'mention_score':   mention_score,
            'verdict':         verdict,
            'confidence':      confidence,
        }

    # Then, add any coins found ONLY via mention scanning (not in subreddit list)
    for coin, tally in mention_tally.items():
        if coin in sentiment_dict:
            # Already handled above, just update mention fields if richer
            existing = sentiment_dict[coin]
            bull_cnt = tally.get('bull', 0)
            bear_cnt = tally.get('bear', 0)
            total    = tally.get('total', 0)
            if total > existing.get('mention_count', 0):
                verdict, confidence, mention_score = compute_verdict(
                    total, bull_cnt, bear_cnt, existing['sentiment']
                )
                existing.update({
                    'mention_count': total,
                    'bull_mentions': round(bull_cnt),
                    'bear_mentions': round(bear_cnt),
                    'mention_score': mention_score,
                    'verdict':       verdict,
                    'confidence':    confidence,
                })
        else:
            # Coin only found via cross-subreddit mentions
            bull_cnt = tally.get('bull', 0)
            bear_cnt = tally.get('bear', 0)
            total    = tally.get('total', 0)
            verdict, confidence, mention_score = compute_verdict(
                total, bull_cnt, bear_cnt, 0.0
            )
            combined_score = mention_score  # No subreddit baseline
            sentiment_dict[coin] = {
                'sentiment':      round(combined_score, 4),
                'posts_analyzed': 0,
                'total_upvotes':  0,
                'signal':         ('BULLISH' if combined_score > 0.1
                                   else 'BEARISH' if combined_score < -0.1
                                   else 'NEUTRAL'),
                'mention_count':  total,
                'bull_mentions':  round(bull_cnt),
                'bear_mentions':  round(bear_cnt),
                'mention_score':  mention_score,
                'verdict':        verdict,
                'confidence':     confidence,
            }

    # ── Holdings Correlation ───────────────────────────────────────────────────
    holdings = load_current_holdings()
    if holdings:
        print(f"\n📊 HOLDINGS CORRELATION — {len(holdings)} active positions")
        correlation = compute_holdings_correlation(sentiment_dict, holdings)
        market_overall = compute_market_overall(sentiment_dict)

        sentiment_dict['_holdings_correlation'] = {
            'timestamp':      datetime.now().isoformat(),
            'market_overall': market_overall,
            'holdings':       correlation,
        }

        # Print correlation table
        print(f"\n{'Symbol':<16} {'Verdict':<12} {'Conf':<8} {'Mentions':<10} {'Rec'}")
        print(f"{'-'*75}")
        for sym, info in correlation.items():
            print(f"{sym:<16} {info['reddit_verdict']:<12} {info['confidence']:<8} "
                  f"{info['mention_count']:<10} {info['recommendation']}")
    else:
        print("\n   ℹ️  No active holdings found — skipping correlation step.")
        sentiment_dict['_holdings_correlation'] = {
            'timestamp':      datetime.now().isoformat(),
            'market_overall': compute_market_overall(sentiment_dict),
            'holdings':       {},
        }

    # ── Meta ──────────────────────────────────────────────────────────────────
    sentiment_dict['_meta'] = {
        'date':               today_str,
        'timestamp':          datetime.now().isoformat(),
        'subreddits_scraped': len(SUBREDDITS),
        'total_posts':        len(df),
        'coins_tracked':      len([k for k in sentiment_dict if not k.startswith('_')]),
        'version':            '3.0',
    }

    # ── Save latest JSON ───────────────────────────────────────────────────────
    json_path = os.path.join(OUTPUT_DIR, "reddit_sentiment_latest.json")
    with open(json_path, 'w') as f:
        json.dump(sentiment_dict, f, indent=2)

    # ── Print final summary ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"📊 COIN SENTIMENT SUMMARY — {today_str}")
    print(f"{'='*70}")
    print(f"{'Coin':<12} {'Sentiment':>10} {'Signal':<10} {'Verdict':<12} "
          f"{'Conf':<8} {'Mentions':>8}")
    print(f"{'-'*70}")

    # Sort by mention count + sentiment for most relevant first
    ranked_coins = sorted(
        [(k, v) for k, v in sentiment_dict.items() if not k.startswith('_')],
        key=lambda x: (x[1].get('mention_count', 0), x[1].get('sentiment', 0)),
        reverse=True
    )

    for coin, data in ranked_coins[:25]:  # Show top 25 by activity
        verdict_emoji = {
            'SHOOT_UP':   '🚀',
            'SHOOT_DOWN': '💥',
            'WATCH':      '👀',
            'LOW_DATA':   '⚪',
        }.get(data.get('verdict', 'LOW_DATA'), '⚪')

        signal_str = ('🟢 BULLISH' if data['sentiment'] > 0.1
                      else '🔴 BEARISH' if data['sentiment'] < -0.1
                      else '⚪ NEUTRAL')

        print(f"{coin:<12} {data['sentiment']:>+10.4f} {signal_str:<12} "
              f"{verdict_emoji} {data.get('verdict','LOW_DATA'):<10} "
              f"{data.get('confidence','LOW'):<8} "
              f"{data.get('mention_count',0):>7}")

    print(f"\n✅ Latest JSON saved → {json_path}")
    print(f"   Coins tracked: {sentiment_dict['_meta']['coins_tracked']}")
    print(f"{'='*70}\n")

    return sentiment_dict


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    scrape_all()
