"""
Crypto news -> Telegram channel bot (Armenian).

Every run:
  1. Pulls fresh items from crypto news RSS feeds.
  2. Drops anything already seen.
  3. Asks AI (Gemini free tier or Claude, see ai.py) to cluster duplicates,
     rate importance and write Armenian posts (select -> write -> proofread).
  4. Posts only important stories to Telegram, with live market facts
     (CoinGecko prices + Fear & Greed index) attached.
  5. Saves state (seen items + recent posts) to state/state.json.

Required env vars:
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHANNEL_ID, and GEMINI_API_KEY (free) or ANTHROPIC_API_KEY
Optional:
  MIN_IMPORTANCE (default 7), MAX_POSTS_PER_RUN (default 3), MAX_AGE_HOURS (default 6),
  AI_PROVIDER (gemini|anthropic), GEMINI_MODEL, CLAUDE_MODEL, AI_PROOFREAD=0 to skip proofreading,
  DRY_RUN=1 (print instead of posting), TEST_MESSAGE=1 (send a test post and exit)
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import os
import re
import sys
import time
from calendar import timegm
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ai  # noqa: E402

log = logging.getLogger("newsbot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------- settings

FEEDS = {
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "Cointelegraph": "https://cointelegraph.com/rss",
    "Decrypt": "https://decrypt.co/feed",
    "The Block": "https://www.theblock.co/rss.xml",
    "Bitcoin Magazine": "https://bitcoinmagazine.com/.rss/full/",
    "CryptoSlate": "https://cryptoslate.com/feed/",
}

# Coins we can attach live prices for: symbol -> CoinGecko id
TRACKED_COINS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "BNB": "binancecoin",
    "XRP": "ripple",
    "DOGE": "dogecoin",
    "ADA": "cardano",
    "TON": "the-open-network",
    "TRX": "tron",
    "AVAX": "avalanche-2",
    "LINK": "chainlink",
    "HYPE": "hyperliquid",
    "SUI": "sui",
    "LTC": "litecoin",
}

MIN_IMPORTANCE = int(os.getenv("MIN_IMPORTANCE", "7"))
MAX_POSTS_PER_RUN = int(os.getenv("MAX_POSTS_PER_RUN", "3"))
MAX_AGE_HOURS = float(os.getenv("MAX_AGE_HOURS", "6"))
MAX_ITEMS_TO_MODEL = 40
DRY_RUN = os.getenv("DRY_RUN") == "1"
TEST_MESSAGE = os.getenv("TEST_MESSAGE") == "1"

STATE_FILE = Path(__file__).resolve().parent.parent / "state" / "state.json"
SEEN_TTL_SEC = 7 * 24 * 3600
RECENT_POSTS_TTL_SEC = 48 * 3600
HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (crypto-news-telegram-bot)"}

# ---------------------------------------------------------------- state


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.warning("state.json is corrupt, starting fresh")
    return {"initialized": False, "seen": {}, "recent_posts": []}


def save_state(state: dict) -> None:
    now = time.time()
    state["seen"] = {k: v for k, v in state["seen"].items() if now - v < SEEN_TTL_SEC}
    state["recent_posts"] = [p for p in state["recent_posts"] if now - p["ts"] < RECENT_POSTS_TTL_SEC]
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- news


def _clean(text: str, limit: int) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = html.unescape(re.sub(r"\s+", " ", text)).strip()
    return text[:limit]


def _item_id(link: str, title: str) -> str:
    key = (link or title).split("?")[0].strip().lower()
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def fetch_news() -> list[dict]:
    now = time.time()
    items: list[dict] = []
    for source, url in FEEDS.items():
        try:
            resp = requests.get(url, headers=HTTP_HEADERS, timeout=20)
            resp.raise_for_status()
            feed = feedparser.parse(resp.content)
        except Exception as exc:  # one broken feed must not stop the bot
            log.warning("feed %s failed: %s", source, exc)
            continue
        for e in feed.entries:
            ts_struct = e.get("published_parsed") or e.get("updated_parsed")
            if not ts_struct:
                continue
            ts = timegm(ts_struct)
            if now - ts > MAX_AGE_HOURS * 3600:
                continue
            title = _clean(e.get("title", ""), 300)
            link = e.get("link", "")
            if not title or not link:
                continue
            items.append({
                "id": _item_id(link, title),
                "source": source,
                "title": title,
                "summary": _clean(e.get("summary", ""), 500),
                "link": link,
                "published": datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "ts": ts,
            })
    # newest first, unique by id
    uniq: dict[str, dict] = {}
    for it in sorted(items, key=lambda x: x["ts"], reverse=True):
        uniq.setdefault(it["id"], it)
    log.info("fetched %d fresh items from %d feeds", len(uniq), len(FEEDS))
    return list(uniq.values())


# ---------------------------------------------------------------- market facts


def fetch_market() -> dict:
    market: dict = {"coins": {}, "fear_greed": None, "global": None}
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={
                "ids": ",".join(TRACKED_COINS.values()),
                "vs_currencies": "usd",
                "include_24hr_change": "true",
            },
            headers=HTTP_HEADERS,
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
        for sym, cg_id in TRACKED_COINS.items():
            if cg_id in data and "usd" in data[cg_id]:
                market["coins"][sym] = {
                    "price": data[cg_id]["usd"],
                    "change_24h": data[cg_id].get("usd_24h_change"),
                }
    except Exception as exc:
        log.warning("CoinGecko prices failed: %s", exc)
    try:
        r = requests.get("https://api.coingecko.com/api/v3/global", headers=HTTP_HEADERS, timeout=20)
        r.raise_for_status()
        g = r.json()["data"]
        market["global"] = {
            "btc_dominance": g["market_cap_percentage"]["btc"],
            "mcap_change_24h": g["market_cap_change_percentage_24h_usd"],
            "total_mcap": g["total_market_cap"]["usd"],
        }
    except Exception as exc:
        log.warning("CoinGecko global failed: %s", exc)
    try:
        r = requests.get("https://api.alternative.me/fng/", params={"limit": 1}, headers=HTTP_HEADERS, timeout=20)
        r.raise_for_status()
        fg = r.json()["data"][0]
        market["fear_greed"] = {"value": int(fg["value"]), "label": fg["value_classification"]}
    except Exception as exc:
        log.warning("Fear & Greed failed: %s", exc)
    return market


def fmt_price(p: float) -> str:
    if p >= 1000:
        return f"${p:,.0f}"
    if p >= 1:
        return f"${p:,.2f}"
    return f"${p:.4f}"


def fmt_change(c: float | None) -> str:
    if c is None:
        return ""
    arrow = "📈" if c >= 0 else "📉"
    return f"{arrow} {c:+.1f}%"


FG_EMOJI = {"Extreme Fear": "😱", "Fear": "😨", "Neutral": "😐", "Greed": "🤑", "Extreme Greed": "🔥"}
FG_RU = {
    "Extreme Fear": "Экстрим. страх",
    "Fear": "Страх",
    "Neutral": "Нейтрально",
    "Greed": "Жадность",
    "Extreme Greed": "Экстрим. жадность",
}


def facts_block(coins: list[str], market: dict) -> str:
    """Monospaced market table, trader style (shown inside <pre>)."""
    rows = []
    shown = [c for c in coins if c in market["coins"]][:3]
    if "BTC" not in shown and "BTC" in market["coins"]:
        shown = (shown + ["BTC"])[:3] if shown else ["BTC", "ETH"]
        shown = [s for s in shown if s in market["coins"]]
    for sym in shown:
        c = market["coins"][sym]
        ch = c["change_24h"]
        chg = "" if ch is None else f"{'▲' if ch >= 0 else '▼'} {abs(ch):.2f}%"
        rows.append(f"{sym:<5} {fmt_price(c['price']):>10}  {chg}")
    fg = market.get("fear_greed")
    if fg:
        rows.append(f"{'F&G':<5} {str(fg['value']):>10}  {FG_RU.get(fg['label'], fg['label'])} {FG_EMOJI.get(fg['label'], '')}".rstrip())
    g = market.get("global")
    if g:
        rows.append(f"{'BTC.D':<5} {g['btc_dominance']:>9.1f}%")
    return "\n".join(rows)


def market_summary_for_model(market: dict) -> str:
    parts = []
    for sym, c in market["coins"].items():
        ch = f"{c['change_24h']:+.2f}%" if c["change_24h"] is not None else "n/a"
        parts.append(f"{sym} {fmt_price(c['price'])} ({ch} 24h)")
    if market.get("global"):
        g = market["global"]
        parts.append(f"Total crypto mcap 24h change {g['mcap_change_24h']:+.2f}%, BTC dominance {g['btc_dominance']:.1f}%")
    if market.get("fear_greed"):
        parts.append(f"Fear & Greed {market['fear_greed']['value']} ({market['fear_greed']['label']})")
    return "; ".join(parts) or "market data unavailable"


# ---------------------------------------------------------------- AI


def ask_ai(items: list[dict], market: dict, recent_posts: list[dict]) -> list[dict]:
    market_text = market_summary_for_model(market)
    stories = ai.select_stories(
        items[:MAX_ITEMS_TO_MODEL], market_text, [p["title"] for p in recent_posts], MIN_IMPORTANCE
    )
    log.info("%d stories rated %d+", len(stories), MIN_IMPORTANCE)
    return ai.write_armenian(stories[:MAX_POSTS_PER_RUN], market_text)


# ---------------------------------------------------------------- Telegram

SENTIMENT = {
    "bullish": ("🟢", "БЫЧИЙ"),
    "bearish": ("🔴", "МЕДВЕЖИЙ"),
    "neutral": ("⚪️", "НЕЙТРАЛЬНО"),
}
SOURCE_NAMES = {
    "coindesk.com": "CoinDesk",
    "cointelegraph.com": "Cointelegraph",
    "decrypt.co": "Decrypt",
    "theblock.co": "The Block",
    "bitcoinmagazine.com": "Bitcoin Magazine",
    "cryptoslate.com": "CryptoSlate",
}
DIVIDER = "━━━━━━━━━━━━━━━━━━"


def source_name(url: str) -> str:
    from urllib.parse import urlparse

    host = urlparse(url).netloc.lower().removeprefix("www.")
    return SOURCE_NAMES.get(host, host or "Source")


def format_post(story: dict, market: dict) -> str:
    e = html.escape
    sentiment = story.get("sentiment", "neutral")
    emoji, label = SENTIMENT.get(sentiment, SENTIMENT["neutral"])
    importance = int(story.get("importance", 7))
    impact = "🚨 ВЫСОКОЕ ВЛИЯНИЕ" if importance >= 9 else "⚡️ СРЕДНЕЕ ВЛИЯНИЕ"
    coins = [str(c).upper() for c in story.get("coins", [])]

    parts = [
        f"{emoji} <b>{label}</b>  ·  {impact}",
        DIVIDER,
        f"<b>{e(story['title_hy'])}</b>",
        "",
        e(story["body_hy"]),
    ]
    if story.get("impact_hy"):
        parts += ["", f"<blockquote>💡 <b>Влияние:</b> {e(story['impact_hy'])}</blockquote>"]
    facts = facts_block(coins, market)
    if facts:
        parts += ["", "📊 <b>Рынок</b>", f"<pre>{e(facts)}</pre>"]
    url = story["source_url"]
    parts += [DIVIDER, f'🔗 Источник: <a href="{e(url, quote=True)}">{e(source_name(url))}</a>']
    tags = " ".join(f"#{c}" for c in coins[:4])
    parts += [f"{tags} #{label.capitalize()}".strip()]
    return "\n".join(parts)[:4000]


def send_telegram(text: str) -> None:
    if DRY_RUN:
        print("\n----- DRY RUN POST -----\n" + text + "\n------------------------")
        return
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHANNEL_ID"]
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    for attempt in range(3):
        r = requests.post(
            url,
            data={"chat_id": chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": "true"},
            timeout=20,
        )
        if r.status_code == 429:
            wait = r.json().get("parameters", {}).get("retry_after", 5)
            log.warning("Telegram rate limit, waiting %ss", wait)
            time.sleep(wait + 1)
            continue
        if not r.ok:
            raise RuntimeError(f"Telegram error {r.status_code}: {r.text[:300]}")
        return
    raise RuntimeError("Telegram: too many retries")


# ---------------------------------------------------------------- main


def main() -> int:
    if TEST_MESSAGE:
        market = fetch_market()
        send_telegram(
            "✅ <b>БОТ ОНЛАЙН</b>\n" + DIVIDER + "\nԲոտը միացված է և աշխատում է\n\n"
            "📊 <b>Рынок</b>\n<pre>" + html.escape(facts_block(["BTC", "ETH"], market)) + "</pre>"
        )
        log.info("test message sent")
        return 0

    state = load_state()
    items = fetch_news()
    new_items = [it for it in items if it["id"] not in state["seen"]]
    now = time.time()

    if not state.get("initialized"):
        # First run: remember what is already out there, so the channel isn't flooded with old news.
        for it in items:
            state["seen"][it["id"]] = now
        state["initialized"] = True
        save_state(state)
        log.info("first run: marked %d existing items as seen, nothing posted", len(items))
        return 0

    if not new_items:
        log.info("no new items")
        save_state(state)
        return 0

    log.info("%d new items, asking AI (%s)", len(new_items), ai.provider())
    market = fetch_market()
    try:
        stories = ask_ai(new_items, market, state["recent_posts"])
    except ai.AIUnavailable as exc:
        # Temporary (overload/timeout): leave items unseen, the next run retries them.
        log.warning("AI temporarily unavailable, will retry next run: %s", exc)
        save_state(state)
        return 0
    except Exception as exc:
        log.error("AI step failed: %s", exc)
        save_state(state)
        return 1

    posted = 0
    for story in stories[:MAX_POSTS_PER_RUN]:
        try:
            send_telegram(format_post(story, market))
        except Exception as exc:
            log.error("posting failed: %s", exc)
            continue
        state["recent_posts"].append({"ts": now, "title": story["title_hy"]})
        posted += 1
        time.sleep(3)

    for it in new_items:
        state["seen"][it["id"]] = now
    save_state(state)
    log.info("done: %d stories qualified, %d posted", len(stories), posted)
    return 0


if __name__ == "__main__":
    sys.exit(main())
