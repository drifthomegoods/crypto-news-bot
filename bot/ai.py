"""
AI layer: picks important stories and writes them in Armenian.

Three passes for quality:
  1. select   - cluster duplicates, rate importance, extract facts (in English,
                where models are strongest)
  2. write    - write Armenian posts from the facts, guided by style_hy.md
  3. proofread- fix grammar / unnatural phrasing, keep facts and numbers

Guards: Armenian-script check and a number check (no number may appear that
is not in the facts or market data). A failed proofread falls back to the draft.

Providers: Gemini (free tier, default when GEMINI_API_KEY is set) or Anthropic.
For Gemini the bot discovers the best available free Flash model by itself;
set GEMINI_MODEL to force one.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import requests

log = logging.getLogger("newsbot.ai")

STYLE_FILE = Path(__file__).resolve().parent / "style_hy.md"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
ARMENIAN = re.compile(r"[Ա-և]")


def provider() -> str:
    p = os.getenv("AI_PROVIDER", "").strip().lower()
    if p in ("gemini", "anthropic"):
        return p
    if os.getenv("GEMINI_API_KEY"):
        return "gemini"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    raise RuntimeError("No AI key: set GEMINI_API_KEY (free) or ANTHROPIC_API_KEY")


# ---------------------------------------------------------------- Gemini

_gemini_models: list[str] | None = None


def _version(name: str) -> float:
    m = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
    return float(m.group(1)) if m else 0.0


def rank_gemini_models(names: list[str]) -> list[str]:
    """Best free text models first: Flash before Flash-Lite, stable before preview, newer first."""
    bad = ("tts", "image", "audio", "live", "embedding", "vision", "thinking-exp", "learnlm", "robotics", "computer-use")
    cands = [n for n in names if "flash" in n and not any(b in n for b in bad)]

    def key(n: str):
        lite = "lite" in n
        unstable = any(s in n for s in ("preview", "exp"))
        alias = n.endswith("-latest")
        return (lite, not alias, unstable, -_version(n), n)

    cands = list(dict.fromkeys(cands))
    # Old generations get retired for new projects; skip them when newer ones exist.
    newest = max((_version(n) for n in cands), default=0.0)
    if newest >= 3:
        cands = [n for n in cands if n.endswith("-latest") or _version(n) >= 3]
    return sorted(cands, key=key)


def gemini_models() -> list[str]:
    global _gemini_models
    if _gemini_models is not None:
        return _gemini_models
    forced = os.getenv("GEMINI_MODEL", "").strip()
    found: list[str] = []
    try:
        r = requests.get(
            f"{GEMINI_BASE}/models",
            params={"pageSize": 1000},
            headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]},
            timeout=20,
        )
        r.raise_for_status()
        for m in r.json().get("models", []):
            if "generateContent" in m.get("supportedGenerationMethods", []):
                found.append(m["name"].removeprefix("models/"))
    except Exception as exc:
        log.warning("could not list Gemini models: %s", exc)
    ranked = rank_gemini_models(found) or ["gemini-flash-latest", "gemini-flash-lite-latest"]
    _gemini_models = ([forced] if forced else []) + [m for m in ranked if m != forced]
    log.info("Gemini model order: %s", ", ".join(_gemini_models[:4]))
    return _gemini_models


class AIUnavailable(RuntimeError):
    """Temporary provider problem (overload, timeout). Items are retried next run."""


def _gemini(system: str, user: str, temperature: float) -> str:
    import time

    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {
            "temperature": temperature,
            "responseMimeType": "application/json",
            "maxOutputTokens": 8192,
        },
    }
    headers = {"x-goog-api-key": os.environ["GEMINI_API_KEY"], "Content-Type": "application/json"}
    errors: list[str] = []
    auth_errors = 0
    models = gemini_models()[:9]
    for model in models:
        for attempt in range(2):
            try:
                r = requests.post(
                    f"{GEMINI_BASE}/models/{model}:generateContent", headers=headers, json=body, timeout=90
                )
            except requests.RequestException as exc:
                errors.append(f"{model}: {type(exc).__name__}")
                log.warning("Gemini %s: %s, trying next model", model, type(exc).__name__)
                break
            if r.status_code in (500, 503) and attempt == 0:
                log.warning("Gemini %s: HTTP %s (overloaded), retrying in 5s", model, r.status_code)
                time.sleep(5)
                continue
            if r.status_code != 200:
                if r.status_code in (401, 403) or "API_KEY_INVALID" in r.text:
                    auth_errors += 1
                errors.append(f"{model}: HTTP {r.status_code}")
                log.warning("Gemini %s: HTTP %s %s, trying next model", model, r.status_code, r.text[:150].replace("\n", " "))
                break
            cands = r.json().get("candidates") or []
            parts = (cands[0].get("content") or {}).get("parts", []) if cands else []
            text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
            if text.strip():
                if model != models[0]:
                    log.info("answered by %s", model)
                return text
            errors.append(f"{model}: empty ({cands[0].get('finishReason') if cands else 'no candidates'})")
            log.warning("Gemini %s: empty reply, trying next model", model)
            break
    if auth_errors and auth_errors == len(errors):
        raise RuntimeError("Gemini rejected the API key - check GEMINI_API_KEY: " + "; ".join(errors))
    raise AIUnavailable("all Gemini models busy or failed: " + "; ".join(errors))


# ---------------------------------------------------------------- Anthropic


def _anthropic(system: str, user: str, temperature: float) -> str:
    import anthropic

    msg = anthropic.Anthropic().messages.create(
        model=os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5"),
        max_tokens=4000,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def llm_json(system: str, user: str, temperature: float = 0.3) -> dict:
    fn = _gemini if provider() == "gemini" else _anthropic
    text = fn(system, user, temperature)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON in model reply")
    return json.loads(text[start : end + 1])


# ---------------------------------------------------------------- prompts

SELECT_PROMPT = """You are the news editor of a crypto-trading Telegram channel.
You receive fresh news items from several outlets, live market data, and titles the channel posted in the last 48 hours.

1. Group items describing the same story (outlets often cover the same event). One story = one post.
2. Drop stories already posted (see recent_posts) unless there is a genuinely new, material development.
3. Rate importance for crypto traders, 1-10:
   9-10 market-moving: major exchange hacks/failures, ETF approvals/rejections, Fed decisions, major regulation or court rulings, BTC/ETH moves above ~5%, large liquidation cascades, big institutional buys/sells.
   7-8 significant: notable news on top-20 coins or major exchanges/companies, meaningful ETF-flow or on-chain data, important protocol upgrades, notable regulatory steps.
   1-6 routine: price predictions, opinion, sponsored/promo content, guides, minor altcoins, small partnerships, rehashes.
4. For stories rated 7+, extract the facts.

Fields per story:
- item_ids: ids of all items in the group
- importance: 1-10
- sentiment: "bullish" | "bearish" | "neutral"
- coins: tickers directly affected, most relevant first (e.g. ["BTC","BNB"]); may be empty
- source_url: link of the most authoritative item in the group
- facts_en: 3-5 plain English sentences with every key fact and number taken ONLY from the items. Mark unconfirmed claims as unconfirmed. No speculation.
- impact_en: one sentence on likely market impact and which coins. Analysis, not advice.

Return ONLY JSON: {"stories":[{"item_ids":[],"importance":8,"sentiment":"bearish","coins":[],"source_url":"","facts_en":"","impact_en":""}]}
Include only stories rated 7 or higher; if none, return {"stories":[]}."""

WRITE_PROMPT = """You write posts for an Armenian-language crypto-trading Telegram channel.
For each story you get English facts. Write it in natural, fluent Eastern Armenian (Armenian script), as a skilled Armenian financial journalist would, not as a translation.
Use ONLY the facts given; never add numbers, names or events. Keep every number exactly as in the facts.
Follow the style guide below strictly: terminology table, punctuation (։ « » ՝), structure and examples.

Return ONLY JSON: {"posts":[{"i":0,"title_hy":"","body_hy":"","impact_hy":""}]} with one post per story, "i" = story index.

STYLE GUIDE:
"""

PROOF_PROMPT = """You are a strict Armenian copy editor for a crypto news channel.
You get English facts and an Armenian draft for each story. Improve the draft:
- fix grammar, case endings, spelling and punctuation (։ « » ՝, comma before «որ»)
- replace calques, Russianisms and unnatural word order with natural Eastern Armenian
- apply the terminology table of the style guide
- make sure every fact matches the English facts and no number was added, dropped or changed
Keep the meaning, structure and length. Do not add information.

Return ONLY JSON: {"posts":[{"i":0,"title_hy":"","body_hy":"","impact_hy":""}]} with the same "i" values.

STYLE GUIDE:
"""


def _style() -> str:
    try:
        return STYLE_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        return "(no style guide)"


# ---------------------------------------------------------------- guards

_NUM = re.compile(r"\d+(?:[.,  ]\d+)*")
_FREE_NUMS = {"24", "48", "1", "2", "3", "7", "30", "100"}


def _numbers(text: str) -> set[str]:
    return {re.sub(r"\D", "", m) for m in _NUM.findall(text or "")}


def numbers_ok(post: dict, allowed_text: str) -> bool:
    allowed = _numbers(allowed_text) | _FREE_NUMS
    used = _numbers(" ".join(str(post.get(k, "")) for k in ("title_hy", "body_hy", "impact_hy")))
    extra = used - allowed
    if extra:
        log.warning("numbers not in facts: %s", sorted(extra))
    return not extra


def armenian_ok(post: dict) -> bool:
    return all(len(ARMENIAN.findall(str(post.get(k, "")))) >= 10 for k in ("title_hy", "body_hy"))


def _valid(post: dict | None, allowed_text: str) -> bool:
    return bool(post) and bool(post.get("title_hy")) and bool(post.get("body_hy")) and armenian_ok(post) and numbers_ok(post, allowed_text)


# ---------------------------------------------------------------- pipeline


def select_stories(items: list[dict], market_text: str, recent_titles: list[str], min_importance: int) -> list[dict]:
    payload = {
        "now_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "market_data": market_text,
        "recent_posts": recent_titles,
        "items": [{k: it[k] for k in ("id", "source", "published", "title", "summary", "link")} for it in items],
    }
    data = llm_json(SELECT_PROMPT, json.dumps(payload, ensure_ascii=False), temperature=0.2)
    out = []
    for s in data.get("stories", []):
        try:
            if int(s.get("importance", 0)) >= min_importance and s.get("facts_en") and str(s.get("source_url", "")).startswith("http"):
                out.append(s)
        except (TypeError, ValueError):
            continue
    out.sort(key=lambda s: int(s["importance"]), reverse=True)
    return out


def _by_index(data: dict) -> dict[int, dict]:
    res = {}
    for p in data.get("posts", []):
        try:
            res[int(p["i"])] = p
        except (KeyError, TypeError, ValueError):
            continue
    return res


def write_armenian(stories: list[dict], market_text: str) -> list[dict]:
    if not stories:
        return []
    style = _style()
    facts = [{"i": i, "facts_en": s["facts_en"], "impact_en": s.get("impact_en", ""), "coins": s.get("coins", [])} for i, s in enumerate(stories)]
    drafts = _by_index(llm_json(WRITE_PROMPT + style, json.dumps({"stories": facts}, ensure_ascii=False), temperature=0.4))

    proofed: dict[int, dict] = {}
    if os.getenv("AI_PROOFREAD", "1") != "0" and drafts:
        review = [{"i": i, "facts_en": facts[i]["facts_en"], "draft": drafts[i]} for i in drafts if i < len(stories)]
        try:
            proofed = _by_index(llm_json(PROOF_PROMPT + style, json.dumps({"stories": review}, ensure_ascii=False), temperature=0.2))
        except Exception as exc:
            log.warning("proofread failed, using drafts: %s", exc)

    result = []
    for i, s in enumerate(stories):
        allowed = s["facts_en"] + " " + s.get("impact_en", "") + " " + market_text
        chosen = next((p for p in (proofed.get(i), drafts.get(i)) if _valid(p, allowed)), None)
        if not chosen:
            log.warning("story %d dropped: no valid Armenian text", i)
            continue
        result.append({**s, "title_hy": chosen["title_hy"].strip(), "body_hy": chosen["body_hy"].strip(), "impact_hy": str(chosen.get("impact_hy", "")).strip()})
    return result
