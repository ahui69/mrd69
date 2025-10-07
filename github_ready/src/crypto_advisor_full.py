"""
/workspace/mrd69/overmind/crypto_advisor_full.py

CRYPTO ADVISOR FULL - ENHANCED VERSION
======================================

🚀 Must-Have Features ✅ COMPLETED:
✅ Real-time alerts - portfolio monitoring + traditional price/volume alerts
✅ Automatic portfolio tracker - ETH balance scanning, USD/PLN conversion, performance tracking
✅ Token scoring system - composite scoring (liquidity 40%, trust 30%, community 30%) with risk levels
✅ Backtesting engine - SMA crossover strategy with Sharpe ratio, drawdown, win rate analysis

🔧 Pro Features (NEXT PHASE):
🔧 Social sentiment analysis - Twitter/X, Reddit, Telegram analysis + sentiment heatmap
🔧 Risk radar - automated scam/honeypot/rugpull detection algorithms
🔧 CMC/CoinGecko integration - enhanced API integration with rate limiting + news feeds
🔧 Daily reports - automated PDF/HTML generation with charts, trends, predictions

🚀 Ultra Features (ADVANCED):
🚀 GitHub activity mining - developer activity monitoring for project health scoring
🚀 DeFi signaling - liquidity pool tracking (Raydium, Uniswap, PancakeSwap) across chains
🚀 Trading bot - semi-autonomous trading with risk management (alert → analysis → execution)
🚀 Investment mentor - AI explanations of market mechanics and investment reasoning

Architecture:
- Dataclass-based portfolio management with JSON persistence
- Composite scoring algorithms with configurable weights
- Comprehensive backtesting framework with multiple strategy support
- Enhanced CLI with portfolio management, scoring, and backtesting commands
```
- Wbudowane rozszerzenia Etherscan Plus (gas, balances, tx, ERC20/NFT, contract meta)

Darmowe integracje:
- CoinGecko (bez klucza): ceny, rynki, wykresy, global
- Binance public REST (bez klucza): klines, premiumIndex/funding
- alternative.me (bez klucza): Fear & Greed Index
- Etherscan (z ENV ETHERSCAN_API_KEY; działa też z premium)

Integracje wewnętrzne:
- memory.get_memory().add_fact(...) — fakty, alerty, raporty
- psychika.psyche_event()/autopilot_cycle(...) — meta-sygnały
- common_llm.llm_main — generacja porad; domyślnie Qwen/Qwen2.5-72B-Instruct

Ulepszenia:
- Risk score łączący RSI, zmienność, impuls, pump, funding, FNG, BTC dominance
- Watch-loop z alertami: pump, funding, ekstremalny FNG, tani/drogi GAS
- Buforowany HTTP, retry, cache in-memory
- Kompatybilne CLI: screener/analyze/report/watch/erc20/portfolio/alerts/backtest
"""

from __future__ import annotations

import html
import json
import os
from . import config
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry as _Retry

# Load environment variables
load_dotenv()

# ───────────────────────────────────────────────────────────────────────────
# I/O
# ───────────────────────────────────────────────────────────────────────────
ROOT = Path("/workspace/mrd69")
OUT_DIR = Path(os.getenv("OUT_DIR", str(ROOT / "out" / "crypto")))
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = OUT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())


def _slug(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", (s or "").strip()).strip("-").lower()
    return s or "item"


def _write_json(name: str, obj: Any) -> str:
    p = OUT_DIR / name
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return str(p)


def _write_text(name: str, text: str) -> str:
    p = OUT_DIR / name
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    return str(p)


def log_line(kind: str, payload: dict[str, Any]) -> None:
    with (LOG_DIR / f"{kind}.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": _now_iso(), **payload}, ensure_ascii=False) + "\n")


# ───────────────────────────────────────────────────────────────────────────
# HTTP + Cache
# ───────────────────────────────────────────────────────────────────────────

try:
    from urllib3.util.retry import Retry as _Retry
except Exception:
    _Retry = None

WEB_TIMEOUT = config.WEB_HTTP_TIMEOUT
UA = config.WEB_USER_AGENT or "OvermindCrypto/4.1"


def _http() -> requests.Session:
    s = requests.Session()
    if _Retry is not None:
        r = _Retry(
            total=3,
            backoff_factor=0.6,
            allowed_methods=frozenset(["GET", "POST"]),
            status_forcelist=(429, 500, 502, 503, 504),
        )
        ad = HTTPAdapter(max_retries=r, pool_connections=16, pool_maxsize=32)
        s.mount("http://", ad)
        s.mount("https://", ad)
    s.headers.update({"User-Agent": UA})
    return s


HTTP = _http()
_CACHE: dict[tuple[str, str], tuple[float, Any]] = {}
_CACHE_TTL = int(os.getenv("CACHE_TTL", "30") or "30")
_CACHE_LOCK = threading.Lock()


def _ck(url: str, params: dict[str, Any] | None) -> tuple[str, str]:
    return (url, json.dumps(params or {}, sort_keys=True, ensure_ascii=False))


def _cache_get(url: str, params: dict[str, Any] | None) -> Any:
    if _CACHE_TTL <= 0:
        return None
    k = _ck(url, params)
    with _CACHE_LOCK:
        v = _CACHE.get(k)
        if not v:
            return None
        exp, data = v
        if exp < time.time():
            _CACHE.pop(k, None)
            return None
        return data


def _cache_set(
    url: str, params: dict[str, Any] | None, data: Any, ttl: int | None = None
) -> None:
    if _CACHE_TTL <= 0:
        return
    k = _ck(url, params)
    t = ttl if ttl is not None else _CACHE_TTL
    with _CACHE_LOCK:
        _CACHE[k] = (time.time() + t, data)


def _get_json(
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> Any:
    c = _cache_get(url, params)
    if c is not None:
        return c
    r = HTTP.get(url, params=params or {}, headers=headers or {}, timeout=WEB_TIMEOUT)
    r.raise_for_status()
    j = r.json()
    _cache_set(url, params, j)
    return j


# ───────────────────────────────────────────────────────────────────────────
# FREE APIs: CoinGecko + Binance + Fear&Greed
# ───────────────────────────────────────────────────────────────────────────
# CoinGecko
CG_BASE = "https://api.coingecko.com/api/v3"
VS_DEFAULT = os.getenv("COINGECKO_VS", "usd")


def cg_list() -> list[dict[str, Any]]:
    return _get_json(f"{CG_BASE}/coins/list")


def cg_markets(
    vs: str = "usd", ids: str | None = None, n: int = 100, sparkline: bool = False
) -> list[dict[str, Any]]:
    p = {
        "vs_currency": vs,
        "order": "market_cap_desc",
        "per_page": min(250, max(1, n)),
        "page": 1,
        "sparkline": str(bool(sparkline)).lower(),
    }
    if ids:
        p["ids"] = ids
    return _get_json(f"{CG_BASE}/coins/markets", p)


def cg_top_markets(vs: str = "usd", n: int = 50) -> list[dict[str, Any]]:
    out = []
    left = n
    page = 1
    while left > 0:
        k = min(250, left)
        p = {
            "vs_currency": vs,
            "order": "market_cap_desc",
            "per_page": k,
            "page": page,
            "sparkline": "true",
        }
        js = _get_json(f"{CG_BASE}/coins/markets", p)
        if not js:
            break
        out.extend(js)
        left -= len(js)
        page += 1
    return out[:n]


def cg_chart(
    id_: str, days: int = 1, vs: str = "usd", interval: str = "minute"
) -> dict:
    return _get_json(
        f"{CG_BASE}/coins/{id_}/market_chart",
        {"vs_currency": vs, "days": days, "interval": interval},
    )


def cg_history(id_: str, days: int = 30, vs: str = "usd") -> dict:
    return cg_chart(id_, days=days, vs=vs, interval="daily")


def cg_global() -> dict:
    return _get_json(f"{CG_BASE}/global")


def cg_simple_price(
    ids: str,
    vs_currencies: str = "usd",
    include_market_cap: bool = False,
    include_24hr_vol: bool = False,
    include_24hr_change: bool = False,
) -> dict:
    """Get simple price data for coins from CoinGecko."""
    params = {
        "ids": ids,
        "vs_currencies": vs_currencies,
    }
    if include_market_cap:
        params["include_market_cap"] = "true"
    if include_24hr_vol:
        params["include_24hr_vol"] = "true"
    if include_24hr_change:
        params["include_24hr_change"] = "true"

    return _get_json(f"{CG_BASE}/simple/price", params)


def resolve_symbol(s: str) -> str:
    s = (s or "").strip().lower()
    if not s:
        return s
    if "-" in s and len(s) > 3:
        return s
    lst = cg_list()
    hit = [x for x in lst if (x.get("symbol", "") or "").lower() == s]
    if not hit:
        return s
    coin_id = hit[0].get("id")
    return str(coin_id) if coin_id else s


# Binance public
BIN_BASE = "https://api.binance.com"
FAPI_BASE = "https://fapi.binance.com"  # futures


def binance_klines(
    symbol: str, interval: str = "1m", limit: int = 120
) -> list[list[Any]]:
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": max(5, min(1000, limit)),
    }
    return _get_json(f"{BIN_BASE}/api/v3/klines", params)


def binance_funding(symbol: str) -> float | None:
    try:
        j = _get_json(f"{FAPI_BASE}/fapi/v1/premiumIndex", {"symbol": symbol.upper()})
        rate = j.get("lastFundingRate")
        return float(rate) if rate is not None else None
    except Exception:
        return None


def derive_binance_symbol_from_cg(id_or_sym: str) -> str | None:
    try:
        rid = resolve_symbol(id_or_sym)
        mk = cg_markets(ids=rid, vs="usd", n=1)
        sym = (mk[0].get("symbol") or "").upper()
        if not sym:
            return None
        alias = {"WBTC": "BTC", "WETH": "ETH"}
        sym = alias.get(sym, sym)
        return sym + "USDT"
    except Exception:
        return None


# Fear & Greed
FNG_URL = "https://api.alternative.me/fng/"


def fng_index() -> dict[str, Any]:
    try:
        j = _get_json(FNG_URL, {"limit": 1, "format": "json"})
        data = (j.get("data") or [{}])[0]
        return {
            "value": int(data.get("value", "50")),
            "classification": data.get("value_classification", "Neutral"),
            "ts": data.get("timestamp"),
        }
    except Exception:
        return {"value": 50, "classification": "Neutral", "ts": None}


# ───────────────────────────────────────────────────────────────────────────
# ETHERSCAN PLUS (wbudowane)
# ───────────────────────────────────────────────────────────────────────────
ES_BASE = "https://api.etherscan.io/api"
ETHERSCAN_KEY = os.getenv("ETHERSCAN_API_KEY", "").strip()


def _es_call(params: dict[str, Any]) -> dict[str, Any]:
    if ETHERSCAN_KEY:
        params = dict(params, apikey=ETHERSCAN_KEY)
    r = HTTP.get(
        ES_BASE, params=params, timeout=int(os.getenv("ETHERSCAN_TIMEOUT_S", "20"))
    )
    r.raise_for_status()
    return r.json()


def es_eth_price_usd() -> float | None:
    try:
        j = _es_call({"module": "stats", "action": "ethprice"})
        price = (j.get("result") or {}).get("ethusd")
        return float(price) if price is not None else None
    except Exception:
        return None


def es_gas_oracle() -> dict[str, Any]:
    j = _es_call({"module": "gastracker", "action": "gasoracle"})
    res = j.get("result") or {}

    def to_gwei(x):
        return float(x)

    return {
        "safe": to_gwei(res.get("SafeGasPrice", "0")),
        "propose": to_gwei(res.get("ProposeGasPrice", "0")),
        "fast": to_gwei(res.get("FastGasPrice", "0")),
        "base_fee": float(res.get("suggestBaseFee", "0")),
        "gas_used_ratio": res.get("gasUsedRatio"),
        "ts": int(time.time()),
    }


def es_gas_band_alert(low_gwei: float = 8.0, high_gwei: float = 45.0) -> str | None:
    try:
        g = es_gas_oracle()
        p = g["propose"]
        if p <= low_gwei:
            return f"Gas low {p} gwei"
        if p >= high_gwei:
            return f"Gas high {p} gwei"
    except Exception:
        return None
    return None


def es_native_balance(address: str) -> float | None:
    if not address:
        return None
    j = _es_call(
        {"module": "account", "action": "balance", "address": address, "tag": "latest"}
    )
    try:
        wei = int(j.get("result") or "0")
        return wei / 10**18
    except Exception:
        return None


def es_erc20_latest_transfers(contract: str, limit: int = 10) -> list[dict[str, Any]]:
    if not ETHERSCAN_KEY or not contract:
        return []
    js = _es_call(
        {
            "module": "account",
            "action": "tokentx",
            "contractaddress": contract,
            "page": 1,
            "offset": max(1, min(100, limit)),
            "sort": "desc",
        }
    )
    out = []
    for tx in (js.get("result") or [])[:limit]:
        try:
            out.append(
                {
                    "hash": tx.get("hash"),
                    "from": tx.get("from"),
                    "to": tx.get("to"),
                    "value": float(tx.get("value", "0"))
                    / (10 ** int(tx.get("tokenDecimal", "18") or "18")),
                    "token": tx.get("tokenSymbol") or "",
                    "time": int(tx.get("timeStamp") or "0"),
                }
            )
        except Exception:
            continue
    return out


def es_account_txs(address: str, limit: int = 50) -> list[dict[str, Any]]:
    if not address:
        return []
    j = _es_call(
        {
            "module": "account",
            "action": "txlist",
            "address": address,
            "page": 1,
            "offset": max(1, min(10000, limit)),
            "sort": "desc",
        }
    )
    return (j.get("result") or [])[:limit]


# ───────────────────────────────────────────────────────────────────────────
# TA
# ───────────────────────────────────────────────────────────────────────────
def ema(x: list[float], n: int) -> list[float]:
    if not x:
        return []
    k = 2 / (n + 1)
    out = [x[0]]
    for v in x[1:]:
        out.append(out[-1] + k * (v - out[-1]))
    return out


def sma(x: list[float], n: int) -> list[float]:
    if not x:
        return []
    out = []
    s = 0.0
    q = []
    for v in x:
        q.append(v)
        s += v
        if len(q) > n:
            s -= q.pop(0)
        out.append(s / len(q))
    return out


def rsi(prices: list[float], n: int = 14) -> float:
    if len(prices) < n + 1:
        return 50.0
    gains = []
    losses = []
    for a, b in zip(prices[:-1], prices[1:], strict=False):
        d = b - a
        gains.append(max(0.0, d))
        losses.append(max(0.0, -d))
    au = sum(gains[-n:]) / n
    ad = sum(losses[-n:]) / n
    if ad == 0:
        return 100.0
    rs = au / ad
    return 100.0 - (100.0 / (1.0 + rs))


def volatility(x: list[float]) -> float:
    if len(x) < 2:
        return 0.0
    mu = sum(x) / len(x)
    if mu == 0:
        return 0.0
    var = sum((v - mu) ** 2 for v in x) / len(x)
    return (var**0.5) / abs(mu)


def impulse(prices: list[float], n: int = 8) -> float:
    if len(prices) < n + 1:
        return 0.0
    e = ema(prices, n)
    return (e[-1] / e[-n] - 1.0) if len(e) >= n else 0.0


def fair_value_bands(
    prices: list[float], n: int = 20, mult: float = 1.8
) -> tuple[float, float, float]:
    if len(prices) < n:
        m = sum(prices) / len(prices) if prices else 0.0
        return m, m, m
    ma = sma(prices, n)[-1]
    dev = (sum((p - ma) ** 2 for p in prices[-n:]) / n) ** 0.5
    return ma, ma - mult * dev, ma + mult * dev


def support_resistance(
    prices: list[float], window: int = 20, eps: float = 0.015
) -> tuple[list[float], list[float]]:
    if len(prices) < window:
        return [], []
    seg = prices[-window:]
    m = min(seg)
    M = max(seg)
    s = []
    r = []
    for v in seg:
        if abs(v - m) / max(1e-9, m) < eps:
            s.append(v)
        if abs(v - M) / max(1e-9, M) < eps:
            r.append(v)
    return (
        sorted({round(x, 8) for x in s})[:5],
        sorted({round(x, 8) for x in r})[:5],
    )


# ───────────────────────────────────────────────────────────────────────────
# Screener / Trend / Pump
# ───────────────────────────────────────────────────────────────────────────
def trend_map(id_: str, vs: str = "usd") -> dict[str, str]:
    try:
        m5 = cg_chart(id_, days=1, vs=vs, interval="minute").get("prices", [])
        h1 = cg_chart(id_, days=7, vs=vs, interval="hourly").get("prices", [])
        d1 = cg_chart(id_, days=30, vs=vs, interval="daily").get("prices", [])

        def _trend(pr):
            vals = [p[1] for p in pr][-60:] or [0]
            if len(vals) < 5:
                return "neutral"
            e8 = ema(vals, 8)[-1]
            e21 = ema(vals, 21)[-1]
            if e8 > e21:
                return "up"
            if e8 < e21:
                return "down"
            return "neutral"

        return {"M5": _trend(m5), "H1": _trend(h1), "D1": _trend(d1)}
    except Exception:
        return {"M5": "neutral", "H1": "neutral", "D1": "neutral"}


def pump_signal(id_: str, vs: str = "usd") -> dict[str, Any]:
    ch1 = cg_chart(id_, days=1, vs=vs, interval="minute")
    pr1 = ch1.get("prices", [])
    vol1 = ch1.get("total_volumes", [])
    ch7 = cg_chart(id_, days=7, vs=vs, interval="hourly")
    pr7 = ch7.get("prices", [])
    vol7 = ch7.get("total_volumes", [])
    prices_1 = [p[1] for p in pr1]
    vols_1 = [v[1] for v in vol1]
    prices_7 = [p[1] for p in pr7]
    vols_7 = [v[1] for v in vol7]
    if len(prices_1) < 10 or len(prices_7) < 10:
        return {"score": 0, "pump_m1h": False, "pump_d1": False, "detail": {}}
    last = prices_1[-1]
    p60 = prices_1[-60] if len(prices_1) > 60 else prices_1[0]
    p1d = prices_1[0]
    ret60 = last / p60 - 1.0
    ret1d = last / p1d - 1.0
    v60 = sum(vols_1[-60:]) / max(1, len(vols_1[-60:]))
    vbase = sum(vols_7) / max(1, len(vols_7))
    v_ratio = v60 / max(1e-9, vbase)
    accel = impulse(prices_1, 8)
    pump_m1h = (ret60 > 0.045 and v_ratio > 1.3) or (accel > 0.04 and v_ratio > 1.2)
    pump_d1 = ret1d > 0.10 and v_ratio > 1.2
    score = int(
        max(
            0,
            min(
                100,
                62 * max(0.0, ret60)
                + 35 * max(0.0, ret1d)
                + 25 * max(0.0, accel)
                + 20 * max(0.0, v_ratio - 1.0),
            ),
        )
    )
    return {
        "score": score,
        "pump_m1h": bool(pump_m1h),
        "pump_d1": bool(pump_d1),
        "detail": {"ret60": ret60, "ret1d": ret1d, "v_ratio": v_ratio, "accel": accel},
    }


def screener_top(n: int = 30, vs: str = "usd") -> list[dict[str, Any]]:
    mk = cg_top_markets(vs=vs, n=n) or []
    out = []
    for co in mk:
        try:
            sp = (co.get("sparkline_in_7d") or {}).get("price") or []
            if len(sp) < 48:
                continue
            prices = sp[-168:] if len(sp) >= 168 else sp
            last = prices[-1]
            rsi14 = rsi(prices[-60:], 14)
            e8, e21 = ema(prices, 8)[-1], ema(prices, 21)[-1]
            trendH = "up" if e8 > e21 else ("down" if e8 < e21 else "neutral")
            vol48 = volatility(prices[-48:])
            risk = min(
                10, max(1, 4 + (1 if rsi14 > 70 else 0) + (1 if vol48 > 0.08 else 0))
            )
            last24 = prices[-24:]
            pump = pump_signal(co.get("id", ""), vs=vs)["score"]
            up_pot = int(max(0.0, (max(last24) / max(1e-9, last) - 1.0)) * 100)
            score = int((100 - risk * 7) + pump * 0.6 + up_pot * 0.5)
            out.append(
                {
                    "id": co.get("id") or "",
                    "symbol": (co.get("symbol") or "").upper(),
                    "name": co.get("name") or "",
                    "price": last,
                    "mcap": co.get("market_cap"),
                    "rsi14": rsi14,
                    "risk": risk,
                    "pump": pump,
                    "up_potential": up_pot,
                    "trend": {"H1": trendH},
                    "score": score,
                }
            )
        except Exception:
            continue
    out.sort(key=lambda x: x["score"], reverse=True)
    return out


# ───────────────────────────────────────────────────────────────────────────
# Analiza, makro i porady
# ───────────────────────────────────────────────────────────────────────────
def _binance_enrich(symbol_usdt: str | None) -> dict[str, Any]:
    if not symbol_usdt:
        return {"funding": None, "kline_5m": []}
    try:
        fund = binance_funding(symbol_usdt)
    except Exception:
        fund = None
    try:
        kl = binance_klines(symbol_usdt, interval="5m", limit=200)
    except Exception:
        kl = []
    return {"funding": fund, "kline_5m": kl}


def _macro_context() -> dict[str, Any]:
    fng = fng_index()
    try:
        gl = cg_global() or {}
    except Exception:
        gl = {}
    dom = ((gl.get("data") or {}).get("market_cap_percentage") or {}).get("btc")
    mktcap = ((gl.get("data") or {}).get("total_market_cap") or {}).get(VS_DEFAULT)
    return {"fng": fng, "btc_dominance": dom, "total_mcap": mktcap}


def scan_symbol(id_or_sym: str, vs: str = "usd") -> dict:
    id_ = resolve_symbol(id_or_sym)
    chart30 = cg_history(id_, days=30, vs=vs)
    prices = [p[1] for p in chart30.get("prices", [])]
    if not prices:
        return {"ok": False, "error": "no-prices"}
    mkall = cg_markets(vs=vs, ids=id_, sparkline=False) or [{}]
    mk = mkall[0] if mkall else {}
    last = prices[-1]
    rsi14 = rsi(prices, 14)
    vol30 = volatility(prices[-30:])
    imp = impulse(prices, 8)
    ema21 = ema(prices, 21)[-1]
    sma50 = sma(prices, 50)[-1] if len(prices) >= 50 else sum(prices) / len(prices)
    ma, lo_band, hi_band = fair_value_bands(prices, 20, 1.8)
    sup, res = support_resistance(prices, 20, 0.015)
    tmap = trend_map(id_, vs=vs)
    pump = pump_signal(id_, vs=vs)

    sym_usdt = derive_binance_symbol_from_cg(id_or_sym)
    bin_extra = _binance_enrich(sym_usdt)

    macro = _macro_context()

    risk = 5
    if vol30 > 0.09:
        risk += 1
    if pump["pump_m1h"] or pump["pump_d1"]:
        risk += 1
    if rsi14 < 32:
        risk -= 1
    if rsi14 > 68:
        risk += 1
    fng = macro["fng"]["value"]
    if fng >= 75:
        risk += 1
    if fng <= 25:
        risk -= 1
    fund = bin_extra["funding"]
    if fund is not None:
        if fund > 0.002:  # More sensitive threshold
            risk += 1
        if fund < -0.002:
            risk -= 1
    risk = max(1, min(10, risk))

    meta = {
        "id": id_,
        "symbol": (mk.get("symbol", "") or "").upper(),
        "name": mk.get("name", "") or id_,
        "price": last,
        "mcap": mk.get("market_cap"),
        "vol24h": mk.get("total_volume"),
        "rsi14": rsi14,
        "vol30": vol30,
        "impulse8": imp,
        "ema21": ema21,
        "sma50": sma50,
        "fv_mid": ma,
        "fv_lo": lo_band,
        "fv_hi": hi_band,
        "support": sup,
        "resistance": res,
        "trend": tmap,
        "pump": pump,
        "risk": risk,
        "binance": {"symbolUSDT": sym_usdt, "funding": fund},
        "macro": macro,
    }
    return {
        "ok": True,
        "meta": meta,
        "prices": prices,
        "binance_klines_5m": bin_extra["kline_5m"],
    }


# LLM
def _llm_chat(
    messages: list[dict[str, str]], temperature: float = 0.50, max_tokens: int = 1100
) -> str:
    # Try to use common_llm if available
    common_llm_available = False
    try:
        import importlib.util

        if importlib.util.find_spec("common_llm") is not None:
            common_llm_available = True
    except ImportError:
        pass

    if common_llm_available:
        try:
            from common_llm import llm_main  # type: ignore

            out = llm_main.chat(
                messages, temperature=temperature, max_tokens=max_tokens
            )  # type: ignore
            txt = (out.get("text") or "").strip()
            if txt:
                return txt
        except Exception as e:
            print(f"Error using common_llm: {e}")
    base = (config.LLM_BASE_URL or "https://api.deepinfra.com/v1/openai").rstrip(
        "/"
    )
    key = config.LLM_API_KEY.strip()
    model = (config.LLM_MODEL or 'Qwen/Qwen2.5-72B-Instruct').strip()
    if not key:
        return "\n".join(
            [m.get("content", "") for m in messages if m.get("role") == "user"]
        )
    try:
        url = base + "/chat/completions"
        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        r = HTTP.post(
            url,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=WEB_TIMEOUT,
        )
        r.raise_for_status()
        j = r.json()
        return (
            ((j.get("choices") or [{}])[0].get("message") or {})
            .get("content", "")
            .strip()
        )
    except Exception:
        return "\n".join(
            [m.get("content", "") for m in messages if m.get("role") == "user"]
        )


def portfolio_plan(meta: dict, horizon: str, risk_profile: str) -> dict:
    base_usd = float(os.getenv("PORTFOLIO_USD", "5000"))
    meta["risk"]
    pump = meta["pump"]["score"]
    rsi14 = meta["rsi14"]
    fng = ((meta.get("macro") or {}).get("fng") or {}).get("value", 50)
    fund = (meta.get("binance") or {}).get("funding")
    raw = 0.6 if risk_profile == "high" else (0.4 if risk_profile == "medium" else 0.25)
    raw *= 1.0 + max(0.0, min(0.3, (60 - abs(50 - rsi14)) / 300))
    raw *= 1.0 + (
        0.12 if (pump >= 60 and pump < 85) else (-0.12 if pump >= 85 else 0.0)
    )
    if isinstance(fund, float):
        raw *= 0.92 if fund > 0.02 else (1.06 if fund < -0.02 else 1.0)
    raw *= 0.93 if fng >= 75 else (1.05 if fng <= 25 else 1.0)
    pos_pct = max(0.04, min(0.28, raw))
    pos_usd = round(base_usd * pos_pct, 2)
    hard_sl = round(meta["fv_lo"] * 0.985, 8)
    trailing = 0.08 if risk_profile != "low" else 0.05
    return {
        "position_pct": int(pos_pct * 100),
        "position_usd": pos_usd,
        "hard_sl": hard_sl,
        "trailing": trailing,
    }


def advice_text(
    meta: dict, horizon: str, risk_profile: str, up_only_bias: bool = False
) -> str:
    plan = portfolio_plan(meta, horizon, risk_profile)
    rec = (
        "enter"
        if meta["trend"]["H1"] == "up" and meta["rsi14"] < 70
        else ("hold" if meta["trend"]["H1"] == "up" else "wait")
    )
    base = []
    base.append(f"### Sygnały ({meta['symbol']})")
    base.append(f"- Cena: {meta['price']:.6f}  | RSI14: {meta['rsi14']:.2f}")
    base.append(f"  Vol30: {meta['vol30']:.2%}  | Impulse8: {meta['impulse8']:.2%}")
    base.append(
        f"- EMA21: {meta['ema21']:.6f}  | SMA50: {meta['sma50']:.6f}"
        f"  | FV mid/lo/hi: {meta['fv_mid']:.6f} / {meta['fv_lo']:.6f} / {meta['fv_hi']:.6f}"
    )
    base.append(
        f"- Trend M5/H1/D1: {meta['trend']['M5']}/{meta['trend']['H1']}/{meta['trend']['D1']}"
    )
    base.append(
        f"- PUMP score: {meta['pump']['score']} "
        f"(m1h={meta['pump']['pump_m1h']}, d1={meta['pump']['pump_d1']})"
    )
    if meta["support"]:
        base.append(f"- Wsparcia ~ {', '.join([f'{x:.4f}' for x in meta['support']])}")
    if meta["resistance"]:
        base.append(f"- Opory ~ {', '.join([f'{x:.4f}' for x in meta['resistance']])}")
    fng = meta["macro"]["fng"]
    dom = meta["macro"]["btc_dominance"]
    base.append(
        f"- FNG: {fng['value']} ({fng['classification']})"
        + (f" | BTC.D: {dom:.2f}%" if dom is not None else "")
    )
    if meta["binance"]["funding"] is not None:
        base.append(f"- Funding (Binance): {meta['binance']['funding']:.4%}")
    base.append(f"- Risk score: {meta['risk']}/10 | Wstępna decyzja: **{rec}**")
    base.append(
        f"- Plan: pozycja ~ {plan['position_pct']}% (≈ {plan['position_usd']}$), "
        f"SL {plan['hard_sl']}, trailing {int(plan['trailing']*100)}%"
    )

    system = (
        "Jesteś doradcą krypto. Konkrety, poziomy, scenariusze bull/base/bear, "
        "invalidacje. Zero lania wody."
    )
    user = (
        "\n".join(base)
        + f"""

Horyzont: {horizon}, profil ryzyka: {risk_profile}.
Daj: alokacja, wejścia/wyjścia, TP/SL, zarządzanie pozycją, warunki unieważnienia, 
ryzyka zewnętrzne.
Zakończ TL;DR.
"""
    )
    txt = _llm_chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.45,
        max_tokens=1100,
    )
    return "\n".join(base) + "\n\n" + txt.strip()


def make_report(
    id_or_sym: str,
    horizon: str = "30d",
    risk_profile: str = "medium",
    up_only_bias: bool = False,
) -> dict[str, str]:
    sc = scan_symbol(id_or_sym)
    if not sc.get("ok"):
        path = _write_text(f"{_slug(id_or_sym)}_error.md", "# Brak danych z CoinGecko")
        return {"md": path}
    meta = sc["meta"]
    txt = f"# {meta['name']} ({meta['symbol']}) — Analiza & Porady\n\n" + advice_text(
        meta, horizon, risk_profile, up_only_bias
    )
    md = _write_text(f"{_slug(meta['symbol'])}_advice.md", txt)
    html_body = html.escape(txt).replace("\n", "<br/>\n")
    html_path = _write_text(
        f"{_slug(meta['symbol'])}_advice.html",
        (
            f"<html><meta charset='utf-8'><title>{meta['symbol']}</title>"
            f"<body>{html_body}</body></html>"
        ),
    )
    _persist_fact(
        f"[CRYPTO] report {meta['symbol']} risk={meta['risk']} "
        f"pump={meta['pump']['score']} fng={meta['macro']['fng']['value']}",
        tags=["crypto", "report"],
    )
    _psy_hint(
        f"crypto_report {meta['symbol']} risk={meta['risk']} "
        f"pump={meta['pump']['score']} fng={meta['macro']['fng']['value']}"
    )
    return {"md": md, "html": html_path}


# ───────────────────────────────────────────────────────────────────────────
# Memory / Psychika
# ───────────────────────────────────────────────────────────────────────────
def _mem():
    try:
        from memory import get_memory  # type: ignore

        return get_memory()
    except Exception:
        return None


def _persist_fact(
    text: str, tags: list[str] | None = None, conf: float = 0.6
) -> str | None:
    m = _mem()
    if not m:
        return None
    try:
        return m.add_fact(
            text.strip(), tags=sorted(set((tags or []) + ["crypto"])), conf=conf
        )
    except Exception:
        return None


def _psy_hint(hint: str) -> None:
    try:
        import psychika as _psy  # type: ignore

        if hasattr(_psy, "psyche_event"):
            _psy.psyche_event("crypto_signal", {"hint": hint})  # type: ignore
        elif hasattr(_psy, "autopilot_cycle"):
            _psy.autopilot_cycle(hint)  # type: ignore
    except Exception:
        return


# ───────────────────────────────────────────────────────────────────────────
# Watch / Whale / Alerts
# ───────────────────────────────────────────────────────────────────────────
def _alert(msg: str, tags: list[str], conf: float = 0.65) -> None:
    _persist_fact(f"[ALERT] {msg}", tags=["alert"] + tags, conf=conf)
    _psy_hint(msg)
    log_line("alert", {"msg": msg, "tags": tags})


def whale_watch(address: str, min_eth: float = 200.0) -> list[dict[str, Any]]:
    if not ETHERSCAN_KEY or not address:
        return []
    j = _es_call(
        {
            "module": "account",
            "action": "txlist",
            "address": address,
            "page": 1,
            "offset": 50,
            "sort": "desc",
        }
    )
    ev = []
    for tx in (j.get("result") or [])[:50]:
        try:
            val_eth = float(tx.get("value", "0")) / (10**18)
            if val_eth >= min_eth:
                ev.append(
                    {
                        "hash": tx.get("hash"),
                        "time": int(tx.get("timeStamp") or "0"),
                        "from": tx.get("from"),
                        "to": tx.get("to"),
                        "value_eth": val_eth,
                        "direction": (
                            "in"
                            if tx.get("to", "").lower() == address.lower()
                            else "out"
                        ),
                    }
                )
        except Exception:
            continue
    return ev


def portfolio_watch_loop(
    user_id: str, interval_sec: int = 300, threshold_pct: float = 5.0
):
    """Enhanced watch loop that monitors portfolio positions and triggers alerts.

    Args:
        user_id: Portfolio user ID to monitor
        interval_sec: Check interval in seconds
        threshold_pct: Alert threshold for position changes (%)
    """
    print(
        f"[portfolio-watch] start; user={user_id} every {interval_sec}s, threshold={threshold_pct}%"
    )

    while True:
        try:
            portfolio = load_portfolio(user_id)
            if not portfolio.positions:
                print(f"[portfolio-watch] {user_id}: No positions to monitor")
                time.sleep(interval_sec)
                continue

            # Scan current balances
            updated_portfolio = scan_portfolio_balances(portfolio)

            # Check for significant changes
            for pos in updated_portfolio.positions:
                old_pos = next(
                    (p for p in portfolio.positions if p.symbol == pos.symbol), None
                )
                if old_pos and abs(pos.price_change_24h or 0) >= threshold_pct:
                    direction = "📈" if (pos.price_change_24h or 0) > 0 else "📉"
                    _alert(
                        f"{direction} {pos.symbol}: {pos.price_change_24h:.2f}% (${pos.value_usd:.2f})",
                        tags=["portfolio", user_id, pos.symbol],
                        conf=0.8,
                    )

            # Check portfolio total performance
            if abs(updated_portfolio.performance_24h or 0) >= threshold_pct:
                direction = (
                    "🚀" if (updated_portfolio.performance_24h or 0) > 0 else "💥"
                )
                _alert(
                    f"{direction} Portfolio {user_id}: {updated_portfolio.performance_24h:.2f}% (${updated_portfolio.total_value_usd:.2f})",
                    tags=["portfolio", user_id, "total"],
                    conf=0.9,
                )

            # Save updated portfolio
            save_portfolio(updated_portfolio)

        except Exception as e:
            print(f"[portfolio-watch] Error: {e}")

        time.sleep(interval_sec)


def watch_loop(
    ids: list[str],
    interval_sec: int = 120,
    vs: str = "usd",
    whales: list[str] | None = None,
    erc20: str | None = None,
):
    alerts_path = OUT_DIR / "alerts.jsonl"
    print(f"[watch] start; ids={ids} every {interval_sec}s")
    while True:
        # Makro
        try:
            macro = _macro_context()
            fng_val = macro["fng"]["value"]
            if fng_val >= 85 or fng_val <= 15:
                _alert(
                    f"FNG ekstremum={fng_val}",
                    tags=["crypto", "macro", "fng"],
                    conf=0.7,
                )
        except Exception:
            pass
        # GAS alert
        try:
            gm = es_gas_band_alert()
            if gm:
                _alert(gm, tags=["crypto", "gas"], conf=0.6)
        except Exception:
            pass

        for id_ in ids:
            try:
                sym_usdt = derive_binance_symbol_from_cg(id_)
                funding = binance_funding(sym_usdt) if sym_usdt else None
                sig = pump_signal(id_, vs=vs)
                sc = scan_symbol(id_, vs=vs)
                meta = sc["meta"]
                line = {
                    "ts": _now_iso(),
                    "id": id_,
                    "sym": meta["symbol"],
                    "px": meta["price"],
                    "pump": sig["score"],
                    "rsi": meta["rsi14"],
                    "trend": meta["trend"],
                    "risk": meta["risk"],
                    "funding": funding,
                }
                with open(alerts_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(line) + "\n")
                if sig["score"] >= 75 or sig["pump_m1h"]:
                    _alert(
                        (
                            f"PUMP {meta['symbol']} • score={sig['score']} • "
                            f"trend {meta['trend']['M5']}/"
                            f"{meta['trend']['H1']}/{meta['trend']['D1']}"
                        ),
                        tags=["crypto", "pump"],
                        conf=0.7,
                    )
                if isinstance(funding, float) and abs(funding) >= 0.03:
                    _alert(
                        f"Funding anomalia {meta['symbol']} • {funding:.2%}",
                        tags=["crypto", "funding"],
                        conf=0.65,
                    )
            except Exception as e:
                log_line("error", {"where": "watch_symbol", "id": id_, "err": str(e)})

        if whales:
            for addr in whales:
                try:
                    wh = whale_watch(addr, min_eth=200.0)
                    if wh:
                        _alert(
                            (
                                f"Whale @{addr[:8]}...: {len(wh)} dużych tx "
                                f"(ostatnia {wh[0]['value_eth']} ETH {wh[0]['direction']})"
                            ),
                            tags=["crypto", "whale"],
                            conf=0.65,
                        )
                except Exception as e:
                    log_line("error", {"where": "whale", "err": str(e)})

        if erc20:
            try:
                txs = es_erc20_latest_transfers(erc20, limit=10)
                if txs:
                    _persist_fact(
                        f"[ERC20] last {len(txs)} transfers for {erc20}",
                        tags=["crypto", "erc20"],
                        conf=0.6,
                    )
            except Exception as e:
                log_line("error", {"where": "erc20", "err": str(e)})

        time.sleep(interval_sec)


# ───────────────────────────────────────────────────────────────────────────
# ENHANCED FEATURES - Must-Have Implementation
# ───────────────────────────────────────────────────────────────────────────

# ── Portfolio Tracker ─────────────────────────────────────────────────────────


@dataclass
class PortfolioPosition:
    """Pozycja w portfolio - token z adresem i balansem."""

    symbol: str
    contract_address: str = ""
    balance: float = 0.0
    value_usd: float = 0.0
    value_pln: float = 0.0
    price_usd: float = 0.0
    price_change_24h: float = 0.0
    last_updated: str = ""


@dataclass
class Portfolio:
    """Portfolio użytkownika z adresami i pozycjami."""

    user_id: str
    name: str
    addresses: list[str]  # adresy ETH/BSC/itp.
    positions: list[PortfolioPosition]
    total_value_usd: float = 0.0
    total_value_pln: float = 0.0
    last_scan: str = ""
    performance_24h: float = 0.0


def load_portfolio(user_id: str) -> Portfolio:
    """Załaduj portfolio użytkownika z pliku JSON."""
    portfolio_file = OUT_DIR / f"portfolio_{user_id}.json"

    if portfolio_file.exists():
        try:
            with open(portfolio_file, encoding="utf-8") as f:
                data = json.load(f)
            return Portfolio(**data)
        except Exception as e:
            print(f"Error loading portfolio: {e}")

    # Create new portfolio
    return Portfolio(
        user_id=user_id, name=f"Portfolio {user_id}", addresses=[], positions=[]
    )


def save_portfolio(portfolio: Portfolio) -> None:
    """Zapisz portfolio do pliku JSON."""
    portfolio_file = OUT_DIR / f"portfolio_{portfolio.user_id}.json"
    portfolio.last_scan = _now_iso()

    try:
        with open(portfolio_file, "w", encoding="utf-8") as f:
            json.dump(asdict(portfolio), f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving portfolio: {e}")


def scan_portfolio_balances(portfolio: Portfolio) -> Portfolio:
    """Skanuj balansy tokenów w portfolio używając Etherscan."""
    if not portfolio.addresses:
        return portfolio

    new_positions = []
    total_usd = 0.0

    # Get USD to PLN rate
    try:
        usd_pln_data = _get_json("https://api.exchangerate-api.com/v4/latest/USD")
        usd_to_pln = usd_pln_data.get("rates", {}).get("PLN", 4.0)
    except:
        usd_to_pln = 4.0  # fallback rate

    for address in portfolio.addresses:
        try:
            # ETH balance
            eth_balance = es_native_balance(address)
            if eth_balance and eth_balance > 0:
                eth_data = cg_simple_price(ids="ethereum")
                eth_price = eth_data.get("ethereum", {}).get("usd", 0.0)
                eth_value_usd = eth_balance * eth_price

                position = PortfolioPosition(
                    symbol="ETH",
                    contract_address="",
                    balance=eth_balance,
                    value_usd=eth_value_usd,
                    value_pln=eth_value_usd * usd_to_pln,
                    price_usd=eth_price,
                    price_change_24h=eth_data.get("ethereum", {}).get(
                        "usd_24h_change", 0.0
                    ),
                    last_updated=_now_iso(),
                )
                new_positions.append(position)
                total_usd += eth_value_usd

        except Exception as e:
            log_line(
                "error", {"where": "portfolio_scan", "address": address, "err": str(e)}
            )

    portfolio.positions = new_positions
    portfolio.total_value_usd = total_usd
    portfolio.total_value_pln = total_usd * usd_to_pln
    portfolio.last_scan = _now_iso()

    return portfolio


def portfolio_performance_report(portfolio: Portfolio) -> dict[str, Any]:
    """Generuj raport performance portfolio."""
    if not portfolio.positions:
        return {"error": "Portfolio is empty"}

    # Calculate metrics
    top_position = max(portfolio.positions, key=lambda p: p.value_usd, default=None)
    best_performer = max(
        portfolio.positions, key=lambda p: p.price_change_24h, default=None
    )
    worst_performer = min(
        portfolio.positions, key=lambda p: p.price_change_24h, default=None
    )

    report = {
        "portfolio_id": portfolio.user_id,
        "total_value": {
            "usd": portfolio.total_value_usd,
            "pln": portfolio.total_value_pln,
        },
        "positions_count": len(portfolio.positions),
        "top_position": {
            "symbol": top_position.symbol if top_position else "",
            "value_usd": top_position.value_usd if top_position else 0,
            "percentage": (
                (top_position.value_usd / portfolio.total_value_usd * 100)
                if top_position and portfolio.total_value_usd > 0
                else 0
            ),
        },
        "performance_24h": {
            "best": {
                "symbol": best_performer.symbol if best_performer else "",
                "change_pct": best_performer.price_change_24h if best_performer else 0,
            },
            "worst": {
                "symbol": worst_performer.symbol if worst_performer else "",
                "change_pct": (
                    worst_performer.price_change_24h if worst_performer else 0
                ),
            },
        },
        "last_scan": portfolio.last_scan,
        "addresses_tracked": len(portfolio.addresses),
    }

    return report


# ── Token Scoring System ──────────────────────────────────────────────────────
def calculate_token_score(token_id: str, vs: str = "usd") -> dict[str, Any]:
    """
    Oblicz composite score dla tokena na podstawie:
    - Płynności (volume, market cap)
    - Zaufania do kontraktu (age, verified, audits)
    - Wielkości społeczności (holders, social metrics)
    """
    try:
        # Basic data
        coin_data = cg_simple_price(
            ids=token_id, include_24hr_vol=True, include_market_cap=True
        )
        if token_id not in coin_data:
            return {"error": f"Token {token_id} not found"}

        token = coin_data[token_id]
        price = token.get(f"{vs}", 0)
        volume_24h = token.get(f"{vs}_24h_vol", 0)
        market_cap = token.get(f"{vs}_market_cap", 0)

        # Liquidity score (0-100)
        liquidity_score = 0
        if volume_24h > 0 and market_cap > 0:
            volume_to_mcap_ratio = volume_24h / market_cap
            if volume_to_mcap_ratio > 0.1:  # High liquidity
                liquidity_score = 90
            elif volume_to_mcap_ratio > 0.05:
                liquidity_score = 70
            elif volume_to_mcap_ratio > 0.01:
                liquidity_score = 50
            else:
                liquidity_score = 20

        # Contract trust score (simplified - in real implementation would check contract age, audits)
        trust_score = 50  # Base score
        if market_cap > 100_000_000:  # >100M mcap = more established
            trust_score += 30
        elif market_cap > 10_000_000:  # >10M mcap
            trust_score += 15

        # Community score (based on market cap as proxy for community size)
        community_score = 0
        if market_cap > 1_000_000_000:  # >1B = huge community
            community_score = 95
        elif market_cap > 100_000_000:  # >100M = large community
            community_score = 80
        elif market_cap > 10_000_000:  # >10M = medium community
            community_score = 60
        elif market_cap > 1_000_000:  # >1M = small community
            community_score = 35
        else:
            community_score = 10

        # Overall composite score
        composite_score = (
            liquidity_score * 0.4 + trust_score * 0.3 + community_score * 0.3
        )

        return {
            "token_id": token_id,
            "price": price,
            "market_cap": market_cap,
            "volume_24h": volume_24h,
            "scores": {
                "liquidity": liquidity_score,
                "trust": trust_score,
                "community": community_score,
                "composite": round(composite_score, 1),
            },
            "risk_level": (
                "low"
                if composite_score > 80
                else ("medium" if composite_score > 50 else "high")
            ),
            "last_updated": _now_iso(),
        }

    except Exception as e:
        return {"error": str(e)}


# ── Backtesting Engine ────────────────────────────────────────────────────────
@dataclass
class BacktestResult:
    """Wynik backtestingu strategii."""

    strategy_name: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_return_pct: float
    max_drawdown_pct: float
    trades_count: int
    win_rate_pct: float
    sharpe_ratio: float


def simple_sma_strategy_backtest(
    token_id: str,
    days: int = 30,
    short_period: int = 5,
    long_period: int = 20,
    initial_capital: float = 1000.0,
) -> BacktestResult:
    """
    Prosty backtest strategii SMA crossover.
    Buy gdy SMA(short) > SMA(long), sell gdy odwrotnie.
    """
    try:
        # Get historical data
        chart_data = cg_chart(token_id, days=days, vs="usd", interval="hour")
        if "prices" not in chart_data:
            raise ValueError("No price data available")

        prices = [float(p[1]) for p in chart_data["prices"]]
        if len(prices) < long_period:
            raise ValueError(f"Not enough data: {len(prices)} < {long_period}")

        # Calculate SMAs
        def sma(data: list[float], period: int) -> list[float]:
            return [
                sum(data[i - period : i]) / period for i in range(period, len(data) + 1)
            ]

        sma_short = sma(prices, short_period)
        sma_long = sma(prices, long_period)

        # Backtest simulation
        capital = initial_capital
        position = 0.0  # Amount of tokens held
        trades = []
        in_position = False

        for i in range(min(len(sma_short), len(sma_long))):
            price = prices[i + long_period - 1]  # Adjust for SMA lag

            # Buy signal: short SMA crosses above long SMA
            if not in_position and sma_short[i] > sma_long[i]:
                position = capital / price
                capital = 0.0
                in_position = True
                trades.append({"type": "buy", "price": price, "amount": position})

            # Sell signal: short SMA crosses below long SMA
            elif in_position and sma_short[i] < sma_long[i]:
                capital = position * price
                position = 0.0
                in_position = False
                trades.append({"type": "sell", "price": price, "amount": position})

        # Close final position if still holding
        if in_position:
            final_price = prices[-1]
            capital = position * final_price
            trades.append({"type": "sell", "price": final_price, "amount": position})

        # Calculate metrics
        total_return_pct = ((capital - initial_capital) / initial_capital) * 100
        win_trades = len(
            [
                t
                for i, t in enumerate(trades[::2])
                if i < len(trades[1::2]) and trades[1::2][i]["price"] > t["price"]
            ]
        )
        win_rate_pct = (win_trades / max(1, len(trades) // 2)) * 100

        # Simplified max drawdown (peak-to-trough)
        portfolio_values = []
        temp_capital = initial_capital
        temp_position = 0.0
        temp_in_position = False

        for i in range(min(len(sma_short), len(sma_long))):
            price = prices[i + long_period - 1]
            if not temp_in_position and sma_short[i] > sma_long[i]:
                temp_position = temp_capital / price
                temp_capital = 0.0
                temp_in_position = True
            elif temp_in_position and sma_short[i] < sma_long[i]:
                temp_capital = temp_position * price
                temp_position = 0.0
                temp_in_position = False

            current_value = temp_capital + (
                temp_position * price if temp_in_position else 0
            )
            portfolio_values.append(current_value)

        peak = portfolio_values[0]
        max_drawdown_pct = 0.0
        for value in portfolio_values:
            peak = max(peak, value)
            drawdown = ((peak - value) / peak) * 100
            max_drawdown_pct = max(max_drawdown_pct, drawdown)

        # Simple Sharpe ratio (assuming 0% risk-free rate)
        returns = [
            (portfolio_values[i] - portfolio_values[i - 1]) / portfolio_values[i - 1]
            for i in range(1, len(portfolio_values))
        ]
        avg_return = sum(returns) / len(returns) if returns else 0
        return_std = (
            (sum([(r - avg_return) ** 2 for r in returns]) / len(returns)) ** 0.5
            if returns
            else 1
        )
        sharpe_ratio = avg_return / return_std if return_std > 0 else 0

        return BacktestResult(
            strategy_name=f"SMA({short_period},{long_period})",
            start_date=_now_iso(),  # Simplified
            end_date=_now_iso(),
            initial_capital=initial_capital,
            final_capital=capital,
            total_return_pct=total_return_pct,
            max_drawdown_pct=max_drawdown_pct,
            trades_count=len(trades),
            win_rate_pct=win_rate_pct,
            sharpe_ratio=sharpe_ratio,
        )

    except Exception:
        # Return failed backtest
        return BacktestResult(
            strategy_name=f"SMA({short_period},{long_period})",
            start_date="",
            end_date="",
            initial_capital=initial_capital,
            final_capital=initial_capital,
            total_return_pct=0.0,
            max_drawdown_pct=0.0,
            trades_count=0,
            win_rate_pct=0.0,
            sharpe_ratio=0.0,
        )


# ───────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────
def _cli():
    import argparse

    ap = argparse.ArgumentParser(prog="crypto_advisor_full")
    sub = ap.add_subparsers(dest="cmd")

    s1 = sub.add_parser("screener")
    s1.add_argument("-n", type=int, default=30)
    s1.add_argument("--vs", default=VS_DEFAULT)
    s2 = sub.add_parser("analyze")
    s2.add_argument("id_or_sym")
    s2.add_argument("--vs", default=VS_DEFAULT)
    s3 = sub.add_parser("report")
    s3.add_argument("id_or_sym")
    s3.add_argument("--horizon", default="30d")
    s3.add_argument("--risk", default="medium")
    s4 = sub.add_parser("watch")
    s4.add_argument("--ids", required=True, help="comma separated CoinGecko ids")
    s4.add_argument("--vs", default=VS_DEFAULT)
    s4.add_argument("--interval", type=int, default=120)
    s5 = sub.add_parser("erc20")
    s5.add_argument("--contract", required=True)
    s5.add_argument("--limit", type=int, default=10)

    # New Must-Have features
    s6 = sub.add_parser("portfolio")
    s6.add_argument("--user", required=True, help="User ID for portfolio")
    s6.add_argument(
        "--action", choices=["scan", "report", "add_address", "watch"], default="report"
    )
    s6.add_argument("--address", help="ETH address to add to portfolio")
    s6.add_argument(
        "--interval", type=int, default=300, help="Watch interval in seconds"
    )
    s6.add_argument("--threshold", type=float, default=5.0, help="Alert threshold (%)")

    s7 = sub.add_parser("score")
    s7.add_argument("token_id", help="CoinGecko token ID")
    s7.add_argument("--vs", default="usd")

    s8 = sub.add_parser("backtest")
    s8.add_argument("token_id", help="CoinGecko token ID")
    s8.add_argument("--days", type=int, default=30)
    s8.add_argument("--short", type=int, default=5, help="Short SMA period")
    s8.add_argument("--long", type=int, default=20, help="Long SMA period")
    s8.add_argument("--capital", type=float, default=1000.0, help="Initial capital")

    args = ap.parse_args()
    if args.cmd == "screener":
        out = screener_top(n=args.n, vs=args.vs)
        path = _write_json("screener.json", out)
        print(path)
    elif args.cmd == "analyze":
        out = scan_symbol(args.id_or_sym, vs=args.vs)
        path = _write_json(f"{_slug(args.id_or_sym)}_scan.json", out)
        print(path)
    elif args.cmd == "report":
        paths = make_report(
            args.id_or_sym, horizon=args.horizon, risk_profile=args.risk
        )
        print(json.dumps(paths, ensure_ascii=False))
    elif args.cmd == "watch":
        ids = [x.strip() for x in args.ids.split(",") if x.strip()]
        watch_loop(ids, interval_sec=args.interval, vs=args.vs)
    elif args.cmd == "erc20":
        out = es_erc20_latest_transfers(args.contract, limit=args.limit)
        path = _write_json("erc20_transfers.json", out)
        print(path)
    elif args.cmd == "portfolio":
        portfolio = load_portfolio(args.user)
        if args.action == "add_address" and args.address:
            if args.address not in portfolio.addresses:
                portfolio.addresses.append(args.address)
                save_portfolio(portfolio)
                print(f"Added address {args.address} to portfolio {args.user}")
            else:
                print(f"Address {args.address} already in portfolio")
        elif args.action == "scan":
            portfolio = scan_portfolio_balances(portfolio)
            save_portfolio(portfolio)
            print(f"Portfolio scanned. Total value: ${portfolio.total_value_usd:.2f}")
        elif args.action == "watch":
            portfolio_watch_loop(args.user, args.interval, args.threshold)
        else:  # report
            report = portfolio_performance_report(portfolio)
            path = _write_json(f"portfolio_{args.user}_report.json", report)
            print(path)
    elif args.cmd == "score":
        score = calculate_token_score(args.token_id, vs=args.vs)
        path = _write_json(f"{_slug(args.token_id)}_score.json", score)
        print(path)
    elif args.cmd == "backtest":
        result = simple_sma_strategy_backtest(
            args.token_id,
            days=args.days,
            short_period=args.short,
            long_period=args.long,
            initial_capital=args.capital,
        )
        path = _write_json(f"{_slug(args.token_id)}_backtest.json", asdict(result))
        print(path)
    else:
        ap.print_help()


if __name__ == "__main__":
    _cli()
