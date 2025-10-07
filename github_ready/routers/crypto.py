from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

try:
    from src.crypto_advisor_full import (
        Portfolio,
        calculate_token_score,
        load_portfolio,
        make_report,
        portfolio_performance_report,
        save_portfolio,
        scan_portfolio_balances,
        screener_top,
        simple_sma_strategy_backtest,
        scan_symbol,
    )
except Exception:  # pragma: no cover - optional dependency
    Portfolio = None  # type: ignore
    calculate_token_score = None  # type: ignore
    load_portfolio = None  # type: ignore
    make_report = None  # type: ignore
    portfolio_performance_report = None  # type: ignore
    save_portfolio = None  # type: ignore
    scan_portfolio_balances = None  # type: ignore
    screener_top = None  # type: ignore
    simple_sma_strategy_backtest = None  # type: ignore
    scan_symbol = None  # type: ignore


router = APIRouter(prefix="/api/crypto", tags=["crypto"])


def _ensure_available() -> None:
    if screener_top is None:
        raise HTTPException(status_code=503, detail="Crypto engine not available")


class PortfolioAddress(BaseModel):
    address: str = Field(..., min_length=6)


@router.get("/health")
def crypto_health() -> Dict[str, Any]:
    if screener_top is None:
        raise HTTPException(status_code=503, detail="Crypto engine not available")
    return {"ok": True}


@router.get("/screener")
def crypto_screener(
    limit: int = Query(30, ge=1, le=200),
    vs: str = Query("usd", min_length=3, max_length=10),
) -> Dict[str, Any]:
    _ensure_available()
    try:
        items = screener_top(limit, vs)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Screener failed: {exc}")
    return {"items": items, "count": len(items)}


@router.get("/token/{token_id}")
def crypto_token(token_id: str, vs: str = Query("usd", min_length=3, max_length=10)) -> Dict[str, Any]:
    _ensure_available()
    try:
        data = scan_symbol(token_id, vs=vs)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Token scan failed: {exc}")
    return data


@router.get("/token/{token_id}/report")
def crypto_report(
    token_id: str,
    horizon: str = Query("30d"),
    risk: str = Query("medium"),
) -> Dict[str, Any]:
    _ensure_available()
    try:
        report = make_report(token_id, horizon=horizon, risk_profile=risk)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Report failed: {exc}")
    return report


@router.get("/token/{token_id}/score")
def crypto_score(token_id: str, vs: str = Query("usd")) -> Dict[str, Any]:
    _ensure_available()
    try:
        score = calculate_token_score(token_id, vs=vs)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Score failed: {exc}")
    return score


@router.get("/token/{token_id}/backtest")
def crypto_backtest(
    token_id: str,
    days: int = Query(90, ge=7, le=365),
    short: int = Query(5, ge=1, le=60),
    long: int = Query(20, ge=2, le=200),
    capital: float = Query(1000.0, ge=10.0),
) -> Dict[str, Any]:
    _ensure_available()
    try:
        result = simple_sma_strategy_backtest(
            token_id,
            days=days,
            short_period=short,
            long_period=long,
            initial_capital=capital,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Backtest failed: {exc}")
    return asdict(result)


@router.get("/portfolio/{user_id}")
def portfolio_get(user_id: str) -> Dict[str, Any]:
    _ensure_available()
    if load_portfolio is None:
        raise HTTPException(status_code=503, detail="Portfolio storage unavailable")
    portfolio = load_portfolio(user_id)
    return {"portfolio": asdict(portfolio)}


@router.post("/portfolio/{user_id}/addresses")
def portfolio_add_address(user_id: str, body: PortfolioAddress) -> Dict[str, Any]:
    _ensure_available()
    if load_portfolio is None or save_portfolio is None:
        raise HTTPException(status_code=503, detail="Portfolio storage unavailable")
    portfolio = load_portfolio(user_id)
    if body.address not in portfolio.addresses:
        portfolio.addresses.append(body.address)
        save_portfolio(portfolio)
    return {"portfolio": asdict(portfolio)}


@router.post("/portfolio/{user_id}/scan")
def portfolio_scan(user_id: str) -> Dict[str, Any]:
    _ensure_available()
    if load_portfolio is None or save_portfolio is None or scan_portfolio_balances is None:
        raise HTTPException(status_code=503, detail="Portfolio scanning unavailable")
    portfolio = load_portfolio(user_id)
    try:
        updated = scan_portfolio_balances(portfolio)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Portfolio scan failed: {exc}")
    save_portfolio(updated)
    return {"portfolio": asdict(updated)}


@router.get("/portfolio/{user_id}/report")
def portfolio_report(user_id: str) -> Dict[str, Any]:
    _ensure_available()
    if load_portfolio is None or portfolio_performance_report is None:
        raise HTTPException(status_code=503, detail="Portfolio reporting unavailable")
    portfolio = load_portfolio(user_id)
    try:
        report = portfolio_performance_report(portfolio)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Portfolio report failed: {exc}")
    return report