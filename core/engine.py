from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import indicators as ind
from .yahoo_chart import Ohlcv


@dataclass(frozen=True)
class Snapshot:
    close: float
    change_pct: float | None
    rsi14: float | None
    adx14: float | None
    atrp14: float | None
    relvol20: float | None


@dataclass(frozen=True)
class Recommendation:
    label: str  # COMPRA / VENTA / NEUTRAL
    score: float  # -100..100
    confidence: int  # 0..100
    color_hex: str
    regime: str


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def compute_all(ohlcv: Ohlcv) -> dict[str, np.ndarray]:
    c = ohlcv.c
    h = ohlcv.h
    l = ohlcv.l
    v = ohlcv.v

    ema50 = ind.ema(c, 50)
    ema200 = ind.ema(c, 200)
    rsi14 = ind.rsi(c, 14)
    macd_line, macd_sig, macd_hist = ind.macd(c)
    adx14, di_p, di_m = ind.adx(h, l, c, 14)
    atr14 = ind.atr(h, l, c, 14)
    bbl, bbm, bbu, bbp = ind.bbands(c, 20, 2.0)

    vol_sma20 = ind.sma(v, 20)
    rel_vol = v / (vol_sma20 + 1e-12)

    return {
        "EMA50": ema50,
        "EMA200": ema200,
        "RSI14": rsi14,
        "MACD_HIST": macd_hist,
        "ADX14": adx14,
        "DIp": di_p,
        "DIm": di_m,
        "ATR14": atr14,
        "BBP": bbp,
        "REL_VOL": rel_vol,
    }


def snapshot(ohlcv: Ohlcv, feats: dict[str, np.ndarray]) -> Snapshot:
    c = ohlcv.c
    close = float(c[-1])
    change_pct = None
    if len(c) >= 2 and c[-2] != 0:
        change_pct = float((c[-1] / c[-2] - 1.0) * 100.0)

    rsi14 = feats["RSI14"][-1]
    adx14 = feats["ADX14"][-1]
    atr14 = feats["ATR14"][-1]
    relv = feats["REL_VOL"][-1]

    rsi14_v = None if np.isnan(rsi14) else float(rsi14)
    adx14_v = None if np.isnan(adx14) else float(adx14)
    atrp_v = None
    if not np.isnan(atr14) and close != 0:
        atrp_v = float(atr14) / close * 100.0
    relv_v = None if np.isnan(relv) else float(relv)

    return Snapshot(
        close=close,
        change_pct=change_pct,
        rsi14=rsi14_v,
        adx14=adx14_v,
        atrp14=atrp_v,
        relvol20=relv_v,
    )


def recommend(ohlcv: Ohlcv, feats: dict[str, np.ndarray]) -> Recommendation:
    c = ohlcv.c
    close = float(c[-1])

    ema50 = feats["EMA50"][-1]
    ema200 = feats["EMA200"][-1]
    rsi14 = feats["RSI14"][-1]
    macd_hist = feats["MACD_HIST"][-1]
    adx14 = feats["ADX14"][-1]
    di_p = feats["DIp"][-1]
    di_m = feats["DIm"][-1]
    bbp = feats["BBP"][-1]
    relv = feats["REL_VOL"][-1]

    adx_v = 0.0 if np.isnan(adx14) else float(adx14)
    trending = adx_v >= 25.0
    if adx_v >= 35.0:
        regime = "Tendencia fuerte"
    elif adx_v >= 25.0:
        regime = "Tendencia"
    elif adx_v >= 15.0:
        regime = "Mixto"
    else:
        regime = "Rango"

    def s_price_vs(a: float) -> float:
        if np.isnan(a):
            return 0.0
        return 1.0 if close > float(a) else -1.0

    trend = np.mean(
        [
            s_price_vs(ema200),
            (1.0 if (not np.isnan(ema50) and not np.isnan(ema200) and float(ema50) > float(ema200)) else (-1.0 if (not np.isnan(ema50) and not np.isnan(ema200)) else 0.0)),
            s_price_vs(ema50),
        ]
    )

    mom = np.mean(
        [
            0.0 if np.isnan(rsi14) else _clamp((float(rsi14) - 50.0) / 20.0, -1.0, 1.0),
            0.0 if np.isnan(macd_hist) else float(np.tanh(float(macd_hist) * 5.0)),
        ]
    )

    vol = np.mean(
        [
            0.0 if np.isnan(bbp) else _clamp((float(bbp) - 0.5) * 2.0, -1.0, 1.0),
        ]
    )

    volume = 0.0 if np.isnan(relv) else _clamp((float(relv) - 1.0) / 1.0, -0.5, 0.8)

    if trending:
        w = {"trend": 0.45, "mom": 0.30, "vol": 0.10, "volume": 0.15}
    else:
        w = {"trend": 0.30, "mom": 0.35, "vol": 0.20, "volume": 0.15}

    raw = w["trend"] * trend + w["mom"] * mom + w["vol"] * vol + w["volume"] * volume
    score = _clamp(raw * 100.0, -100.0, 100.0)

    # Confidence: alignment + regime strength
    alignment = float(np.mean([abs(trend), abs(mom), abs(vol), abs(volume)]))
    regime_strength = _clamp((adx_v - 10.0) / 25.0, 0.0, 1.0)
    confidence = int(round(_clamp((0.55 * alignment + 0.45 * regime_strength) * 100.0, 0.0, 100.0)))

    if score >= 60:
        label, color = "COMPRA FUERTE", "#00D18F"
    elif score >= 20:
        label, color = "COMPRA", "#2F81F7"
    elif score <= -60:
        label, color = "VENTA FUERTE", "#FF4B4B"
    elif score <= -20:
        label, color = "VENTA", "#FFA657"
    else:
        label, color = "NEUTRAL", "#8B949E"

    # Add a DI confirmation to confidence (small bump)
    if adx_v >= 25 and not (np.isnan(di_p) or np.isnan(di_m)):
        if (score > 0 and float(di_p) > float(di_m)) or (score < 0 and float(di_m) > float(di_p)):
            confidence = int(_clamp(confidence + 5, 0, 100))

    return Recommendation(label=label, score=float(score), confidence=confidence, color_hex=color, regime=regime)


def consensus_bucket(net: int) -> str:
    if net >= 3:
        return "COMPRA FUERTE"
    if net >= 1:
        return "COMPRA"
    if net <= -3:
        return "VENTA FUERTE"
    if net <= -1:
        return "VENTA"
    return "NEUTRAL"


def technical_signals(close: float, feats: dict[str, np.ndarray]) -> tuple[dict[str, int], dict[str, int]]:
    """Investing-like: MA and oscillators signals in {-1,0,1}."""
    ema20 = feats.get("EMA20")
    ema50 = feats.get("EMA50")
    ema200 = feats.get("EMA200")
    rsi14 = feats.get("RSI14")
    macd_hist = feats.get("MACD_HIST")
    adx14 = feats.get("ADX14")
    di_p = feats.get("DIp")
    di_m = feats.get("DIm")
    bbp = feats.get("BBP")

    ma: dict[str, int] = {}
    osc: dict[str, int] = {}

    # We only have EMA50/EMA200 in compute_all; optional EMA20 if caller provides.
    def _last(a: np.ndarray | None) -> float:
        if a is None or len(a) == 0:
            return float("nan")
        return float(a[-1])

    e50 = _last(ema50)
    e200 = _last(ema200)
    e20 = _last(ema20) if ema20 is not None else float("nan")

    ma["Precio vs EMA50"] = 0 if np.isnan(e50) else (1 if close > e50 else -1)
    ma["Precio vs EMA200"] = 0 if np.isnan(e200) else (1 if close > e200 else -1)
    ma["EMA50 vs EMA200"] = 0 if (np.isnan(e50) or np.isnan(e200)) else (1 if e50 > e200 else -1)
    ma["Precio vs EMA20"] = 0 if np.isnan(e20) else (1 if close > e20 else -1)

    r = _last(rsi14)
    osc["RSI14"] = 0 if np.isnan(r) else (1 if r > 55 else (-1 if r < 45 else 0))
    mh = _last(macd_hist)
    osc["MACD Hist"] = 0 if np.isnan(mh) else (1 if mh > 0 else (-1 if mh < 0 else 0))

    ax = _last(adx14)
    dp = _last(di_p)
    dm = _last(di_m)
    osc["DI+ vs DI-"] = 0 if (np.isnan(ax) or ax < 25 or np.isnan(dp) or np.isnan(dm)) else (1 if dp > dm else -1)

    b = _last(bbp)
    osc["Bollinger %B"] = 0 if np.isnan(b) else (1 if b < 0.2 else (-1 if b > 0.8 else 0))

    return ma, osc


def consensus(ma: dict[str, int], osc: dict[str, int]) -> dict[str, object]:
    ma_vals = list(ma.values())
    osc_vals = list(osc.values())
    all_vals = ma_vals + osc_vals

    b_ma = sum(1 for v in ma_vals if v > 0)
    s_ma = sum(1 for v in ma_vals if v < 0)
    b_osc = sum(1 for v in osc_vals if v > 0)
    s_osc = sum(1 for v in osc_vals if v < 0)
    net = (b_ma + b_osc) - (s_ma + s_osc)
    return {
        "Consenso": consensus_bucket(net),
        "Net": net,
        "MA Compra": b_ma,
        "MA Venta": s_ma,
        "Osc Compra": b_osc,
        "Osc Venta": s_osc,
        "Total": len(all_vals),
    }

