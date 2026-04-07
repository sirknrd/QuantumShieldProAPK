from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import requests


@dataclass(frozen=True)
class Ohlcv:
    t: np.ndarray  # unix seconds
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    v: np.ndarray


class YahooError(RuntimeError):
    pass


def _to_f64(x: Any) -> np.ndarray:
    a = np.array(x, dtype="float64")
    return a


def fetch_ohlcv(symbol: str, range_: str = "1y", interval: str = "1d", timeout: float = 15.0) -> Ohlcv:
    """
    Yahoo Finance chart endpoint (JSON).
    Example:
      https://query2.finance.yahoo.com/v8/finance/chart/AAPL?range=1y&interval=1d
    """
    symbol = symbol.strip().upper()
    if not symbol:
        raise YahooError("Ticker vacío")

    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        "range": range_,
        "interval": interval,
        "includePrePost": "false",
        "events": "div,splits",
    }
    r = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    if r.status_code >= 400:
        raise YahooError(f"HTTP {r.status_code}")
    data = r.json()

    chart = (data or {}).get("chart", {})
    err = chart.get("error")
    if err:
        raise YahooError(str(err))

    res = (chart.get("result") or [])
    if not res:
        raise YahooError("Sin resultados")
    res0 = res[0]

    ts = res0.get("timestamp") or []
    ind = ((res0.get("indicators") or {}).get("quote") or [])
    if not ts or not ind:
        raise YahooError("Datos incompletos")
    q = ind[0]

    o = q.get("open") or []
    h = q.get("high") or []
    l = q.get("low") or []
    c = q.get("close") or []
    v = q.get("volume") or []

    t = np.array(ts, dtype="int64")
    o = _to_f64(o)
    h = _to_f64(h)
    l = _to_f64(l)
    c = _to_f64(c)
    v = _to_f64(v) if len(v) else np.zeros_like(c)

    # filter NaNs (Yahoo sometimes returns nulls)
    mask = ~(np.isnan(o) | np.isnan(h) | np.isnan(l) | np.isnan(c))
    t, o, h, l, c, v = t[mask], o[mask], h[mask], l[mask], c[mask], v[mask]
    if len(c) < 50:
        raise YahooError("Muy pocos datos")

    return Ohlcv(t=t, o=o, h=h, l=l, c=c, v=v)

