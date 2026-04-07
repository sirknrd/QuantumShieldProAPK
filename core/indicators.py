from __future__ import annotations

import numpy as np


def sma(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype="float64")
    out = np.full_like(x, np.nan)
    if n <= 0 or len(x) < n:
        return out
    c = np.cumsum(np.insert(x, 0, 0.0))
    out[n - 1 :] = (c[n:] - c[:-n]) / n
    return out


def ema(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype="float64")
    out = np.full_like(x, np.nan)
    if n <= 0 or len(x) < n:
        return out
    alpha = 2.0 / (n + 1.0)
    out[n - 1] = np.nanmean(x[:n])
    for i in range(n, len(x)):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def rsi(x: np.ndarray, n: int = 14) -> np.ndarray:
    x = np.asarray(x, dtype="float64")
    out = np.full_like(x, np.nan)
    if len(x) < n + 1:
        return out
    dx = np.diff(x)
    up = np.where(dx > 0, dx, 0.0)
    dn = np.where(dx < 0, -dx, 0.0)
    # Wilder smoothing
    avg_up = np.full(len(x), np.nan, dtype="float64")
    avg_dn = np.full(len(x), np.nan, dtype="float64")
    avg_up[n] = np.mean(up[:n])
    avg_dn[n] = np.mean(dn[:n])
    for i in range(n + 1, len(x)):
        avg_up[i] = (avg_up[i - 1] * (n - 1) + up[i - 1]) / n
        avg_dn[i] = (avg_dn[i - 1] * (n - 1) + dn[i - 1]) / n
    rs = avg_up / (avg_dn + 1e-12)
    out = 100.0 - (100.0 / (1.0 + rs))
    out[: n] = np.nan
    return out


def macd(x: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype="float64")
    ef = ema(x, fast)
    es = ema(x, slow)
    line = ef - es
    sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist


def true_range(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    h = np.asarray(h, dtype="float64")
    l = np.asarray(l, dtype="float64")
    c = np.asarray(c, dtype="float64")
    prev_c = np.roll(c, 1)
    prev_c[0] = np.nan
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    return tr


def atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, n: int = 14) -> np.ndarray:
    tr = true_range(h, l, c)
    out = np.full_like(tr, np.nan)
    if len(tr) < n + 1:
        return out
    out[n] = np.nanmean(tr[1 : n + 1])
    for i in range(n + 1, len(tr)):
        out[i] = (out[i - 1] * (n - 1) + tr[i]) / n
    out[:n] = np.nan
    return out


def adx(h: np.ndarray, l: np.ndarray, c: np.ndarray, n: int = 14) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h = np.asarray(h, dtype="float64")
    l = np.asarray(l, dtype="float64")
    c = np.asarray(c, dtype="float64")
    up = h - np.roll(h, 1)
    dn = np.roll(l, 1) - l
    up[0] = np.nan
    dn[0] = np.nan
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = true_range(h, l, c)

    # Wilder smoothing
    def wilder(a: np.ndarray, n_: int) -> np.ndarray:
        out = np.full_like(a, np.nan)
        out[n_] = np.nanmean(a[1 : n_ + 1])
        for i in range(n_ + 1, len(a)):
            out[i] = (out[i - 1] * (n_ - 1) + a[i]) / n_
        out[:n_] = np.nan
        return out

    tr_s = wilder(tr, n)
    p_s = wilder(plus_dm, n)
    m_s = wilder(minus_dm, n)
    di_p = 100.0 * (p_s / (tr_s + 1e-12))
    di_m = 100.0 * (m_s / (tr_s + 1e-12))
    dx = 100.0 * np.abs(di_p - di_m) / (di_p + di_m + 1e-12)
    adx_ = wilder(dx, n)
    return adx_, di_p, di_m


def bbands(x: np.ndarray, n: int = 20, k: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype="float64")
    mid = sma(x, n)
    # rolling std (simple)
    out_std = np.full_like(x, np.nan)
    if len(x) >= n:
        for i in range(n - 1, len(x)):
            out_std[i] = np.nanstd(x[i - n + 1 : i + 1], ddof=0)
    up = mid + k * out_std
    lo = mid - k * out_std
    # %B
    pb = (x - lo) / (up - lo + 1e-12)
    return lo, mid, up, pb

